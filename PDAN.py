import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import copy



class TokenSummarizationMHA(nn.Module):
    def __init__(self, num_tokens, dim=256, num_heads=8, dropout=0.1):
        super(TokenSummarizationMHA, self).__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.tokens = nn.Parameter(torch.randn(1, self.num_tokens, self.dim) * 0.02)


    def forward(self, v):
        v = torch.permute(v, (0, 2, 1)) # permute from (dim, T) to (T, dim)
        bs, t, d = v.shape
        tokens = self.tokens.expand(bs, -1, -1)
        attn_output, _ = self.attn(query=tokens, key=v, value=v)

        return attn_output

class PDAN(nn.Module):
    def __init__(self, num_stages=1, num_layers=5, num_f_maps=512, dim=1024, num_classes=157, num_summary_tokens=10,
                        cross_attention_init='zeros'):
        super(PDAN, self).__init__()
        self.stage1 = SSPDAN(num_layers, num_f_maps, dim, num_classes, num_summary_tokens, cross_attention_init)
        self.stages = nn.ModuleList([copy.deepcopy(SSPDAN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])


    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(out * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class SSPDAN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, num_summary_tokens, cross_attention_init):
        super(SSPDAN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(PDAN_Block(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        self.summarization_module = None
        self.summary = None

        self.stage1_bottleneck = torch.nn.Conv1d(in_channels=dim, out_channels=num_f_maps, kernel_size=1)
        if num_summary_tokens:
            self.num_summary_tokens = num_summary_tokens
            self.summarization_module = TokenSummarizationMHA(num_tokens=num_summary_tokens, dim=num_f_maps,
                                                              num_heads=4)

            self.cross_attention = nn.MultiheadAttention(num_f_maps, 4, bias=False,  batch_first=True)

        self.init_cross_attention(cross_attention_init)


    def init_cross_attention(self, mode):
        if mode == 'zeros':
            init.zeros_(self.cross_attention.in_proj_weight)
        if mode == 'kaiming':
            init.kaiming_normal(self.cross_attention.in_proj_weight, mode='fan_out')

        if self.cross_attention.in_proj_bias is not None:
            init.zeros_(self.cross_attention.in_proj_weights)



    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for idx, layer in enumerate(self.layers):
            if self.summarization_module:
                if idx == 0:
                    self.summary = self.summarization_module(out)
                else:
                    self.summary += self.summarization_module(out)
            out = layer(out, mask)

        self.summary = self.summary / len(self.layers)
        #  apply cross attention
        res = self.cross_attention(query=out.permute(0, 2, 1), key=self.summary, value=self.summary)[0]
        #res = self.layer_norm(res)
        res = res.permute(0, 2, 1) + out
        out = res
        out = self.conv_out(out) * mask[:, 0:1, :]

        return out

    def forward_(self, x, mask):
        out = self.conv_1x1(x)
        for idx, layer in enumerate(self.layers):
            prev_input = out
            out = layer(out, mask)

            if self.summarization_module:
                #  apply cross attention
                summary = self.summarization_module(prev_input)
                res = self.cross_attention(query=out.permute(0, 2, 1), key=summary, value=summary)[0]
                res = self.layer_norm(res)
                res = res.permute(0, 2, 1) + out
                out = res

        out = self.conv_out(out) * mask[:, 0:1, :]


        return out


class PDAN_Block(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(PDAN_Block, self).__init__()
        self.conv_attention=DAL(in_channels, out_channels, kernel_size=3, padding=dilation, dilated=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, mask):
        out = F.relu(self.conv_attention(x))
        out = self.conv_1x1(out)
        return (x + out) * mask[:, 0:1, :]

class DAL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilated=1, groups=1, bias=False,
                 num_heads_cross_attention=1):
        super(DAL, self).__init__()
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilated = dilated
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        self.rel_t = nn.Parameter(torch.randn(out_channels, 1, kernel_size), requires_grad=True)
        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        # self.kernal_size = 2 * self.dilated + 1
        # self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernal_size, bias=bias, padding = self.kernal_size//2)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()


    def forward_initial(self, x):
        batch, channels, time = x.size()
        padded_x = F.pad(x, (self.padding, self.padding))
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        kernal_size = 2*self.dilated + 1
        k_out = k_out.unfold(2, kernal_size, self.stride)  # unfold(dim, size, step)
        k_out=torch.cat((k_out[:,:,:,0].unsqueeze(3),k_out[:,:,:,0+self.dilated].unsqueeze(3),k_out[:,:,:,0+2*self.dilated].unsqueeze(3)),dim=3)  #dilated
        v_out = v_out.unfold(2, kernal_size, self.stride)
        v_out=torch.cat((v_out[:,:,:,0].unsqueeze(3),v_out[:,:,:,0+self.dilated].unsqueeze(3),v_out[:,:,:,0+2*self.dilated].unsqueeze(3)),dim=3)  #dilated
        v_out = v_out + self.rel_t
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, time, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnctk,bnctk -> bnct', out, v_out).view(batch, -1, time)
        return out

    def forward(self, x):
        return self.forward_initial(x)

    def forward_summary(self, x, summary=None):
        batch, channels, time = x.size()
        padded_x = F.pad(x, (self.padding, self.padding))

        kernal_size = 2 * self.dilated + 1
        #
        if summary is not None:
            padded_x = padded_x.unfold(2, kernal_size, self.stride)
            padded_x = torch.cat((padded_x[:, :, :, 0].unsqueeze(3), padded_x[:, :, :, 0 + self.dilated].unsqueeze(3),
                               padded_x[:, :, :, 0 + 2 * self.dilated].unsqueeze(3)), dim=3)  # dilated
            # padded_x: bs, dim, unfold, ks
            bs, dim, unfold_t, ks = padded_x.shape
            summary_expanded = summary.unsqueeze(2).repeat(1, 1, unfold_t, 1)
            summary_expanded = summary_expanded.view(bs*unfold_t, -1, summary.shape[-1])
            padded_x = padded_x.view(bs*unfold_t, dim, ks)
            padded_x = torch.permute(padded_x, (0, 2, 1))

            skip = padded_x
            padded_x, _ = self.cross_attention(query=padded_x, key=summary_expanded, value=summary_expanded)
            padded_x += skip

            padded_x = padded_x.permute((0, 2, 1))

            k_out = self.key_conv(padded_x)
            v_out = self.value_conv(padded_x)

            k_out = k_out.permute((0, 2, 1))
            v_out = v_out.permute((0, 2, 1))
            v_out = v_out.reshape(bs, dim, unfold_t, ks).contiguous()
            k_out = k_out.reshape(bs, dim, unfold_t, ks).contiguous()

        q_out = self.query_conv(x)

        v_out = v_out + self.rel_t
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, time, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnctk,bnctk -> bnct', out, v_out).view(batch, -1, time)
        return out

    def reset_parameters(self):
        init.kaiming_normal(self.key_conv.weight, mode='fan_out')
        init.kaiming_normal(self.value_conv.weight, mode='fan_out')
        init.kaiming_normal(self.query_conv.weight, mode='fan_out')
        init.normal(self.rel_t, 0, 1)



