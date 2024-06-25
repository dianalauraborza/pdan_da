from __future__ import division
import time
import os
import argparse
from collections import OrderedDict
import sys
import wandb
import torchvision.models as models
import torch
import config


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default=None)
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-model', type=str, default='PDAN')
parser.add_argument('-APtype', type=str, default='map')
parser.add_argument('-randomseed', type=str, default='False')
parser.add_argument('-load_model', type=str, default=False)
parser.add_argument('-batch_size', type=str, default=6)
parser.add_argument('-num_channel', type=str, default=128)
parser.add_argument('-num_summary_tokens', type=int, default=10)
parser.add_argument('-run_mode', type=str, default='False')
parser.add_argument('-feat', type=str, default='False')


args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# set random seed
if args.randomseed=="False":
    SEED = 0
elif args.randomseed=="True":
    SEED = random.randint(1, 100000)
else:
    SEED = int(args.randomseed)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED!!!:', SEED)

from torch.optim import lr_scheduler
from torch.autograd import Variable

import json

import pickle
import math

from apmeter import APMeter

best_map = -1
batch_size = int(args.batch_size)


if args.dataset == 'charades':
    from charades_i3d_per_video import MultiThumos as Dataset
    from charades_i3d_per_video import mt_collate_fn as collate_fn
    if args.run_mode == 'debug':
        print('debug!!!')
        train_split = './data/charades_test.json'
        test_split = './data/charades_test.json'
    else:
        train_split = './data/charades.json'
        test_split = './data/charades.json'
    # print('load feature from:', args.rgb_root)
    rgb_root = args.rgb_root#'/user/dborza/home/mynef/data/stars/user/rdai/PhD_work/cvpr2020/Charades_v1/charades_feat_rgb'
    skeleton_root = '/Path/to/charades_feat_pose'
    flow_root = '/Path/to/charades_feat_flow'
    rgb_of=[rgb_root,flow_root]
    classes = 157

print('rgb root is ', rgb_root)

def load_data(train_split, val_split, root):
    # Load Data
    print('load data', root)
    if len(train_split) > 0:
        if str(args.feat) == '2d':
            dataset = Dataset(train_split, 'training', root, batch_size, classes, int(args.pool_step))
        else:
            dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    if str(args.feat) == '2d':
        val_dataset = Dataset(val_split, 'testing', root, batch_size, classes, int(args.pool_step))
    else:
        val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    global best_map
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_map, train_loss, last_lr = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch)
            # lr_sched.step(val_loss)
            print('---->', train_map, train_map.numpy())
            wandb.log({
                'train_map': train_map.numpy(),
                'train_loss': train_loss,
                'val_map': val_map.numpy(),
                'val_loss': val_loss,
                'last_lr': last_lr
            })

            if best_map > val_map:
                best_map = val_map
                weights_path = f'./checkpoints/model{epoch}.pth'
                torch.save(model.state_dict(), weights_path)
                print(f'Saved weights to {weights_path}')

            probs.append(prob_val)
            sched.step(val_loss)


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, :int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)
    outputs_final = activation
    outputs_final = outputs_final[-1]
    outputs_final = outputs_final.permute(0, 2, 1)  

    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)

    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)  

    loss = loss_f 

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
    if args.APtype == 'wap':
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    print('train-map:', train_map)


    apm.reset()

    epoch_loss = tot_loss / num_iter

    last_lr = optimizer.param_groups[0]['lr']

    return train_map, epoch_loss, last_lr


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print('val-map:', val_map)
    print(100 * apm.value())
    apm.reset()

    return full_probs, epoch_loss, val_map


def load_weights_from_pretrained(old_model_path, new_model):
    loaded_model = torch.load(old_model_path,  map_location=torch.device('cpu'))
    loaded_state_dict = loaded_model.module.state_dict()
    new_state_dict = new_model.state_dict()

    for name, param in loaded_state_dict.items():
        if name in new_state_dict:
            print('\t --- load ', name)
            try:
                new_state_dict[name].copy_(param)
            except RuntimeError:
                print(f'failed to load {name}', file=sys.stderr)

    new_model.load_state_dict(new_state_dict)

def run_sweep():
    with wandb.init() as run:
        learning_rate = run.config.learning_rate
        learning_rate_decay = run.config.learning_rate_decay
        num_epochs = run.config.epochs

        optimizer = optim.Adam(rgb_model.parameters(), lr=learning_rate, weight_decay=1e-6)  # weight_decay=1e-6
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)
        print('learning rate ', learning_rate)
        print('learning rate decay', learning_rate_decay)

        # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10, verbose=True)
        run([(rgb_model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion,
            num_epochs=num_epochs)

if __name__ == '__main__':
    print(str(args.model))
    print('batch_size:', batch_size)
    print('cuda_avail', torch.cuda.is_available())


    wandb.login(key=config.WANDB_KEY)

    config_dict = dict()


    if args.mode == 'flow':
        print('flow mode', flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'skeleton':
        print('Pose mode', skeleton_root)
        dataloaders, datasets = load_data(train_split, test_split, skeleton_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    if args.train:
        num_channel = args.num_channel
        if args.mode == 'skeleton':
            input_channnel = 256
        else:
            input_channnel = 1024

        num_classes = classes
        mid_channel=int(args.num_channel)

        if args.model=="PDAN":
            print("you are processing PDAN_original")
            from PDAN import PDAN
            # rgb_model = Net(mid_channel, input_channnel, classes)
            stage=1
            block=5
            num_channel=512
            input_channnel=1024
            num_classes=classes
            rgb_model = PDAN(stage, block, num_channel, input_channnel, num_classes, num_summary_tokens=int(args.num_summary_tokens))
            pytorch_total_params = sum(p.numel() for p in rgb_model.parameters() if p.requires_grad)
            print('pytorch_total_params', pytorch_total_params)
            #exit()
            print ('stage:', stage, 'block:', block, 'num_channel:', num_channel, 'input_channnel:', input_channnel,
                   'num_classes:', num_classes)

        # checkpoint = torch.load(str(args.load_model), map_location=torch.device('cpu'))

#        if args.load_model != "False":
#            # rgb_model = torch.load(str(args.load_model),  map_location=torch.device('cpu')).module
#            # rgb_model.to('cpu')
#            load_weights_from_pretrained(args.load_model, rgb_model)

        # rgb_model = torch.nn.DataParallel(rgb_model)

        pytorch_total_params = sum(p.numel() for p in rgb_model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('num_channel:', num_channel, 'input_channnel:', input_channnel,'num_classes:', num_classes)
        rgb_model.cuda()

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        print(lr)
        # optimizer = optim.Adam(rgb_model.parameters(), lr=lr, weight_decay=1e-6)#weight_decay=1e-6
        # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10, verbose=True)

        config_dict['lr'] = lr
        config_dict['num_classes'] = num_classes
        config_dict['dataset'] = args.dataset
        config_dict['epochs'] = args.epoch
        config_dict['num_summary_tokens'] = args.num_summary_tokens
        config_dict['pretrained_model'] = args.load_model

        sweep_config = {
            "program": "train.py",
            "method": "grid",
            "metric": {
                "name": "val_map",
                "goal": "maximize"
            },
            "parameters": {
                "lr": {
                    "values": [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
                },
                "lr_decay": {
                    "values": [1, 0.1, 0.01, 0.001, 0.0001]
                },
                "epochs": {
                    "values": [100]
                }
            }
        }

        # wandb.init(
        #     project=config.PROJECT_NAME,
        #     config=config_dict
        # )
        wandb.init(project=config.PROJECT_NAME)
        sweep_id = wandb.sweep(sweep_config)

        # Start a sweep agent
        wandb.agent(sweep_id, function=run_sweep)

        # run([(rgb_model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion,
        #     num_epochs=int(args.epoch))


