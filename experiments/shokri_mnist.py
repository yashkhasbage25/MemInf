import os
import torch
import argparse
import numpy as np
import pprint as pp
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from easydict import EasyDict
from dataset_utils import *

# --------------------------------------------- #
#                 dataset utils                 #
# --------------------------------------------- #
class TypicalDataset(nn.Module):

    def __init__(self, x, y, num_classes):

        self.x = x
        self.y = y
        self.num_classes = num_classes

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        img, target = self.x[index], self.y[index]
        img = img.float().unsqueeze(0)

        return img, target

# --------------------------------------------- #
#                   networks                    #
# --------------------------------------------- #

class TargetNet(nn.Module):
    def __init__(self, num_classes=10):
        super(TargetNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, smax=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        if smax:
            output = F.softmax(output, 1)
        return output

class ShadowNet(nn.Module):

    def __init__(self, im_size=28, num_classes=10):
        super(ShadowNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(im_size * im_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, smax=False):

        out = x.view(x.size(0), -1)
        out = self.net(out)
        if smax:
            out = F.softmax(out, 1)
        return out


class AttackNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AttackNet, self).__init__()

        hidden_dim = 500
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x, smax=False):

        out = x.view(x.size(0), -1)
        out = self.net(out)
        if smax:
            out = F.softmax(out, 1)
        return out

# ------------------------------------------- #
#             transform datasets              #
# ------------------------------------------- #

def get_target_dataset(x, 
        y, 
        num_target_train_size, 
        num_target_test_size,
        total_size
    ):

    perm = torch.randperm(total_size)
    
    relevant_indices = perm[:num_target_train_size]

    target_train_x = x[relevant_indices]
    target_train_y = y[relevant_indices]

    relevant_indices = perm[:num_target_test_size]

    target_test_x = x[relevant_indices]
    target_test_y = torch.cat((torch.ones(num_target_train_size), torch.zeros(num_target_test_size - num_target_train_size)))

    print('target dataset details:')
    print('target_train_x:', target_train_x.shape)
    print('target_train_y:', target_train_y.shape)
    print('target_test_x:', target_test_x.shape)
    print('target_test_y:', target_test_y.shape)
    
    return {
        'train_x': target_train_x,
        'train_y': target_train_y,
        'test_x': target_test_x,
        'test_y': target_test_y
    }

def get_shadow_datasets(x, 
        y,
        num_shadow_nets,
        num_shadow_train_size,
        num_shadow_test_size,
        full_dataset_size
    ):

    shadow_datasets = list()

    for _ in range(num_shadow_nets):
        shuffled_indices = torch.randperm(full_dataset_size)
        shadow_dataset = {
            'train_x': x[shuffled_indices[:num_shadow_train_size]],
            'test_x': x[shuffled_indices[num_shadow_train_size:num_shadow_train_size+num_shadow_test_size]],
            'train_y': y[shuffled_indices[:num_shadow_train_size]],
            'test_y': y[shuffled_indices[num_shadow_train_size:num_shadow_train_size+num_shadow_test_size]]
        }
        shadow_datasets.append(shadow_dataset)

    print('shadow_datasets[0] summary:')
    print('train_x:', shadow_datasets[0]['train_x'].shape)
    print('train_y:', shadow_datasets[0]['train_y'].shape)
    print('test_x:', shadow_datasets[0]['test_x'].shape)
    print('test_y:', shadow_datasets[0]['test_y'].shape)

    return shadow_datasets

def prepare_attack_dataset(attack_datasets, num_classes, num_shadow_nets):

    classwise_attack_dataset = [{'x': None, 'y': None} for _ in range(num_classes)]
    print('attack dataset summary:')
    with torch.no_grad():
        for class_idx in range(num_classes):

            x, y = list(), list()
            for shadow_idx in range(num_shadow_nets):
                attack_dataset = attack_datasets[shadow_idx]
                # train set
                valid_indices = (attack_dataset['train_truth'] == class_idx)
                train_x = attack_dataset['train_probs'][valid_indices]
                train_y = torch.ones((sum(valid_indices),), dtype=torch.int64)
                # test set
                valid_indices = (attack_dataset['test_truth'] == class_idx)
                test_x = attack_dataset['test_probs'][valid_indices]
                test_y = torch.zeros((sum(valid_indices),), dtype=torch.int64)

                x.append(train_x)
                x.append(test_x)
                y.append(train_y)
                y.append(test_y)

            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)

            print('class', class_idx, 'x:', x.shape, 'y:', y.shape)

            classwise_attack_dataset[class_idx]['x'] = x.detach().cpu()
            classwise_attack_dataset[class_idx]['y'] = y.detach().cpu()


    return classwise_attack_dataset

# --------------------------------------------- #
#                  train loops                  #
# --------------------------------------------- #

def train(model, 
        x, y, 
        lr,
        criterion, 
        device, 
        num_epochs,
        num_classes,
        cfg
    ):

    dataloader = data.DataLoader(TypicalDataset(x, y, num_classes), batch_size=cfg.batch_size, shuffle=True, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.l2)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(num_epochs):
        num_samples = 0
        running_corrects = 0.0
        with torch.enable_grad():
            model.train()
            for batch, truth in dataloader:
                batch = batch.to(device)
                truth = truth.to(device)
                num_samples += truth.numel()
                optimizer.zero_grad()

                output = model(batch)
                loss = criterion(output, truth)
                if torch.isnan(loss):
                    raise RuntimeError('loss is nan')

                _, preds = torch.max(output, 1)
                running_corrects += torch.sum(preds == truth)

                loss.backward()
                optimizer.step()        

        print('acc:', (running_corrects / num_samples).item())

    return model

def train_shadow(shadow_net, 
        shadow_dataset, 
        criterion, 
        device, 
        num_epochs, 
        num_classes,
        cfg
    ):

    train_x = shadow_dataset['train_x']
    train_y = shadow_dataset['train_y']

    test_x = shadow_dataset['test_x']
    test_y = shadow_dataset['test_y']

    train_dataloader = data.DataLoader(
        TypicalDataset(train_x, 
            train_y, 
            num_classes
        ), 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=1
    )
    test_dataloader = data.DataLoader(
        TypicalDataset(test_x, 
            test_y, 
            num_classes
        ), 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=1
    )

    optimizer = optim.SGD(shadow_net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.l2)
    for epoch in range(num_epochs):
        with torch.enable_grad():
            shadow_net.train()
            for batch, label in train_dataloader:
                batch = batch.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = shadow_net(batch)
                loss = criterion(output, label)

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            shadow_net.eval()
            running_corrects = 0
            for batch, label in test_dataloader:
                batch = batch.to(device)
                label = label.to(device)

                output = shadow_net(batch)
                
                _, preds = torch.max(output, 1)
                running_corrects += torch.sum(preds == label)

            acc = running_corrects.double() / test_y.shape[0]
            print('acc =', acc.item())

    train_probs = torch.zeros((0, num_classes)).to(device)
    test_probs = torch.zeros((0, num_classes)).to(device)

    shadow_net.eval()
    train_truth = torch.zeros((0,), dtype=torch.int64).to(device)
    test_truth = torch.zeros((0,), dtype=torch.int64).to(device)
    with torch.no_grad():
        for batch, label in train_dataloader:
            batch = batch.to(device)
            label = label.to(device)

            output = shadow_net(batch, smax=False)
            train_probs = torch.cat((train_probs, output), dim=0)
            train_truth = torch.cat((train_truth, label), dim=0)

        for batch, label in test_dataloader:
            batch = batch.to(device)
            label = label.to(device)

            output = shadow_net(batch, smax=False)
            test_probs = torch.cat((test_probs, output), dim=0)
            test_truth = torch.cat((test_truth, label), dim=0)

    return shadow_net, {
        'train_probs': train_probs, 
        'test_probs': test_probs, 
        'train_truth': train_truth,
        'test_truth': test_truth
    }

# --------------------------------------------- #
#                evaluate attack
# --------------------------------------------- #

def evaluate_attack(target_net,
        attack_nets,
        x, y,
        num_classes,
        device
    ):

    dataloader = data.DataLoader(
        TypicalDataset(x, 
            y, 
            num_classes
        ),
        batch_size=1,
        shuffle=True,
        num_workers=1
    )

    target_net.eval()
    for a in attack_nets:
        a.eval()

    with torch.no_grad():
        corrects = 0
        total = 0
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)
            target_scores = target_net(img, smax=False)
            _, target_pred = torch.max(target_scores, 1)
            target_pred = target_pred.item()

            attack_net = attack_nets[target_pred]
            attack_scores = attack_net(target_scores)
            _, attack_pred = torch.max(attack_scores, 1)
            
            total += label.numel()
            corrects += torch.sum(attack_pred == label)
    acc = corrects.item() / total
    print('attack acc:', acc)


# ------------------------------------------ #
#                    start                   #
# ------------------------------------------ #

if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)

    cudnn.deterministic = True
    cudnn.benchmark = False

    dataset_root = osp.join(osp.dirname(os.getcwd()), 'datasets')

    # config
    _cfg = {
        'run_dir': 'shokri_repro_MNIST',
        'dataset': 'MNIST',
        'num_epochs': 30,
        'num_classes': 10,
        'gpu_id': 0,
        'num_target_train_size': 1000,
        'num_shadow_nets': 20,
        'lr': 1e-3,
        'l2': 0,
        'momentum': 0.9,
        'batch_size': 32
    }

    parser = argparse.ArgumentParser()
    for k, v in _cfg.items():
        if isinstance(v, bool):
            parser.add_argument('--%s' % k, action='store_true')
        else:
            parser.add_argument('--%s' % k, type=type(v))
    args = parser.parse_args()
    args = vars(args)
    for k, v in args.items():
        if v is not None:
            _cfg[k] = v

    print('config')
    pp.pprint(_cfg)
    cfg = EasyDict(_cfg)

    ckpt_dir, images_dir, log_dir = get_directories(cfg.run_dir)

    for path in [cfg.run_dir, ckpt_dir, images_dir, log_dir]:
        if not osp.exists(path):
            os.mkdir(path)

    mean, std = get_mean_std(cfg.dataset)
    train_transform, test_transform = get_dataset_transforms(mean, std, augment=False)

    train_data = datasets.MNIST(osp.join(dataset_root, cfg.dataset), train=True, transform=train_transform, download=False)
    test_data = datasets.MNIST(osp.join(dataset_root, cfg.dataset), train=False, transform=test_transform, download=False)

    # all train dataset present as (x, y)
    dataset_train_x = train_data.data
    dataset_train_y = train_data.targets

    # all test dataset present as (x, y)
    dataset_test_x = test_data.data
    dataset_test_y = test_data.targets

    dataset_sizes = {'train': len(train_data), 'test': len(test_data)}

    device = torch.device('cuda:%d' % cfg.gpu_id)

    # target model
    num_target_test_size = 2 * cfg.num_target_train_size

    target_dataset = get_target_dataset(dataset_test_x, 
                                    dataset_test_y, 
                                    cfg.num_target_train_size, 
                                    num_target_test_size,
                                    dataset_sizes['test']
                                )

    num_shadow_train_size = cfg.num_target_train_size
    num_shadow_test_size = cfg.num_target_train_size

    shadow_datasets = get_shadow_datasets(dataset_train_x, 
            dataset_train_y,
            cfg.num_shadow_nets,
            num_shadow_train_size,
            num_shadow_test_size,
            dataset_sizes['train']
        )

    target_net = TargetNet().to(device)
    print('created targer model...')
    print(target_net)
    # train target model
    criterion = nn.CrossEntropyLoss()

    print('training target model')
    target_net = train(target_net, # target_train_x, 
                    target_dataset['train_x'],
                    target_dataset['train_y'], #    target_train_y, 
                    cfg.lr,
                    criterion, 
                    device, cfg.num_epochs,
                    cfg.num_classes, cfg)

    # train shadow models
    shadow_nets = list()
    attack_datasets = list()
    for shadow_idx in range(cfg.num_shadow_nets):
        # shadow_net = ShadowNet().to(device)
        shadow_net = TargetNet().to(device)

        print('training shadow model (', shadow_idx, ')')
        shadow_net, attack_dataset = train_shadow(shadow_net, 
                            shadow_datasets[shadow_idx],
                            criterion,
                            device,
                            cfg.num_epochs,
                            cfg.num_classes,
                            cfg
                        )

        shadow_nets.append(shadow_net)
        attack_datasets.append(attack_dataset)

    attack_datasets = prepare_attack_dataset(attack_datasets, 
            cfg.num_classes,
            cfg.num_shadow_nets
        )

    attack_nets = list()
    # train attack models
    for class_idx in range(cfg.num_classes):
        print('training attack net for class', class_idx)
        attack_net = AttackNet().to(device)

        attack_net = train(attack_net, 
            attack_datasets[class_idx]['x'],
            attack_datasets[class_idx]['y'],
            cfg.lr,
            criterion,
            device,
            cfg.num_epochs,
            cfg.num_classes,
            cfg
        )

        attack_nets.append(attack_net)

    print('evaluating attack')
    evaluate_attack(target_net, 
        attack_nets, 
        target_dataset['test_x'], 
        target_dataset['test_y'],
        cfg.num_classes,
        device
    )
    
