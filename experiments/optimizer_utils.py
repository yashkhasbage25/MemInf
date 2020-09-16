#!/usr/bin/env python3

import torch.optim as optim

def get_optimizer(model, row):

    optim_name = row['optimizer']

    if optim_name == 'Adam':
        return optim.Adam(model.parameters(), lr=row['lr'], weight_decay=row['l2_decay'])
    elif optim_name == 'SGD':
        return optim.SGD(model.parameters(), lr=row['lr'], weight_decay=row['l2_decay'], momentum=row['momentum'])
    else:
        raise Exception('unknown optimizer: {}'.format(row['optimizer']))