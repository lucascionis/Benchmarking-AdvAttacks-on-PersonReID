import argparse
import os
import glob
import pickle

import torch
import torchreid
from torchreid import models, metrics
from torchreid.losses import CrossEntropyLoss, DeepSupervision
# import torch.nn as nn

import models

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from util import data_manager
from util.dataset_loader import ImageDataset
from opts import get_opts

parser = argparse.ArgumentParser(description='adversarial attack')
parser.add_argument('--target_model', type=str, default='aligned')
parser.add_argument('--pre_dir', type=str, default='./models', help='path to the model')
parser.add_argument('--queries_dir', type=str, default='./queries', help='path to be attacked model')
parser.add_argument('--test_batch', default=32, type=int, help="test batch size")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_freezen(net, need_modified=False, after_modified=None):
    # print(net)
    cc = 0
    for child in net.children():
        for param in child.parameters():
            if need_modified: param.requires_grad = after_modified
            # if param.requires_grad: print('child', cc , 'was active')
            # else: print('child', cc , 'was frozen')
        cc += 1


def parse_data_for_train(data):
    imgs = data[0]
    pids = data[1]
    return imgs, pids


def compute_loss(criterion, outputs, targets):
    if isinstance(outputs, (tuple, list)):
        loss = DeepSupervision(criterion, outputs, targets)
    else:
        loss = criterion(outputs, targets)
    return loss


def train_model(model, data, optimizer, criterion):
    imgs, pids = parse_data_for_train(data)

    imgs = imgs.to(device)
    pids = pids.to(device)

    ls = model(imgs, is_training=True)
    if len(ls) == 1: outputs = ls[0]
    if len(ls) == 2: outputs, features = ls
    if len(ls) == 3: outputs, features, local_features = ls
    loss = compute_loss(criterion, outputs, pids)
    # loss = criterion(outputs, pids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_summary = {
        'loss': loss.item(),
        'acc': metrics.accuracy(outputs, pids)[0].item()
    }

    return loss_summary


def main():
    args = parser.parse_args()
    model = args.target_model
    weight_path = args.pre_dir
    queries_path = args.queries_dir
    test_batch = args.test_batch
    dataset = 'market1501'

    opt = get_opts(model)

    # load market151 dataset
    dataset = data_manager.init_img_dataset(root='data', name=dataset, split_id=opt['split_id'],
                                            cuhk03_labeled=opt['cuhk03_labeled'],
                                            cuhk03_classic_split=opt['cuhk03_classic_split'])

    '''
    galleryloader = DataLoader(ImageDataset(dataset.gallery, transform=opt['transform_test']),
                               batch_size=test_batch, shuffle=False, num_workers=opt['workers'],
                               pin_memory=torch.cuda.is_available(), drop_last=False)
    '''

    # load pretraind model
    # model = models.build_model(name=model, num_classes=dataset.num_train_pids)
    # torchreid.utils.load_pretrained_weights(model, weight_path)
    target_net = models.init_model(name=model, pre_dir=weight_path, num_classes=dataset.num_train_pids)
    # check_freezen(target_net, need_modified=True, after_modified=True)
    target_net.to(device)
    target_net.train()

    optimizer = torchreid.optim.build_optimizer(
        target_net,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    criterion = CrossEntropyLoss(
        num_classes=dataset.num_train_pids,
        use_gpu=torch.cuda.is_available,
        label_smooth=True
    )

    # for each query load the indices of the top-k predictions
    queries_idxs = []
    q_path = os.path.join(queries_path, '**/*.pkl')
    for query_idxs in glob.glob(q_path, recursive=True):
        with open(query_idxs, 'rb') as file:
            queries_idxs.append(pickle.load(file))

    gallery_loaders = []
    for query_idxs in queries_idxs:
        # each query_idxs is a numpy array with indices from the gallery
        # create a subset of dataset galley using the indices
        g_subset = Subset(dataset.gallery, query_idxs)
        # create a dataloader using the subset
        # g_loader = DataLoader(g_subset, batch_size=1, shuffle=False, num_workers=2)
        g_loader = DataLoader(ImageDataset(g_subset, transform=opt['transform_train']),
                               batch_size=5, shuffle=False, num_workers=opt['workers'],
                               drop_last=False)
        gallery_loaders.append(g_loader)

    for g_loader in gallery_loaders:
        for idx, batch in enumerate(g_loader):
            loss_summary = train_model(target_net, batch, optimizer, criterion)
            print(loss_summary)
        # eventually save this as model_best_queryID.pth


if __name__ == '__main__':
    main()