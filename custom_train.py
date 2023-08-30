import argparse
import os
import glob
import pickle
import copy

import torch
import torchreid
from torchreid import models, metrics
from torchreid.losses import CrossEntropyLoss, DeepSupervision

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
parser.add_argument('--train_batch', default=10, type=int, help="train batch size")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def DeepSupervision(criterion, xs, labels):
    loss = 0.
    for x in xs:
      loss += criterion(x, labels)
    return loss


def train_model(model, data, labels, optimizer, criterion):
    imgs, _, _, _ = data
    imgs = imgs.to(device)
    labels = torch.tensor(labels, dtype=torch.int64).to(device)

    ls = model(imgs, is_training=True)
    if len(ls) == 1: outputs = ls[0]
    if len(ls) == 2: outputs, features = ls
    if len(ls) == 3: outputs, features, local_features = ls
    loss = DeepSupervision(criterion, outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_summary = {
        'loss': loss.item(),
        'acc': metrics.accuracy(outputs, labels)[0].item()
    }

    return loss_summary


def main():
    args = parser.parse_args()
    model = args.target_model
    weight_path = args.pre_dir
    queries_path = args.queries_dir
    train_batch = args.train_batch
    dataset_name = 'market1501'

    opt = get_opts(model)

    dataset = data_manager.init_img_dataset(root='data', name=dataset_name, split_id=opt['split_id'],
                                            cuhk03_labeled=opt['cuhk03_labeled'],
                                            cuhk03_classic_split=opt['cuhk03_classic_split'])

    # for each query load the indices of the top-k predictions
    queries_idxs = []
    q_path = os.path.join(queries_path, '**/*.pkl')
    for query_idxs in glob.glob(q_path, recursive=True):
        with open(query_idxs, 'rb') as file:
            queries_idxs.append(pickle.load(file))

    target_net = models.init_model(name=model, pre_dir=weight_path, num_classes=dataset.num_train_pids)

    _target_net_cl = copy.deepcopy(target_net.classifier_local)
    target_net.classifier_local = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    # print(target_net)

    # check_freezen(target_net, need_modified=True, after_modified=True)
    target_net.to(device)
    target_net.train()

    criterion = torchreid.losses.CrossEntropyLoss(
        num_classes=2,
        use_gpu=torch.cuda.is_available,
        label_smooth=False
    )

    # Prepare the gallery loaders
    print("Preparing gallery loaders...")
    gallery_loaders = []
    for query_idxs in queries_idxs:
        # each query_idxs is a numpy array with indices from the gallery
        # create a subset of dataset galley using the indices
        idxs = torch.tensor(query_idxs[:, 0], dtype=int)
        labels = query_idxs[:, 1]
        g_subset = Subset(dataset.gallery, idxs)
        # create a dataloader using the subset
        g_loader = DataLoader(ImageDataset(g_subset, transform=opt['transform_train']),
                              batch_size=train_batch, shuffle=False, num_workers=opt['workers'],
                              drop_last=False)
        gallery_loaders.append([g_loader, labels])

    # Train the model using each gallery loader data
    print("Training re-id model with gallery images...")
    for idx, g_l in enumerate(gallery_loaders):
        target_net_copy = copy.deepcopy(target_net)
        optimizer = torchreid.optim.build_optimizer(
            target_net_copy,
            optim="adam",
            lr=0.0003
        )

        loader = g_l[0]
        labels = g_l[1]

        for idx, batch in enumerate(loader):
            labels_batch = labels[idx * train_batch:(idx + 1) * train_batch]
            loss_summary = train_model(target_net_copy, batch, labels_batch, optimizer, criterion)
            print(loss_summary)

        # Saving the trained model
        print("Saving trained model for query {}...".format(idx))
        target_net_copy.classifier_local = _target_net_cl

        file_name = 'retrained_{}_q{}'.format(model, idx)
        torch.save(target_net_copy.state_dict(), '/content/drive/MyDrive/adv_reid/retrained/{}.pth.tar'.format(file_name))

        del target_net_copy, optimizer

if __name__ == '__main__':
    main()