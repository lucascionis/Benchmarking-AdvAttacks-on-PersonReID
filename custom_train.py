import argparse
import os
import glob

import torchreid

from torch.utils.data import DataLoader
from util import data_manager
from util.dataset_loader import ImageDataset
from opts import get_opts


parser = argparse.ArgumentParser(description='adversarial attack')
parser.add_argument('--target_model', type=str, default='aligned')
parser.add_argument('--pre_dir', type=str, default='./models', help='path to the model')
parser.add_argument('--queries_dir', type=str, default='./queries', help='path to be attacked model')
parser.add_argument('--test_batch', default=32, type=int, help="test batch size")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parser.parse_args()
    model = args.model
    weight_path = args.pre_dir
    queries_path = args.queries_dir
    test_batch = args.test_batch

    opt = get_opts(args.targetmodel)

    # load pretraind model
    torchreid.utils.load_pretrained_weights(model, weight_path)

    # load market151 dataset
    dataset = data_manager.init_img_dataset(root=args.root, name=args.dataset, split_id=opt['split_id'],
                                            cuhk03_labeled=opt['cuhk03_labeled'],
                                            cuhk03_classic_split=opt['cuhk03_classic_split'])

    galleryloader = DataLoader(ImageDataset(dataset.gallery, transform=opt['transform_test']),
                               batch_size=test_batch, shuffle=False, num_workers=opt['workers'],
                               pin_memory=torch.cuda.is_available(), drop_last=False)

    # for each query load the indices of the top-k predictions
    queries_idxs = []
    q_path = os.path.join(queries_path, '**/*.pkl')
    for query_idxs in glob.glob(q_path, recursive=True):
        queries_idxs.append(query_idxs)



if __name__=='__main__':
    main()