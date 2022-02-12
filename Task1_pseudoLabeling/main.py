import argparse
import math
from tqdm import tqdm
import warnings

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy

from model.wrn  import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn as nn

warnings.filterwarnings("ignore")

def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epoch):
        model.train()
        running_loss_train = 0.0
        for i in tqdm(range(args.iter_per_epoch)):

            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)

            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)

            optimizer.zero_grad()

            o_l = model(x_l)
            loss = criterion(o_l, y_l)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()

        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss_train / 2000))
        # train_loss.append(running_loss_train / 2000)
        running_loss_train = 0.0

        # with torch.no_grad():
        #     running_loss_test = 0.0
        #     for data in test_loader:
        #         images, labels = data
        #         outputs = model(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         optimizer.zero_grad()
        #         loss_test = criterion(outputs, labels)
        #         running_loss_test += loss_test.item()
        #     test_loss.append(running_loss_test/2500)

            ####################################################################
            # TODO: SUPPLY your code
            ####################################################################
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=512, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=512, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=128*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=128, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)