import argparse
import math
from xmlrpc.client import boolean
from tqdm import tqdm
import warnings
import os
from pathlib import Path
import re

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
    unlabeled_loader    = DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers)
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

    x_t = torch.empty(0, 3, 32, 32).to(device)
    y_t = torch.empty(0).to(device).int()
    
    curr_use_data = 'labelled'
    

    model_txt_path  = Path(args.model_wt_path) / Path("epoch_info.txt")
    model_last_path = Path(args.model_wt_path) / Path("last_trained.h5")

    last_loss_train = 9999999999999.9
    start_model = 0
    best_model = 0
        
    if os.path.exists(model_txt_path):
        with open(model_txt_path, "r") as f:
            txt = f.read()
        start_model = int(re.search('Last model epoch: (.*)\n', txt).group(1)) + 1
        best_model = int(re.search('Best model epoch: (.*)\n', txt).group(1))
        model.load_state_dict(torch.load(model_last_path))

    for epoch in range(start_model, args.epoch):
        model.train()
        running_loss_train = 0.0

        x_t = torch.empty(0, 3, 32, 32).to(device)
        y_t = torch.empty(0).to(device).int()

        if epoch > 10:    
            for unlab_load in unlabeled_loader:
                x_ul, _ = unlab_load

                x_ul = x_ul.to(device)
                o_ul = model(x_ul)
                x_t = torch.cat((x_t, x_ul[torch.where(torch.max(o_ul.softmax(dim=1), axis=1)[0] > 0.95)[0]]))
                y_t = torch.cat((y_t, o_ul.softmax(dim=1).max(dim=1)[1][torch.where(torch.max(o_ul.softmax(dim=1), axis=1)[0] > 0.95)[0]]))

        for i in tqdm(range(args.iter_per_epoch)):
            
            try:
                if i == 0 and curr_use_data == 'X_t':
                    raise StopIteration()
                else:
                    x_l, y_l    = next(labeled_loader)

            except StopIteration:
                if curr_use_data == 'labelled' and y_t.shape[0] != 0:
                        labeled_loader      = iter(DataLoader(list(zip(x_t, y_t)), batch_size = args.train_batch, 
                                                    shuffle = True, 
                                                    num_workers=args.num_workers))
                        curr_use_data = 'X_t'
                else:
                    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                                batch_size = args.train_batch, 
                                                shuffle = True, 
                                                num_workers=args.num_workers))
                    curr_use_data = 'labelled'
                x_l, y_l    = next(labeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)

            optimizer.zero_grad()

            o_l = model(x_l)
            loss = criterion(o_l.softmax(dim=1), y_l)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()

        print('[epoch = %d] loss: %.3f' %
                (epoch, running_loss_train / args.iter_per_epoch))
        # train_loss.append(running_loss_train / 2000)
        
        model_last_path = Path(args.model_wt_path) / Path("last_trained.h5")
        model_wts_path  = Path(args.model_wt_path) / Path(f"epoch_{epoch}_of_{args.epoch}.h5")
        model_txt_path  = Path(args.model_wt_path) / Path("epoch_info.txt")
        
        if not os.path.exists(args.model_wt_path):
            os.makedirs(args.model_wt_path)

        torch.save(model.state_dict(), model_last_path)
        if last_loss_train > running_loss_train:
            torch.save(model.state_dict(), model_wts_path)
            best_model = epoch

        last_loss_train = running_loss_train

        with open(model_txt_path, "w+") as f:
            f.write("Best model epoch: %d\n" % (best_model))
            f.write("Last model epoch: %d\n" % (epoch))

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
    parser.add_argument('--train-batch', default=256, type=int, #512
                        help='train batchsize')
    parser.add_argument('--test-batch', default=512, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=128*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=128, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=0, type=int, #4
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--use-saved-model", type=bool, default=True,
                        help="Use one of the saved model")
    parser.add_argument("--model_wt_path", default="./model_weights/", 
                        type=str, help="Path to the saved model")
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)