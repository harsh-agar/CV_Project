import argparse
import math
from xmlrpc.client import boolean
from tqdm import tqdm
import warnings
import os
from pathlib import Path
import re
import numpy as np
import itertools

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, epoch_log

from model.wrn  import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader, ConcatDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore")

def main(args):
    writer = SummaryWriter()

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

    if not os.path.exists(Path(args.datapath) / Path('dataloaders_%s_%s' %(args.dataset, str(args.num_labeled)))):
        os.makedirs(Path(args.datapath) / Path('dataloaders_%s_%s' %(args.dataset, str(args.num_labeled))))

    labeled_loader_path = Path(args.datapath) / Path('dataloaders_%s_%s' %(args.dataset, str(args.num_labeled))) / Path('labeled_loader.pkl')
    valid_loader_path = Path(args.datapath) / Path('dataloaders_%s_%s' %(args.dataset, str(args.num_labeled))) / Path('valid_loader.pkl')
    unlabeled_loader_path = Path(args.datapath) / Path('dataloaders_%s_%s' %(args.dataset, str(args.num_labeled))) / Path('unlabeled_loader.pkl')
    test_loader_path = Path(args.datapath) / Path('dataloaders_%s_%s' %(args.dataset, str(args.num_labeled))) / Path('test_loader.pkl')

    if os.path.exists(labeled_loader_path) and os.path.exists(valid_loader_path) and os.path.exists(unlabeled_loader_path) and os.path.exists(test_loader_path):
        labeled_dataset = torch.load(labeled_loader_path)
        unlabeled_dataset_split = torch.load(unlabeled_loader_path)
        valid_dataset = torch.load(valid_loader_path) 
        test_dataset = torch.load(test_loader_path)
      
    else:
        torch.save(labeled_dataset, labeled_loader_path)
        torch.save(test_dataset, test_loader_path)

        val_set_idx = np.empty(0)
        for i in set(unlabeled_dataset.targets):
            data_size = int(np.where(unlabeled_dataset.targets ==  i)[0].shape[0] * 0.1)
            val_set_idx = np.append(val_set_idx, np.random.choice(np.where(unlabeled_dataset.targets ==  i)[0], data_size))

        x_val = torch.empty(0, 3, 32, 32)
        y_val = torch.empty(0).int()
        x_unl = torch.empty(0, 3, 32, 32)
        y_unl = torch.empty(0).int()
        for i in tqdm(range(len(unlabeled_dataset))):
            if i in val_set_idx:
                x_val = torch.cat((x_val, unlabeled_dataset[i][0][None, :]))
                y_val = torch.cat((y_val, torch.tensor(unlabeled_dataset.targets[i : i+1])))
            else:
                x_unl = torch.cat((x_unl, unlabeled_dataset[i][0][None, :]))
                y_unl = torch.cat((y_unl, torch.tensor(unlabeled_dataset.targets[i : i+1])))
        valid_dataset = list(zip(x_val, y_val))
        unlabeled_dataset_split = list(zip(x_unl, y_unl))
    
        torch.save(unlabeled_dataset_split, unlabeled_loader_path)
        torch.save(valid_dataset, valid_loader_path)

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    valid_loader        = DataLoader(valid_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers)
    unlabeled_loader    = DataLoader(unlabeled_dataset_split, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers)
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)

    model       = WideResNet(args.model_depth, 
                             args.num_classes, widen_factor=args.model_width, dropRate=args.drop_rate)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

    model_wt_path = Path('model_weights_%s_%s_%.2f' %(args.dataset, args.num_labeled, args.threshold)) 
    model_txt_path  = Path(model_wt_path) / Path("epoch_info.txt" )
    model_last_path = Path(model_wt_path) / Path("last_trained.h5")

    best_loss_val = 9999999999999.9
    start_model = 0
    best_model = 0
        
    if os.path.exists(model_txt_path):
        with open(model_txt_path, "r") as f:
            txt = f.read()
        start_model = int(re.search('Last model epoch: (.*)\n', txt).group(1)) + 1
        best_model = int(re.search('Best model epoch: (.*)\n', txt).group(1))
        model.load_state_dict(torch.load(model_last_path))
        print("Loaded Model: " + str(start_model))

    for epoch in range(start_model, args.epoch):
        model.train()
        running_loss_train = 0.0

        x_t = torch.empty(0, 3, 32, 32)
        y_t = torch.empty(0).int()

        if True: #epoch > 0:    
            for unlab_load in unlabeled_loader:
                x_ul, _ = unlab_load

                x_ul = x_ul.to(device)
                o_ul = model(x_ul)
                x_ul = x_ul.to('cpu')
                o_ul = o_ul.to('cpu')
                x_t = torch.cat((x_t, x_ul[torch.where(torch.max(o_ul.softmax(dim=1), axis=1)[0] > args.threshold)[0]]))
                y_t = torch.cat((y_t, o_ul.softmax(dim=1).max(dim=1)[1][torch.where(torch.max(o_ul.softmax(dim=1), axis=1)[0] > args.threshold)[0]]))
                w_t = (torch.sigmoid(torch.max(o_ul.softmax(dim=1), axis=1)[0])*2 - 1).detach()

        print(x_t.shape[0])
        train_accuracies = epoch_log()
        val_accuracies   = epoch_log()

        for i in tqdm(range(args.iter_per_epoch)):

            try:
                    data_sample = next(labeled_loader)
                    if len(data_sample) == 2:
                        x_l, y_l      = data_sample
                        w_l = torch.ones(x_l.shape[0])
                    else:
                        x_l, y_l, w_l = data_sample

            except StopIteration:
                if x_t.shape[0] > 0:
                    labeled_loader      = iter(itertools.chain(DataLoader(list(zip(x_t, y_t, w_t)), 
                                                                          batch_size = args.train_batch, 
                                                                          shuffle = True, 
                                                                          num_workers=args.num_workers),
                                                               DataLoader(labeled_dataset, 
                                                                          batch_size = args.train_batch, 
                                                                          shuffle = True, 
                                                                          num_workers=args.num_workers)))
                else:
                    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                                          batch_size = args.train_batch, 
                                                          shuffle = True, 
                                                          num_workers=args.num_workers))

                data_sample = next(labeled_loader)
                if len(data_sample) == 2:
                    x_l, y_l      = data_sample
                    w_l = torch.ones(x_l.shape[0])
                else:
                    x_l, y_l, w_l = data_sample
            
            x_l, y_l, w_l    = x_l.to(device), y_l.to(device), w_l.to(device)

            optimizer.zero_grad()

            o_l = model(x_l)
            loss = criterion(o_l.softmax(dim=1), y_l)
            loss = loss * w_l
            loss.mean().backward()
            optimizer.step()

            train_acc = accuracy(o_l, y_l)

            train_accuracies.update(train_acc[0].item(), x_l.shape[0])

            running_loss_train += loss.mean().item()

        val_loss = 0.0
        for val_i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            v_loss = criterion(outputs.softmax(dim=1), labels)
            val_loss += v_loss.item()
            
            val_acc = accuracy(outputs, labels)

            val_accuracies.update(val_acc[0].item(), inputs.shape[0])
        
        print('[epoch = %d] train_loss: %.3f val_loss: %.3f train_accuracy: %.3f val_accuracy: %.3f' %
                (epoch, running_loss_train / args.iter_per_epoch, val_loss / val_i, train_accuracies.avg, val_accuracies.avg))
        # train_loss.append(running_loss_train / 2000)
        
        model_last_path = Path(model_wt_path) / Path("last_trained.h5")
        model_wts_path  = Path(model_wt_path) / Path(f"epoch_{epoch}_of_{args.epoch}.h5")
        model_txt_path  = Path(model_wt_path) / Path("epoch_info.txt")
        
        if not os.path.exists(model_wt_path):
            os.makedirs(model_wt_path)

        torch.save(model.state_dict(), model_last_path)
        if best_loss_val > val_loss:
            torch.save(model.state_dict(), model_wts_path)
            best_model = epoch
            best_loss_val = val_loss

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
                        default=256, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.1, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.001, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=256, type=int, #512
                        help='train batchsize')
    parser.add_argument('--test-batch', default=256, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=256*128, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=256, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=0, type=int, #4
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=22,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--drop-rate", type=float, default=0.4,
                        help="model dropout rate")
    parser.add_argument("--use-saved-model", type=bool, default=True,
                        help="Use one of the saved model")
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)