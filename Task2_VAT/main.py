import argparse
import math
from xmlrpc.client import boolean
from tqdm import tqdm
import warnings
import os
from pathlib import Path
import re

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, epoch_log

from model.wrn  import WideResNet
from vat import VATLoss

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn as nn

import warnings

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
    
    model_txt_path  = Path(args.model_wt_path) / Path("epoch_info.txt")
    model_last_path = Path(args.model_wt_path) / Path("last_trained.h5")
    
    if os.path.exists(model_txt_path):
      with open(model_txt_path, "r") as f:
          txt = f.read()
      start_model = int(re.search('Last model epoch: (.*)\n', txt).group(1)) + 1
      best_model = int(re.search('Best model epoch: (.*)\n', txt).group(1))
      model.load_state_dict(torch.load(model_last_path))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    for epoch in range(args.epoch):
        last_loss = 999999999.9
        for i in range(args.iter_per_epoch):

            if i % args.log_interval == 0:
                ce_losses = epoch_log()
                vat_losses = epoch_log()
                accuracies = epoch_log()

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
            ####################################################################
            # TODO: SUPPLY you code
            ####################################################################

            optimizer.zero_grad()
            vatLoss = VATLoss(args)

            vat_loss = vatLoss(model, x_ul)
            preds = model(x_l)
            classification_loss = criterion(preds.softmax(dim=1), y_l)
            loss = classification_loss + args.alpha * vat_loss
            loss.backward()
            optimizer.step()

            acc = accuracy(preds, y_l)

            ce_losses.update(classification_loss.item(), x_l.shape[0])
            vat_losses.update(loss.item(), x_ul.shape[0])
            accuracies.update(acc[0].item(), x_l.shape[0])

            if i % args.log_interval == 0:
                print(f'\nEpoch: {epoch}\t'
                f'\nIteration: {i}\t'
                f'CrossEntropyLoss {ce_losses.value:.4f} ({ce_losses.avg:.4f})\t'
                f'VATLoss {vat_losses.value:.4f} ({vat_losses.avg:.4f})\t'
                f'Accuracy {accuracies.value:.3f} ({accuracies.avg:.3f})')
        
        model_last_path = Path(args.model_wt_path) / Path("last_trained.h5")
        model_wts_path  = Path(args.model_wt_path) / Path(f"epoch_{epoch}_of_{args.epoch}.h5")
        model_txt_path  = Path(args.model_wt_path) / Path("epoch_info.txt")
        
        if not os.path.exists(args.model_wt_path):
            os.makedirs(args.model_wt_path)

        torch.save(model.state_dict(), model_last_path)
        if last_loss > loss:
            torch.save(model.state_dict(), model_wts_path)
            best_model = epoch

        last_loss = loss

        with open(model_txt_path, "w+") as f:
            f.write("Best model epoch: %d\n" % (best_model))
            f.write("Last model epoch: %d\n" % (epoch))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
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
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=128, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=10.0, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=1.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter") 
    parser.add_argument('--log-interval', type=int, default=100,
                        help='interval for logging training status')
    parser.add_argument("--model_wt_path", default="./model_weights/", 
                    type=str, help="Path to the saved model")
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()


    main(args)