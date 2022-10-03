import os
import argparse

import torch
import torch.nn as nn
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy 
from torch.utils.data   import DataLoader, ConcatDataset


def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        test_dataset, _, _ = get_cifar10(args, args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        _, _, test_dataset = get_cifar100(args, args.datapath)
    
    # dataloader_path  = os.path.abspath('./data/dataloaders_%s_%s' %(args.dataset, str(args.num_labeled)))
    # test_loader_path      = os.path.abspath(dataloader_path + '/test_loader.pkl')

    # if os.path.exists(test_loader_path):
    #         test_dataset = torch.load(test_loader_path)

    test_predictions, mean_test_accuracy = test_cifar10(test_dataset, r"C:\Users\harsh\OneDrive\Desktop\git_project\CV_Project\Task2_VAT\model_weights_cifar10_0.0\epoch_127_of_128.h5")
    print(test_predictions.shape)
    print(mean_test_accuracy)


def test_cifar10(testdataset, filepath = "./model_weights_cifar10/last_trained.h5"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    model = WideResNet(depth=28, num_classes=10, widen_factor=2)
    model.load_state_dict(torch.load(os.path.abspath(filepath)))

    mean_acc = 0.0
    test_loader         = DataLoader(testdataset,
                                     batch_size = 256,
                                     shuffle = False, 
                                     num_workers=1)
    all_preds = torch.empty((0,10))
    for x_test, y_test in test_loader:
        test_preds = model(x_test)
        test_preds = test_preds.softmax(dim=1)
        all_preds  = torch.cat((test_preds, all_preds))
        test_acc   = accuracy(test_preds, y_test)
        mean_acc  += test_acc[0].item()
        
    return all_preds, mean_acc/len(test_loader)


def test_cifar100(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    model = model.load_state_dict(torch.load(filepath))
    test_preds = model(testdataset)
    
    return test_preds.softmax(dim=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
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




