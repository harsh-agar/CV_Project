## Pseudo-labeling  

Pseudo-labeling is a contemporary technique in SSL which assigns labels to the unlabeled data using the predictions of the network trained on the labeled data.

The idea is quite simple, first we train our network using our labeled data and then using our trained network, we predict the class probabilities of our unlabeled data. If the probability of a class is greater than a user defined threshold value for a sample, we assign the sample that particular class and use it for training. We repeat this procedure for every epoch of training.

Set the input arguments accordingly:

```bash
python main.py --dataset cifar10 --num-labeled 4000 --threshold 0.95
```
For example, the above command is to train the model on cifar 10 dataset, using just 4000 labeled images with a threshold of 0.95.