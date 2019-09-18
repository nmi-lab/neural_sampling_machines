# neural_sampling_machine (NSM)

This repository contains the source code for the paper " " accepted for publication
in NeurIPS 2019. 

The file **model.py** implements Linear and Convolutional layers for Pytorch 
that provide all the methods for building Neural Sampling Machine networks. 

## Data sets
The following data sets used in the paper: 
- MNIST
- NMNIST
- EMNIST
- DVS Gestures
- CIFAR10/100

Each directory, named after a data set, contains the corresponding scripts that
implement the NSM as well as the conventional classifiers.


## Source Code and Platform Details

The source code is written in Python and Pytorch under the GPL license. All the
scripts have been tested on the following two machines: 
    - Ryzen ThreadRipper with 64GB physical memory running 
        - Arch Linux
        - Python 3.7.4 and Pytorch 1.2.0
        - GCC 9.1.0
        - Equipped with three Nvidia GeForce GTX 1080 Ti GPUs
    - Intel i7 with 64GB physical memory running
        - Arch Linux
        - Python 3.7.3 and Pytorch 1.0.1
        - GCC 8.2.1
        - Equipped with two Nvidia GeForce RTX 2080 Ti GPUs
