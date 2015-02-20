This is yet another implementation of a convolutional neural network for Nvidia GPU for the CIFAR-10 dataset. 
I wrote this purely for learning purposes. It's not meant to be for general use or even user friendly. 

The CUDA code at the moment is written for a compute 1.2 device, so should work on older CUDA capable GPUs.


COMPILING
=========

To compile you'll need:the following installed

- OpenCV 2.x
- CUDA SDK
- CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

Use CodeBlocks to open the project file. You'll need to edit Settings.h to point to where the CIFAR-10 dataset is installed.


NETWORK ARCHITECTURE
====================
The network is hard coded in the function RunCUDA_CNN_1/RunCUDA_CNN_2 and uses RunCUDA_CNN_2 by default.
RunCUDA_CNN_1 uses no translation pre-procsessing so the input images are 32x32. RunCUDA_CNN_2 does the 
translation and crops the image to 24x24. Edit this function to play around with different architectures.

The following layer types are supported

- ReLu (Rectified Linear Unit)
- tanh
- absolute tanh
- sigmoid
- linear
- max-pool (2x2 down sampling)
- avg-pool (2x2 down sampling)
- softmax


RUNNING
=======
Create a temporary directory in the root folder of ConvNN eg. mkdir /home/user/ConvNN/run1.
This is because ConvNN will dump out quite a few files and it's cleaner to have it in a separate directory.

Run the program via ./ConvNN

It will bring up a text based menu with some options that you just have to answer. Here's an example workflow I used:

    $ ./ConvNN

    What would you like to do?
    1 - train data
    2 - run network on test set
    or any other value to exit

    choice: 1
    How many epochs to train for: 6

    What learning rate?
    1 - 0.01
    2 - 0.001
    3 - 0.0001

    choice: 1

    Do you want to try and load existing saved weights? y/n: y

This will take some time but you should get around a validation error of ~2200. 
Now we run it again but reduce the learning rate to fine tune the weights.

    $ ./ConvNN

    What would you like to do?
    1 - train data
    2 - run network on test set
    or any other value to exit

    choice: 1
    How many epochs to train for: 1

    What learning rate?
    1 - 0.01
    2 - 0.001
    3 - 0.0001

    choice: 2

    Do you want to try and load existing saved weights? y/n: y

Repeat one more time for the smallest learning rate.

    $ ./ConvNN

    What would you like to do?
    1 - train data
    2 - run network on test set
    or any other value to exit

    choice: 1
    How many epochs to train for: 1

    What learning rate?
    1 - 0.01
    2 - 0.001
    3 - 0.0001

    choice: 3

    Do you want to try and load existing saved weights? y/n: y

The validation error should be around 1900. Last step is to run it on the test dataset.

    $ ./ConvNN

    What would you like to do?
    1 - train data
    2 - run network on test set
    or any other value to exit

    choice: 2

I got a test error around 1300-1400.

You can also plot the validation error by loading error_plot.txt. The values are outputted every run of a training dataset.
Technically this is 1/4 of an epoch because I'm using 4 datasets for training.


