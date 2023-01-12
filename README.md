# QuripfeNet
This is the code accompanying the paper "QuripfeNet:  Quantum-Resistant IPFE-based Neural Network".

# Introduction
Privacy preservation is an important issue in many sensitive applications involving neural network. In such applications, the users are reluctant to reveal their private data, such as medical condition, geographical location, financial situation and biometric information, to the cloud server. To avoid exposure of private data, several privacy-preserving neural networks which operate on the encrypted private data, are developed. Unfortunately, existing functional encryption-based privacy-preserving neural networks are mainly built on classical cryptography primitives, which are not secure under the threat from quantum computing. In this paper, we propose a quantum-resistant solution to protect the neural network inference based on an inner-product functional encryption scheme. The selected state-of-the-art inner-product functional encryption scheme works in polynomial form, which is not friendly to the computation in neural network that operates on floating point domain. We proposed a polynomial-based secure convolution layer to allow the neural network to resolve this problem, along with technique that reduces the memory consumption. The proposed QuripfeNet was applied on LeNet-5 and evaluated using the MNIST dataset. On a single threaded implementation, QuripfeNet took 106.6s for inference to classify one image, with an accuracy of 98.24\%, which is very close to the unencrypted version.

# How to use
There is a Makefile accompanied with the source codes in each separate folder. You can build the executable by typing "make".
Note that you need to change the sm version in GPU to suit your device. The default is -arch=sm_75, which is suitable for RTX2060,RTX2070 and RTX2080.

0) This source code provides the prediction of LeNet-5 against the MNIST dataset.

    We used a pre-trained model. The pre-trained model and MNIST dataset are in https://github.com/fan-wenjie/LeNet-5
    We used RLWE-IPFE scheme. It is in https://github.com/josebmera/ringLWE-FE-ref

1) The main function calls a data reading, a model loading, and one of two kinds of testing functions.

    You can select the functions by parameters in src/params.h
    //#define ORI			//Original LeNet-5 using PlainText
      #define PLAIN 	//Proposed LeNet-5 using PlainText
    //#define CPU			//Proposed LeNet-5 using ChiperText(by IPFE)
    //#define GPU			//Proposed LeNet-5 using ChiperText(by cuFE)(not implemented yet)

    testing() is used for the original LeNet-5 using PlainText
    sec_testing() is used for the proposed version including the proposed LeNet-5 using PlainText and the proposed LeNet-5 using ChiperText(by IPFE)
    * You may also comment out one of the prediction functions for testing.

2) The testing function calls one of the predict functions.

    ORI mode: use Predict(lenet, &features, 10). It is the same with the original LeNet-5 library.
    PLAIN mode: use sec_Predict(lenet, &features, 10). It is used the proposed polynomial convolution layer without encryption. It is used to compare ORI and CPU mode.
    CPU mode: use sec_Predict(lenet, &features, 10, msk). It is used the proposed QuripfeNet. It include any proposals.
    * You may also comment out one of the predict functions for testing.

3) Run the following commands to test run the SVM classification protected by cuFE.

    $ ulimit -s unlimited

    Run ORI or PLAIN mode as follow:
    $ ./lenet-5_ipfe-gpu (the sequence number of first data)

    Run CPU mode as follow:
    $ ./testing.sh

