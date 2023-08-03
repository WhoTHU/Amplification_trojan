# Amplification trojan network: Attack deep neural networks by amplifying their inherent weakness
This is the official repository for the paper [Amplification trojan network: Attack deep neural networks by amplifying their inherent weakness](https://www.sciencedirect.com/science/article/abs/pii/S0925231222008773).

### 1. Requirements
All the codes are tested in the following environment:
* Linux (Ubuntu 18.04.4)
* CUDA 11.2
* Numpy 1.21.5 
* Python 3.8.13
* PyTorch 1.10.0
* TensorboardX 2.5

### 2. Train the trojan network
We provide the code for training the amplification trojan networks (ATNets) described in the paper. Users can specify the arguments according to their own requirements. They can use the code to train the ATNets on MNIST and CIFAR10 using untargeted C-FGSM, targeted C-FGSM, untargeted C-BIM and targeted C-BIM, to target a pre-trained network. 

The targeted network should be defined in "./models", and its pretrained checkpoints should be placed in "./checkpoints".

As an example, one can train an ATNet on MNIST using untargeted C-FGSM to target a pre-trained small CNN by
```
python train.py --eps 0.1 --parameters MNIST --sizes 1 16 32 32 32 --cs 100 1 1 --epoch 10 20 --lr 0.001 0.0001 --attack CFGSMUT --model CNN_MNIST --net_dict params_CNN_MNIST_e50.pkl
```
As another example, one can train an ATNet on CIFAR10 using targeted C-BIM to target a pre-trained ResNet18 by
```
python train.py --eps 0.008 --parameters CIFAR10 --sizes 3 32 64 64 64 --cs 500 1 1 --epoch 10 20 --lr 0.001 0.0001 --attack CBIMT --model resnet18 --net_dict params_resnet18_e240.pkl
```
