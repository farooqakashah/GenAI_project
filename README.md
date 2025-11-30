PyTorch-MNIST-cGAN-cDCGAN

This project provides a PyTorch implementation of conditional Generative Adversarial Networks (cGAN) and conditional Deep Convolutional GANs (cDCGAN) for the MNIST handwritten digits dataset.

We experimented with both models by training multiple times.
Our results show that:

cGAN performed best within 20 epochs, giving sharp and stable images.

cDCGAN did not perform well at 20 epochs, so it was further trained up to 50 epochs to reach acceptable image quality.

The architecture (number of layers, filters, activation functions, etc.) differs from the original papers and is simplified for clarity.

Implementation Overview
cGAN Architecture

Takes a noise vector z and a digit label y as input.

Generator and discriminator are fully connected.

Works well on low-resolution datasets like MNIST.

cDCGAN Architecture

Convolutional version of cGAN.

Takes noise + label as input.

Produces more detailed images but requires more training time and tuning.

Results on MNIST
Generated Images Using Fixed Noise
cGAN (20 epochs)	cDCGAN (50 epochs)

	
MNIST vs Generated Images
MNIST (Real)	cGAN (20 epochs)	cDCGAN (50 epochs)

	
	
Training Time

cGAN

Avg per epoch: ~9 sec

20 epochs: ~180 sec

cDCGAN

Avg per epoch: ~47 sec

50 epochs: ~2350 sec

cGAN produced good-quality digits much earlier compared to cDCGAN.

Development Environment

Ubuntu 14.04 LTS

NVIDIA GTX 1080 Ti

CUDA 8.0

Python 2.7

PyTorch 0.1.12

torchvision 0.1.8

matplotlib 1.3.1

imageio 2.2.0
