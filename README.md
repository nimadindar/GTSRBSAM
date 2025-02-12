# GTSRBSAM

This repository aims to implement a classifier for the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The dataset comprises more than 50,000 images in total and includes over 40 classes. For comparison purposes, a ResNet-18 model has also been fine-tuned.

## Approach

### Proposed Model Structure
The proposed model structure includes:
- **4 Convolutional Layers** with a kernel size of 3, stride of 3, and padding of 1.
- **Batch Normalization** layers following each convolutional layer for regularization.
- **Leaky ReLU** as the activation function.
- **Max Pooling** as the pooling layer.
- **3 Skip Connections** to improve learning for deep models.

### Weight Initialization
To effectively initialize the classifier model's weights:
- An **autoencoder** was pretrained on the CIFAR-100 dataset for 100 epochs as warm-up steps.
- The weights of the pretrained convolutions were initialized using the weights from the encoder part of the autoencoder.
- Fully connected layers (128, 64, 43) were added, utilizing **dropout** as a regularization method.
- The classifier was trained for 20 epochs with a learning rate of 0.001.

### Optimization Schemes
- **Adam Optimizer** was used for the "normal" model.
- **Sharpness Aware Minimizer (SAM)** was implemented for better generalization, as introduced in the referenced paper.

## Results
- **SAM Optimizer**: Achieved a test accuracy of **90.82%**.
- **Adam Optimizer**: Achieved a test accuracy of **90.42%**.
- **ResNet-18 Fine-tuning**: Achieved a train accuracy of **98.64%** and a test accuracy of **93.14%** after 20 epochs.

The plots in the repository show the train accuracy and loss per epoch for models trained using SAM and Adam optimizers. All training weights are available in the code files provided. Training was conducted on personal devices.

## Files
The repository is organized as follows:

- **AutoEncoder**
  - `autoencoder.py`: Contains the proposed model for the AutoEncoder.
  - `train_autoencoder.py`: Script for training the autoencoder.
  
- **Classifier**
  - `classifier.py`: Contains the proposed model for the classifier.
  - `train_classifier.py`: Script for training the classifier.
  
- **SAM**
  - `SAM.py`: Implementation of Sharpness Aware Minimization.
  
- **Finetune ResNet**
  - `finetune_resnet.py`: Script for fine-tuning the ResNet model.

## References
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)
