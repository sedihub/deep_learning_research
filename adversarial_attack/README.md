# Adversarial Attacks

Deep neural networks are vulnerable to adversarial attacks: These are inputs that have been modified in an imperceptible way to human eye that completely `deceive` the classifier network. Since classifiers CNNs are often used as the backbone of other deep computer vision tasks (e.g., localization and segmentation), this topic has attracted a great deal of attention.

The questions that piqued my interest in this topic were: Why don't we see a similar situation in biological systems? Wasn't CNNs inspired by experiments on the visual cortex of cat brains?

In the initial analysis, we see that one can easily "fool" one network (ResNet-50): 

<p align="center">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/adversarial_attack/.images/original_and_adversarial_images.png" alt="Sample original and adversarial images." width="80%" height="80%">
<p align="center">

<p align="center">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/adversarial_attack/.images/original_and_adversarial_images_rgb_differences.png" alt="RGB channel differences between original and adversarial images" width="80%" height="80%">
</p>

But, not surprisingly, the adversarial attack is completely ineffective on other architectures: 

<p align="center">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/adversarial_attack/.images/other_architectures.png" alt="Adversarial attack is ineeffective on other architectures" width="80%" height="80%">
</p>

|   Architecture   |       Predicted Class Name           | Pred. Idx |  Confidence |
| ---------------- | ------------------------------------ | --------- | ----------- |
|Resnet-50         |  megalith, megalithic structure      |    649    |    0.780    |
|Efficientnet-B0   |  king penguin, Aptenodytes patagonica|    145    |    0.924    |
|DenseNet-201      |  king penguin, Aptenodytes patagonica|    145    |    0.994    |
|MobileNet-V2      |  king penguin, Aptenodytes patagonica|    145    |    0.958    |
|NasNetMobile      |  king penguin, Aptenodytes patagonica|    145    |    0.904    |

We will also see that passing the image through a low-pass filter is an effective way of countering adversarial attacks. 

<p align="center">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/adversarial_attack/.images/blurring_adversarial_image.png" alt="The effect of blurring" width="100%" height="100%">
</p>



# Notes
## Download Sample Images:
For this project, we need sample images from the ImageNet dataset. You could either download the whole dataset from the [ImageNet website](http://www.image-net.org/) or a 1000-instance sample set from [EliSchwartz's GitHub repository](https://github.com/EliSchwartz/imagenet-sample-images).


## T.B.D:
A secondary set of questions that I will try to address in the course of this explorations are:
 - Can we deceive two networks at once? The answer likely is yes.
 - Can we develop an adversarial attack that is effective on inference the way done in AlexNet?
 - Low-frequency adversarial features? 
