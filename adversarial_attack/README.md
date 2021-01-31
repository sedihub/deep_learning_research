**T.B.D.: Complete adversarial attack for the way AlexNet performs inference...**
<br><br>

# Adversarial Attacks

Deep neural networks are vulnerable to adversarial attacks: These are inputs that have been modified in an imperceptible way to human eye that completely `deceive` the classifier network. Since classifiers CNNs are often used as the backbone of other deep computer vision tasks (e.g., localization and segmentation), this topic has attracted a great deal of attention.

The questions that piqued my interest in this topic were: Why don't we see a similar situation in biological systems? Wasn't CNNs inspired by experiments on the visual cortex of cat brains?

In the initial analysis, we see that one can "fool" one network, say ResNet-50. But the adversarial input that is effective on ResNet-50 is completely ineffective on a different architecture, say EfficientNet-B0.

A secondary set of questions that I will try to address in the course of this explorations are:
 - Can we deceive two networks at once? The answer likely is yes.
 - Can we develop an adversarial attack that is effective on inference the way done in AlexNet? 