# Wavelet-Based CNNs

The input size is one of the main architecture decisions in most computer vision tasks. Conceptually, the input size can be thought of as the frequency threshold of low-pass filter, limiting high-frequency features available to the model. We also saw in the adversarial project that adversarial features, at least emperically, tend to be high frequency. Also, input size offers an opprtunity for optimization, which EfficientNets leverage.

Wavelet transform has a number of advantages over spectograms in temporal signal analysis. It is also used in image compression. While it is true that discrete Fourier transform also decomposes the input (in this case, the image) into a hierarchy of frequency components, the way it does this decomposition may not be amenable to CNN-based architectures. This is infact the subject of another exploration.

The last ingredient of this project is dynamic or switchable networks (see [Deep Learning of Representations: Looking Forward](https://arxiv.org/abs/1305.0445)). These are models with components that can be autonomously or manually turned on and off in order to achieve higher performance. 

Putting all these ingredients together, we finally arrive at the main question of this project: Can we use wavelet transform to construct a CNN-based model that can be progressively given more high-frequency features?
