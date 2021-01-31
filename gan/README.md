
**Work in progress...  ツ**
<br><br>


# Generative Adversarial Networks

An image from a set of say car images can be thought of as a vector in a large (high-dimensional) space that is constrained to a manifold. Given a large set of images, we want to be able to approximate this manifold. Alternatively, we can think of this as the task of emulating a probability distribution. This is the goal of generative models.  

In this exploration we examine adversarial approach to image generation in the context of the MNIST dataset. In a later exploration we will look at auto-encoder-based models.

To set the stage, we first focus on developing and training discriminator and generator models in a supervised fashion. 

## Discriminator Model

To train the discriminator model (at least partially), we feed in scrambled real images (to make things a bit more difficult for the model) and along with random instances drawn from a probability distribution that has the same histogram as the real images:

![Histograms of fake and real images](https://github.com/sedihub/deep_learning_research/blob/master/gan/.images/histigrams_of_mnist_and_fake.png?raw=true) 

The way this is done, as shown below, is the same as the way any one-dimensional distribution is sampled: Compile the cumulative distribution, invert it, and finally sample it using a uniform distribution. Not that this simple-minded approach completely ignores any correlations between the pixel locations and intensities.

![Sampling histogram of MNIST images](https://github.com/sedihub/deep_learning_research/blob/master/gan/.images/emulating_mnist_histogram.png?raw=true) 

The set of images below show sampled instances (fake images) based on the histogram of the MNIST images versus some real images. As mentioned earlier, the make the task of classification a bit more difficult for the discriminator we scramble the images (the right-hand-side set).  


<p align="center">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/gan/.images/mnist_and_fake.png" alt="Original fake and real images" width="45%" height="45%">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/gan/.images/scrambled_mnist_and_fake.png" alt="Scrambled fake and real images" width="45%" height="45%">
</p>
 
The effect of scrambling input image on CNN images was explored in an earlier exploration. The short version is that this forces the discriminator to rely on features learned deeper into the network as the feature maps in the initial layers lack features that allow the discriminator to distinguish fake from real images.


## Generator Model

To see what our generator model can do, we first train it in a supervised setting using the MSE loss. Needless to say, this is futile, but happens to work in the simple case of MNIST for the reason that you can average all instances of 0 and still get a blurred version of the digit zero. One can feed in more information about the input such as average position and rotation.

Training the generator this way (with only the labels and the mean pixel value as input to the generator), we get:

![Sampling histogram of MNIST images](https://github.com/sedihub/deep_learning_research/blob/master/gan/.images/supervised_learning_genrative_model.png?raw=true) 

Finally, let's see what happens if we feed in different average pixel values to the generator model:

<p align="center">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/gan/.images/generated_images_with_image_pixel_means.png" alt="The effect of mean pixel value on the generated images" width="80%" height="80%">
</p>

We can play with the labels, too. I was really hoping to see 3 and 8 looking the same similar to what is seen in knowledge distillation (see [Hinton's paper](https://arxiv.org/abs/1503.02531)). Nonetheless, we can see how in the case with input of 5 and 0, the generator tries to have a combination of both:

<p align="center">
    <img src="https://github.com/sedihub/deep_learning_research/blob/master/gan/.images/combined_hidden_representation.png" alt="Mixing up labels as hidden representation" width="80%" height="80%">
</p>

Note that when we train the generator in the GAN setting, being an unsupervised learning situation, we cannot make the same assumptions about the number of inputs of the model. As we will see in the next section, there I set the input size to 32 and relied on uniform normal input. 


## DCGAN

Now we are finally ready to dive into the main topic of this exploration: Developing a generator in an adversarial setting. Let us first see what we expect at the outset: At convergence, we expect to discriminator to be unable to distinguish real from fake images. This means predicting 50-50 probabilities, which translate into the cross-entropy loss of 0.693 for both the discriminator and the generator. For this to happen, we need to avoid statistical noise. This is in fact one of the reasons the large batch sizes are recommended in the GAN settings.

Some comments on the choices made on the model input and architecture:
 - I use uniform random vectors as input. The main reason for this is that I want the GAN to be a probability generator function. Something along the lines of Gibbs sampling sans the memory issue.
 - I experimented with having sigmoid activation at the end to force the output of the generator to be in the range of 0 to 255. Also I experimented with Tanh and ReLU activations. These choices impeded training. That was why I resorted to leaky ReLU as recommended in the [DCGAN paper](https://arxiv.org/abs/1511.06434). In the hind sight, I can see why that's the case: unlike in supervised learning, there is no continuous march towards lowering the task loss. That's why flat regions in activations can result in weights getting stuck in for certain inputs due to vanishing gradients. 
