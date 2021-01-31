**T.B.D.: Clean up the contents...**

# CNN vs. Dense

CNNs have revolutionized deep computer vision. One of the reasons for this is transfer learning. The idea behind transfer learning in the context of CNNs is that low-lever features extracted are useful/common across many tasks. The intuitive justification for this is that images share boundaries between regions. 

In this exploration, we try to see how sensitive CNNs are to scrambling the input image. We use a dense (a.k.a., fully-connected) architecture as benchmark. We should note two points at the outset:
 - To be fair in the comparison, we should make sure that the CNN layers are such that the receptive field at some hidden layer is the same of the whole image or the scrambled domain.
 - Fully-connected input layer is agnostic to input ordering. So, we expect the dense architecture to be unaffected by the scrambling on the input (we train from scratch in both cases). 
