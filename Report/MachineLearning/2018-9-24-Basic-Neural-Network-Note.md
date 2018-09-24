---
layout: post
title: "Neural-Network-Report"
categories:
  - Blog
tags:
  - content
  - Machine-Learning
last_modified_at: 2018-09-18T11:05:52-05:00
---
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# Report
## ToC
- [Introduction](#1-introduction)
  - Machine Learning
  - Deep Learning
  - Convolutional Neural Network
- [Convolutional Neural Network](#2-convolutional-neural-network)
- [Realization](#3-realization)
- [Reference](#reference)
## 1 Introduction
### Machine Learning

<div align="center">
<img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/MachineLearning/MLField.jpg" width="80%" alt="MLField">    
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/MachineLearning/MLField_Types.jpg" width="80%" alt="MLField_Types">    
</div>

### Deep Learning
- **Convolutional Neural Networks(CNN)**
- AutoEncoder
- Sparse Coding
- Restricted Boltzmann Machine(RBM)
- Deep Belief Networks(DBN)
- Recurrent neural Network(RNN)

### Convolutional Neural Network
1986 Back propagation: [Learning Internal Representations by Error Propagation](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf)

1989 Prototype of CNN: [Backpropagation applied to handwritten zip code recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)

1998 First formal CNN, LeNet-5: [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

2006 CNN back to vision, milestone: [Reducing the dimensionality of data with neural networks](https://www.cs.toronto.edu/~hinton/science.pdf)

2012 Great broke in image recognition, AlexNet: [Imagenet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

2013 Proposed deconvolution layer, ZFNet: [Visualizing and Understanding Convolutional Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

2015 We need to go deeper, GoogLeNet: [Going Deeper With Convolutions](https://arxiv.org/pdf/1409.4842.pdf)

2015 Introduction of new structure, ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

See also, [VGGNet](https://arxiv.org/pdf/1409.1556.pdf), [Dropout](https://arxiv.org/pdf/1207.0580.pdf), [Momentum optimizer](http://proceedings.mlr.press/v28/sutskever13.pdf), [GAN](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), [RNN](https://arxiv.org/pdf/1308.0850.pdf), [AlphaGo](https://www.nature.com/articles/nature16961.pdf), [RCNN](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)

## 2 Convolutional Neural Network

Image here

### Convolution
- Input size
- Output size
- Filter size
- Stride
- Padding

![Convolution_0](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_0.jpg
"Convolution_0")

To understand convolution operation, we denote pixel in image at line i, column j as $ x_{i,j} $, element in filter at line $ m $, column $ n $ as $ w_{m,n} $ and use $ w_b $ to represent bias. Then we denote pixel in Feature map at line $ i $, row $ j $ as $ a_{i,j} $. Then we could get follow formulation:

$$ a_{i,j}=\sum_{m = 0}^{2}\sum_{n = 0}^{2}(w_{m,n}x_{i+m,j+n}+w_b) $$

We can extend it to a more general condition, where $ F $ is size of filter, $ D $ is deep(or channel) of input image:

$$ a_{i,j}=\sum_{d=0}^D \sum_{m = 0}^{F-1} \sum_{n = 0}^{F-1}(w_{d,m,n}x_{d,i+m,j+n}+w_b) $$

After one step:

![Convolution_1](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_1.jpg)

After two steps:

![Convolution_2](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_2.jpg)

Now, we use a gif to show the process

![Convolution](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution.gif "Convolution")

If we set convolution stride as 2, then we'll go through following process:

![Convolution_stride2-1](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_strid2_1.jpg "Convolution_stride2-1")

![Convolution_stride2-2](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_strid2_2.jpg "Convolution_stride2-2")

![Convolution_stride2-3](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_strid2_3.jpg "Convolution_stride2-3")

![Convolution_stride2-4](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_strid2_4.jpg "Convolution_stride2-4")

As for padding operation, it means add aditional zeros around the picture, which always used to make sure output size is a determined number. We'll take follow gif for intutive understanding.

![Convolution_Bias_multiFilter](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Convolution/Convolution_Bias_multiFilter.gif
 "Convolution_Bias_multiFilter")

Naturally, we can find relationship between output size, input size, filter size and stride:

$$ W_2 = \left \lfloor (W_1-F+2P)/S \right \rfloor +1 $$

In above, $ W_2 $ denote output image width, $ W_1 $ denote input image width, $ F $ represent size of filter, $ P $ is number of padding size, which will explain later, and $ S $ is stride you take.

### Activation Function

Use a nonlinear function, effect on the result of convolution, which is linear transform, to add nonlinear parts in the whole network.

- Sigmoid

  $$ \sigma(x)=\frac{1}{1+e^{-x}} $$

  <div align="center">
<img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Activation%20function/sigmoid.png" alt="Sigmoid">    
  </div>
  
  \+ From probability, used to be popular
  
  \- If input far from origin point, tend to cause 'Gradient vanishment' in backpropagation
  
  \- Need exponential operation, which cost compute source
  
- tanh
  
  $$ tanh(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^x-e^{-x}}{e^x+e^{-x}} $$
  
  <div align="center">
<img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Activation%20function/tanh.png" alt="tanh">    
  </div>
  
  \+ Output in (-1, 1)
  
  \- If input far from origin point, tend to cause 'Gradient vanishment' in backpropagation
  
  \- Need exponential operation, which cost compute source
  
- ReLU 

  $$ f(x) = max(0,x) $$
  
  <div align="center">
    <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Activation%20function/ReLU.png" alt="ReLU">    
  </div>
  
  \+ Won't cause 'Gradient vanishment' when input is positive
  
  \+ More fast in computation
  
  \- When input is negtive, it won't work, so the gradient will be zero, and cause same problem as Sigmoid
  
  Whatever, it's still most widely used activation function.
  
- ELU

  $$ f(x)=\left\{ \begin{aligned} &x &, & x>0 \\ \alpha(e^x&-1) & , & x\leq 0 \end{aligned} \right. $$
  
  <div align="center">
    <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Activation%20function/ELU.png" alt="ELU">    
  </div>
  
  \+ When input is negtive, still have output
  
  \- Still have 'Gradient' problem
  
  \- Still have expotional compute
  
- PReLU
 
  $$ f(x)= max(\alpha x, x) $$
  
  <div align="center">
    <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Activation%20function/PReLU.png" alt="PReLU">    
  </div>
  
  \+ There's a small gradient in negtive area
  
  \- Still cann't avoid 'Gradient' problem

### Pooling
Basically, we can use all features we get to train the network, but this will cause a great computation burden. Let's assume out input size is $ 96*96 $, filter size is 8*8, and we use 400 filters, then we'll have (96-8+1)\*(96-8+1)=7921 dimension feature map for one filter, 89\*89\*400 = 3,168,400 dimension feature maps in all.

To solve this problem, we'd like to shrink information size in the feature map. Since it's common in feature map that information at a pixel of a limit region is suitable for all this region, we may compute a specific number, mean or max among this region, this operation is called pooling.

In addition of above function, pooling usually used to keep some invarience attributes, such as rotation, translation or stretch.

- mean-pooling
  
  \+ Keep more background information

- max-pooling

  \+ Keep more texture information

- stochastic-pooling

  The elements in feature map are randomly selected according to their probability size, which computed by their value.
 
![Pooling_schematic](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Pooling/Pooling_schematic.gif){:width="80%"}
  
Pooling usually used in Object Recognition problem, which won't affect the result a lot even with some information loss, but not in task such as Super Resolution, when we want to retain information as far as possible. 

### Fully Connect
![FC](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/FullyConnect/FullyConnect-1.jpg "FC")

When we just want to find the feature and don't care where it is, we need fully connect to help us ignore the space structure. It's realized by convolution, can be seen as a convolution layer. Since it ignore the space structure, it doesn't fit in some tasks such like segmentation.

### Loss Function
The loss function quantifies the amount by which the prediction deviates from the actual values.
- Loss term
  - Gold standard loss
    
    $$ L_{01}(m)=\left\{ \begin{aligned} 0 \ \ \ & if \ m \ge0 \\ 1 \ \ \ & if \ m < 0 \end{aligned} \right. $$
    
    When prediction value right, loss is $ 0 $, when it's wrong, loss is $ 1 $. However, if we use it as our object function, we'll find it's obviously not differentiable, which means disaster for CNN.
  
  - Square loss
  
    $$ l_2(m)=(f(x;\theta)-y)^2 $$
    
    It's widely used in not only CNN, but also many other learning methods.
    
  - Cross entropy loss
  
    It's a more complicate loss function which from the perspective of information, use the concept of entropy in information theory. We offer a link for more [details](https://blog.csdn.net/tsyccnh/article/details/79163834).

- Regularization

  We may come into some [problems](#problems) when we try to train our model with a particular optimization algorithmn. In this situation, if you are facing overfitting problem, you may want to enhance the generalization ability of your network. It's wise to take a $ L_1 $ or $ L_2 $ norm on your parameters.

### Optimization
*This part is based on [An Overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/) which I highly recommend.*

Loss function make it possible for us to quantify the quality of any particular set of weights $ W $. The goal of optimization is to find $ W $ that minimizes the loss function. It's may be not difficult if we are dealing with a Convex Optimization problem. Unfortunately, Neural Networks optimization isn't a Convex Optimization problem. We may use Random Search, Random Local Search, which are all less of purpose, or we can choose Gradient descendent algorithm and it's varients, which commenly used today in backpropagation computation.

#### Gradient descendent
As we all know, a function will grow fastest along it's gradient direction. Naturally, we can use the gradient of our loss function as the *best* direction to update parameters. Analogy to hiking, this approach roughly corresponds to feeling the slope of the hill below our feet and stepping down the direction that feels steepest. 

In one-dimensional functions, the slope is the instantaneous rate of change of the function at any point you might be interested in. The gradient is a generalization of slope for functions that don’t take a single number but a vector of numbers. When the functions of interest(our loss function) take a vector of numbers instead of a single number, we call the derivatives partial derivatives, and the gradient is simply the vector of partial derivatives in each dimension.

Here, we may have [two ways](http://cs231n.github.io/optimization-1/#optimization) to compute the gradient, one is numerical, the other is analytic. Formulation given in articles usually derived from analytic way, which need matrix calculus([The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)) and Chain rule.

***Maybe an example here***

- Step size(Learning rate)

  Before we go further, we may take a look at how the gradient update our parameters. After we get our gradient value, we update our parameters [in negative gradient direction](https://blog.csdn.net/xbinworld/article/details/42886155) with a learning rate, for we wish our loss function to decrease, not increase. 
  
  Now, the gradient tells us the direction in which the function has the steepest rate of increase, but it does not tell us how far along this direction we should step. Choosing the step size (also called the learning rate) is one of the most important (and most headache-inducing) hyperparameter settings in training a neural network. In our blindfolded hill-descent analogy, we feel the hill below our feet sloping in some direction, but the step length we should take is uncertain. If we shuffle our feet carefully we can expect to make consistent but very small progress (this corresponds to having a small step size). Conversely, we can choose to make a large, confident step in an attempt to descend faster, but this may not pay off. At some point taking a bigger step even gives a higher loss as we “overstep”.
  
Next, we'll take a look at three variants of Gradient descendent, which differ in how much data we use to compute the gradient of the objective function. Depending on the amount of data, we make a trade-off between the accuracy of the parameter update and the time it takes to perform an update.
  
- **Batch gradient descent**

  Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset:
  
  $$ \theta=\theta-\eta\nabla_\theta J(\theta) $$
  
  As we need to calculate the gradients for the **whole dataset** to perform just one update, batch gradient descent can be very slow and is intractable for datasets that don't fit in memory. Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.
  
- **Stochastic gradient descent**
  
  Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example $ x^{(i)} $ and label $ y^{(i)} $
  
  $$ \theta = \theta - \eta \nabla_\theta J(\theta;x^{(i)};y^{(i)}) $$
  
  SGD algorithm avoid some similar gradient updates when dataset is large, so it could be much faster. Meanwhile, SGD performs frequent updates with a high variance that may cause the objective function to fluctuate heavily. 
  
  ![sgd_fluctuation](https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Optimization/sgd_fluctuation.png "sgd_fluctuation")
  
  While batch gradient descent converges to the minimum of the basin the parameters are placed in, SGD's fluctuation, on the one hand, enables it to jump to new and potentially better local minima. On the other hand, this ultimately complicates convergence to the exact minimum, as SGD will keep overshooting. However, it has been shown that when we slowly decrease the learning rate, SGD shows the same convergence behaviour as batch gradient descent, almost certainly converging to a local or the global minimum for non-convex and convex optimization respectively. 
  
- **Mini-batch gradient descent**

  Mini-batch gradient descent performs an update for every mini-batch of n training examples:
  
  $$ \theta=\theta-\eta\nabla_\theta J(\theta;x^{(i:i+n)};y^{(i:i+n)}) $$
  
  In this way, it a) reduces the variance of the parameter updates, which can lead to more stable convergence; and b) can make use of highly optimized matrix optimizations common to state-of-the-art deep learning libraries that make computing the gradient w.r.t. a mini-batch very efficient. Common mini-batch sizes range between 50 and 256, but can vary for different applications. Mini-batch gradient descent is typically the algorithm of choice when training a neural network and the term SGD usually is employed also when mini-batches are used. *Note*: In modifications of SGD in the rest of this post, we leave out the parameters $ x^{(i:i+n)} $; $ y^{(i:i+n)} $ for simplicity.
  
#### Challenges
  
  Mini-batch gradient descendent however, does not guarantee good convergence, and offer few challenges need to be addressed:

- Choosing a **proper learning rate** can be difficult. As talked before, a learning rate that is too small leads to painfully slow convergence, while a learning rate that is too large can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.

- Learning rate schedules try to adjust the learning rate during training by e.g. annealing, i.e. reducing the learning rate according to a pre-defined schedule or when the change in objective between epochs falls below a threshold. These schedules and thresholds, however, have to be defined in advance and are thus **unable to adapt** to a dataset's characteristics.

- Additionally, the **same learning rate** applies to all parameter updates. If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.

- Another key challenge of minimizing highly non-convex error functions common for neural networks is **avoiding getting trapped in their numerous suboptimal local minima**. Dauphin et al. argue that the difficulty arises in fact not from local minima but from saddle points, i.e. points where one dimension slopes up and another slopes down. These saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.

#### Gradient descent optimization algorithms

In the following, we'll outline some algorithms that are widely used by the deep learning community to deal with the aforementioned challenges. 

PS: We won't discuss algorithms that are infeasible to compute in practice for high-dimensional data sets, e.g. Newton's method.

- **Momentum**

  SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another, which are common around local optima. In these scenarios, SGD oscillates across the slopes of the ravine while only making hesitant progress along the bottom towards the local optimum as in left image.
  
  <div align="center">
  <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Optimization/without_momentum.gif" width="80%" alt="without_momentum"> 
  <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Optimization/with_momentum.gif" width="80%" alt="with_momentum">
  </div>
  
  Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in right image. It does this by adding a fraction γ of the update vector of the past time step to the current update vector:
  
  $$ v_i = \lambda v_{i-1} + \eta \nabla_\theta^{(j-1)} J(\theta_{(j-1)}), where v_0 = \eta \nabla_\theta J(\theta^{(0)}) $$
  
  $$ \theta^{(j)} = \theta^{(j - 1)} - v_i $$
  
  From a physical point of view, it's same effect of adding a inertia on the ball. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way (until it reaches its terminal velocity if there is air resistance, i.e. γ<1). The same thing happens to our parameter updates: The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.
  
- **Nesterov accelerated gradient**

  Since for Momentum method, it will definite take a step of $ \lambda v_{i-1} $, so it decide to take this variation first, then upgrade gradient at that point. Thus we'll get following formula:
  
  $$ v_i=\lambda v_{i-1}+\eta\nabla_\theta^{(j-1)} J(\theta^{(j-1)}-\lambda v_{i-1}) $$
  
  $$ \theta^{(j)} = \theta^{(j - 1)} - v_i $$
  
  While Momentum first computes the current gradient (small blue vector in Image 4) and then takes a big jump in the direction of the updated accumulated gradient (big blue vector), NAG first makes a big jump in the direction of the previous accumulated gradient (brown vector), measures the gradient and then makes a correction (red vector), which results in the complete NAG update (green vector). This anticipatory update prevents us from going too fast and results in increased responsiveness, which has significantly increased the performance of RNNs on a number of tasks.
  
  <div align="center">
  <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Optimization/nesterov_update_vector.png" width="80%" alt="nesterov_update_vector"> 
  </div>
  
  Now that we are able to adapt our updates to the slope of our error function and speed up SGD in turn, we would also like to adapt our updates to each individual parameter to perform larger or smaller updates depending on their importance.
  
- **Adagrad**

  Adagrad is an algorithm for gradient-based optimization: It adapts the learning rate to the parameters, performing smaller updates  for parameters associated with frequently occurring features, and larger updates for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data. Dean et al. have found that Adagrad greatly improved the robustness of SGD and used it for training large-scale neural nets at Google. Moreover, Pennington et al. used Adagrad to train GloVe word embeddings, as infrequent words require much larger updates than frequent ones.

  Previously, we performed an update for all parameters $ \theta^{(j)} $ at once as every parameter $ \theta_i^{(j)} $ used the same learning rate $ \eta $. Adagrad uses a different learning rate for every parameter $ \theta_i^{(j)} $ at every time step $ j $, we first show Adagrad's per-parameter update, which we then vectorize. For brevity, we use $ g^{(j)} $ to denote the gradient at time step $ j $. $ g_i^{(j)} $ is then the partial derivative of the objective function w.r.t. to the parameter $ \theta_i^{(j)} $ at time step $ j $:

  $$ g_i^{(j)} = \nabla_\theta^{(j)} J(\theta_i^{(j)}) $$

  The SGD update for every parameter $ \theta_i $ at each time step $ j $ then becomes:

  $$ \theta_i^{(j+1)} = \theta_i^{(j)} − \eta g_i^{(j)} $$

  In its update rule, Adagrad modifies the general learning rate η at each time step t for every parameter θi based on the past gradients that have been computed for θi:

  $$ \theta_i^{(j+1)} = \theta_i^{(j)} − \frac {\eta}{\sqrt{G_{i,i}^{(j)} + \epsilon}} g_{i}^{(j)} $$

  $ G_{i,i}^{(j)} \in R^{d×d} $ here is a diagonal matrix where each diagonal element $ i, i $ is the sum of the squares of the gradients w.r.t. $ θ_i $ up to time step $ j $, while $ \epsilon $ is a smoothing term that avoids division by zero (usually on the order of 1e−8). This means more frequent, more large the $ g_{i}^{(j)} $ is, more small the step it will take.

  As $ G^{(j)} $ contains the sum of the squares of the past gradients w.r.t. to all parameters $ \theta $ along its diagonal, we can now vectorize our implementation by performing a matrix-vector product $ \odot $ between $ G^{(j)} $ and $ g^{(t)} $:

  $$ \theta^{(j + 1)} = \theta^{(j)} - \frac{\eta}{\sqrt(G^{(j)} + \epsilon) \odot g^{(j)}} $$

  One of Adagrad's main benefits is that it eliminates the need to manually tune the learning rate. Most implementations use a default value of 0.01 and leave it at that.

  Adagrad's main weakness is its accumulation of the squared gradients in the denominator: Since every added term is positive, the accumulated sum *keeps growing during training*. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge. The following algorithms aim to resolve this flaw.

**For more Information about Gradient Descendent, i.e. More methods, Choose suitable optimizer, check [here](http://ruder.io/optimizing-gradient-descent/)**

<div align="center">
  <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Optimization/contours_evaluation_optimizers.gif" width="80%" alt="contours_evaluation_optimizers"> 
  <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Optimization/saddle_point_evaluation_optimizers.gif" width="80%" alt="saddle_point_evaluation_optimizers">
</div>

### Problems

Machine Learning doesn't mean a automatic way to performe your job, during processing, there's still many problems ought to be solved. We'll take two main problems in ML filed as a brief introduction and also some tips may help solving them.

#### Overfitting and underfitting

If you get a bad result for your experiment, you may think it's because you have a overfitting model that fit training data too well. Hold on a second, before you change your hyperparameters, I really recommend you take a look at the performance on training data, it seems not easy to get a hundred percent accuracy on your training set with CNN.

  <div align="center">
  <img src="https://raw.githubusercontent.com/leapoldzhu/Study-Notes/master/Report/MachineLearning/img/Problems/Overfitting_Underfitting.png" width="80%" alt="nesterov_update_vector"> 
  </div>

#### Gradient Explosion and vanishment

This maybe the biggest question in CNN method. As we see in backpropagation of gradient descent, caused by series of chain rule and some activation function(i.e. Sigmoid function), the gradient of weights in different layers drop dramaticlly, which could soon approach zero, make it's impossible to upgrade parameters in former layers, a.k.a. gradient vanishment. On the contrary, if your absolute value of activation function is greater than 1, it may lead to a huge number after several multiply, which called gradient explosion.

## 3 Realization
CNN codes today are based on deep learning framework, such as list below:
- [Caffe](https://github.com/BVLC/caffe)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [pyTorch](https://github.com/pytorch/pytorch)
- [keras](https://github.com/keras-team/keras)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Theano](https://github.com/Theano/Theano)

What we do is build our own structure, base on exist basic components offered by these frameworks and tune hyperparameters(learning rate, momentum, etc.) to make the best performance of out network. Different framework may base on different languages, released by different companies and always have it's own advantages. , their operation such as convolution will get same result. We take SRCNN as an example, which use matlab to clean data and mainly based on Caffe to train the model: [Code](LINK HERE)

# Reference
- Introduction
  - [深度学习简介](https://www.cnblogs.com/alexcai/p/5506806.html)

  - [卷积神经网络历史](https://blog.csdn.net/maweifei/article/details/52984920)

  - [深度学习大事件一览](https://blog.csdn.net/u013088062/article/details/51118744)

  - [Deep Learning Papers Reading Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)
  
  - [Fully connected and Local connected](https://blog.csdn.net/qq_20259459/article/details/70598929)

- Convolution
  - [卷积神经网络各层分析](https://blog.csdn.net/glory_lee/article/details/77899465)
  
  - [几种常用激活函数的简介](https://blog.csdn.net/kangyi411/article/details/78969642)
  
  - [池化](http://ufldl.stanford.edu/wiki/index.php/%E6%B1%A0%E5%8C%96)
  
  - [对CNN中pooling的理解](https://blog.csdn.net/JIEJINQUANIL/article/details/50042791)
  
  - [Deep learning：(Stochastic Pooling简单理解)](http://www.cnblogs.com/tornadomeet/p/3432093.html)
  
  - [CNN 入门讲解：什么是全连接层（Fully Connected Layer）?](https://zhuanlan.zhihu.com/p/33841176)
  
  - [一文搞懂交叉熵及其背后原理](https://blog.csdn.net/tsyccnh/article/details/79163834)
  
  - [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/optimization-1/#optimization)
  
  - [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
  
  - [An Overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)
  
  = [详解机器学习中的梯度消失、爆炸原因及其解决方法](https://blog.csdn.net/qq_25737169/article/details/78847691)
  
- Realization
  - [深度学习框架汇总](https://blog.csdn.net/wzz18191171661/article/details/70313426)
  
  - [主流深度学习框架对比](https://blog.csdn.net/zuochao_2013/article/details/56024172)
