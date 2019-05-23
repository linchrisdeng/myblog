---
layout:     post
title:      "Introduction to Binary Classification -- Perceptron "
subtitle:   ""
date:       2019-04-01 12:00:00
author:     "LIN"
header-img: "img/in-post/post-MLOptimization-5.jpg"
tags:
    - Machine Learning
    - Optimization    
    - Python
    - Classification
    - Binary
    - Perceptron
catalog: true
mathjax: true
mathjax_autoNumber: true
---



## Introduction

After reviewing and analyzing logistic regression, we break down the door of classification, in this post I will introduce another basic classification method called **Perceptron** which is also the first machine learning method I learned. 

I like the name *Perceptron*, it sounds like a complex auto-machine that perfectly match machine learning this subject. Frankly speaking, after understanding the one, it is not cool or complicated. It is the most straight forward method I have ever seen but derserve you to learn, since it is the base of some advanced algorithms like *Support Vector Machine*, currently the most popular method *Neural Network*. 

Please check this previous [**post**](<https://linchrisdeng.github.io/2019/03/26/Linear-Regression-Parameter-Estimation/>) and make sure you are familiar with my notation style. 

For linear regression, we estimate paramemters $\mathbf w_i$ to *force* our prediction $\hat{y_i} \approx y_i$. Due to this reason, sometime we may define regression as **descriptive** method (I will talk about measure metrics later, forgive the inaccuracy). However, for classification problem, our objective funtion would look like $\mathbf {w}^T\mathbf{x}_i =  0$ which means no point/data lied on the *"regression"* line. Then we can imagine a line cross the figure with two classes  points dropped on both sides like this:

<div>
    <a href="https://plot.ly/~linchrisdeng/63/?share_key=TWE6kEAzPzL16L4rTXMhCx" target="_blank" title="Perceptron" style="display: block; text-align: center;"><img src="https://plot.ly/~linchrisdeng/63.png?share_key=TWE6kEAzPzL16L4rTXMhCx" alt="Perceptron" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="linchrisdeng:63" sharekey-plotly="TWE6kEAzPzL16L4rTXMhCx" src="https://plot.ly/embed.js" async></script>
</div>

Weights were estimated by directly using scikit-learn, two classes were perfectly sepatate by a logistic/tanh like hyperplane. Now you can guess will the *perceptron* is another S curve algorithm? Let's see how does it work. 



## Perceptron Cost Function

![](/img/in-post/post-MLOptimization-6.jpg)

Let's change the view to the top, we can see a *"line"* is separating these two classes wthout cross any of them. Base on the our definition of the *"line" we have:


$$
\begin{array}{ll}
{\mathbf{w}^{T} \mathbf{x}_i>0} & {\text { if } y_{i}=+1} \ \text{(blue scatters)}
\\ 
{\mathbf{w}^{T} \mathbf{x}<0} & {\text { if } y_{i}=-1} \ \text{(orange scatters)}
\end{array}
$$


Just like what we usually do to design cost function, we summarize *"costs"* into one formula: 


$$
\begin{array}{}

\end{array}
$$

$$
g_i(\mathbf{w}) = \text{sign}(-y_{i} \mathbf{w}^{T} \mathbf{x}_i )
\\ \Downarrow
\\
g(\mathbf{w}) = \frac{1}{N}\sum^N_{i=1}\text{sign}(-y_{i} \mathbf{w}^{T} \mathbf{x}_i )
\\
\\ \Downarrow
\\ 
g(\mathbf{w}) = \frac{1}{N}\sum^N_{i=1}\text{max}(0,-y_{i} \mathbf{w}^{T} \mathbf{x}_i )
$$



However, we may find two problems here that 

1. Parameter $\mathbf{w}$ will easily *descent* to $0$
2. Cannot apply high order optimization method (Newton's) 



## ReLU VS Softplus

Here we need to mention a new pair of function which can be used in classification in machine learning and deep learning network. 

ReLU (*Rectified Linear Unit*): 


$$
f(t) = \text{max}(0, t)
$$


Softplus:


$$
\begin{array}{}
\text{soft}(s_1, s_2) = log(e^{s_1} + e^{s_2})\\ \\
f(t) = \text{soft}(0, t) = log(e^0 + e^t) =  log(1+e^t)
\end{array}
$$


<img src="/img/in-post/post-MLOptimization-7.jpg" width="500">



Returning back to the perceptron cost function then transfering it to *Softplus* version.


$$
g(\mathbf{w}) = \frac{1}{N}\sum^N_{i=1}\text{max}(0, -y_{i} \mathbf{w}^{T} \mathbf{x}_i ) = \frac{1}{N}\sum^N_{i=1}\text{soft}(0, -y_{i} \mathbf{w}^{T} \mathbf{x}_i ) \\
\Downarrow \\
g(\mathbf{w}) = \frac{1}{N}\sum^N_{i=1}\text{log}(1 + e^{-y_{i} \mathbf{w}^{T} \mathbf{x}_i} )
$$


You must be familiar with this cost function and now it can answer the question we raised in the beginning *"Why perceptron looks like a **S** curve?"*

**Welcome back to [*tanh* logistic regression](<https://linchrisdeng.github.io/2019/03/29/Logistic-Regression-tanh-Cost-Function/>) !!!**



