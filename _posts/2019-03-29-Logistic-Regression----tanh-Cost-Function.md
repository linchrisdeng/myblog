---
layout:     post
title:      "Logistic Regression -- Hyperbolic Tangent (tanh) Cost Function"
subtitle:   "{-1, +1} class"
date:       2019-03-29 12:00:00
author:     "LIN"
header-img: "img/in-post/post-MLOptimization-4.jpg"
tags:
    - Machine Learning
    - Optimization
    - Logistic Regression
    - Python
    - Gradient Descent
    - Newton's Method
    - Linear
catalog: true
mathjax: true
mathjax_autoNumber: true
---

## Introduction

This post aims to give another cost function for **Logistic Regression** to solve label or target value as $y_p = \{-1, 1\}$. Sometimes the target value is arbitary and the most important reason has been proved by **Yann LeCun** in his paper **[Efficient BackProp](<http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>)** 

> Hyperbolic Tangent has larger gradient, which means it can convergence faster than sigmoid function.


$$
 \sigma(z)=\frac{1}{1+e^{-z}}
\Rightarrow 
\nabla g_{\sigma}(z) = \sigma(z)(1-\sigma(z)) \\

{tanh(z)} = 2\sigma(z) - 1
\Rightarrow 
\nabla g_{tanh}(z) = 2\sigma(z)(1-\sigma(z))
$$


{%include /plot/tanh_00.html%}



Hence, it might be a good idea to swap $y_i  =0$ to $y_i = -1$.



## tanh Cost Function

If you are not familiar with cross-entropy please check this [post](<https://linchrisdeng.github.io/2019/03/27/Machine-Learning-Optimization-II-2/>). In complete analogy, we can build a cross-entropy for tanh cost function.

For sigmoid:


$$
g_{i}(\mathbf{w})=\left\{\begin{array}{ll}{-\log \left(\sigma\left(\mathbf{w}^{T} \mathbf{x}_{\mathbf{i}}\right)\right)} & {\text { if } y_{i}=1} \\ {-\log \left(1-\sigma\left(\mathbf{w}^{T} \mathbf{x}_{\mathbf{i}}\right)\right)} & {\text { if } y_{i}=0}\end{array}\right.
$$


We can transform $tanh(z)$ to $\sigma(z)$


$$
\sigma(z) = \frac{tanh(z + 1)}{2}\\
\Downarrow \\

\begin{aligned} \tanh (z) & \approx+1 \Longleftrightarrow \sigma(z) \approx 1 \\ \tanh (z) & \approx-1 \Longleftrightarrow \sigma(z) \approx 0 \end{aligned} 
$$


For tanh:


$$
g_{i}(\mathbf{w})=\left\{\begin{array}{ll}{-\log \left(\sigma\left(\mathbf{w}^{T} \mathbf x_i\right)\right)} & {\text { if } y_{i}=+1} \\ 
{-\log \left(1-\sigma\left(\mathbf{w}^{T}\mathbf x_i\right)\right)} & {\text { if } y_{i}=-1}\end{array}\right.
$$


Like before, we need to summarize the cost functions into a one-line form, however, this time it's a little tricky to follow normal Log Error form to re-write the cost function. There is a usefull tip as follows:


$$
1-\sigma(z)=1-\frac{1}{1+e^{-z}}=\frac{e^{-z}}{1+e^{-z}}=\frac{1}{1+e^{z}}=\sigma(-z)
$$


We can use $y_i = +1 \ or \ -1$  to create the one-line form:


$$
g_i(\mathbf w) = -\log \left(\sigma\left(y_i\mathbf{w}^{T} \mathbf x_i\right)\right)\\
\Downarrow\\
g_i(\mathbf w) = \log \left(1 +e^{ -y_i\mathbf{w}^{T} \mathbf x_i}\right) \\
\Downarrow\\
g(\mathbf w) = \frac{1}{N}\sum^N_{i = 1}\log \left(1 +e^{ -y_i\mathbf{w}^{T} \mathbf x_i}\right)
$$


 This is the cost function for {-1, +1} linear classification problem, and like {0, +1} cross-entropy cost function, is convex as well:


$$
\begin{align*} 
& \nabla g(\mathbf w) = -\frac{1}{N}\sum^N_{i = 1} \frac{e^{ -y_i\mathbf{w}^{T} \mathbf x_i}}{1 +e^{ -y_i\mathbf{w}^{T} \mathbf x_i}}{y_i\mathbf{x}_i} = -\frac{1}{N}\sum^N_{i = 1} \sigma(-y_i\mathbf{w}^{T} \mathbf x_i)y_i\mathbf{x}_i
\\
& \nabla^2 g(\mathbf w) = -\frac{1}{N}\sum^N_{i = 1} \sigma \left(-y_i\mathbf{w}^{T} \mathbf x_i\right)\left(1 - \sigma(-y_i\mathbf{w}^{T} \mathbf x_i)\right)\mathbf{x}^T_i \mathbf{x}_i 
\\
& \mathbf{z}^T\nabla^2 g(\mathbf w)\mathbf{z} = -\frac{1}{N}\sum^N_{i = 1} \sigma \left(-y_i\mathbf{w}^{T} \mathbf x_i\right)\left(1 - \sigma(-y_i\mathbf{w}^{T} \mathbf x_i)\right)\mathbf(\mathbf{x}_i\mathbf{z})^2
\end{align*}
$$


The minimal eigenvalue of second-order Taylor Approximation is always  $\geq 0$.



## Summary

Converting {0, 1} target to  {-1, +1} by using **Hyperbolic Tangent** (*tanh*) can be defined as a faster version **Logistic Regression** (*sigmoid*). 



-----

More Details please check [GitHub](<https://github.com/linchrisdeng/ML_post/tree/master/ML_03_tanh>) and [nbviewer](<https://nbviewer.jupyter.org/github/linchrisdeng/ML_post/blob/master/ML_03_tanh/Logit_tanh.ipynb>) 