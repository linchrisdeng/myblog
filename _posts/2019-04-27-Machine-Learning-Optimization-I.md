---
layout:     post
title:      "Machine Learning Optimization I"
subtitle:   "Gradient Descent"
date:       2019-04-27 12:00:00
author:     "LIN"
header-img: "img/in-post/post-MLOptimization-0.jpg"
tags:
    - Machine Learning
    - Optimization
catalog: true
mathjax: true
mathjax_autoNumber: true
---

## Introduction

For this blog I wanna give you some intuitions about how does optimization methods work in regression. Without further ado let's clarify a question: 

> A: What does a machine learn?
>
> B: Algorithms! Supervised/Unsupervised Learning! KNN, Naive Bayes! Trees!
>
> A: Well, these are what we human beings need to learn. The question is WHAT does a MACHINE LEARN?
>
> B: I am not sure about your problem...
>
> A: **PARAMETERS!**

Before we start talking about how machine learn parameters, we trully need to review some mathmatics.

### Taylor Series Approximation -- approximating a function

> If you are tired of reading, please check this [LINK](<https://www.khanacademy.org/math/ap-calculus-bc/bc-series-new/bc-10-11/v/maclaurin-and-taylor-series-intuition>)

Taylor series is a way to to approximate a function at one point by using an infinite number of terms. 

The basic form is Taylor Series like:

$$f(x) \approx f(a) + \frac{f'(a)}{1!}(x - a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f^{(3)}(a)}{3!}(x-a)^3 + \cdots $$

Base on this we created two patameter optimization methods: **Gradient Descent** and **Newton's Method**

> Gradient Descent: $$w^k = w^{k-1} - \alpha\nabla g(w^{k-1})$$
>
> Newton's Method: $$w^k = w^{k-1} - \frac{g'(w^{k-1})}{g''(w^{k-1})}$$



## Gradient Descent (FONC -- First Order Necessary Condition)

Let's minimize one function.

$$f(x) = 100x^2(1-x)^2 - x$$

{% include plot/GD_00.html %}

For this problem, we can easily get optimal solution by following steps:

> **Input**: differentiable function g, fixed step length $\alpha$, and set an initial point $w^0$
>
> Initial interation mark $k = 1$, and set a stopping condition $k < 1000$
>
> **Repeat iteration untill fullfill stopping condition:**
>
> $$w^k = w^{k-1} - \alpha\nabla g(w^{k-1})$$
>
> $$k = k +1$$ 

We set Initial point $w^0 = 1.2 $, step length $\alpha = 0.005$



## Newton's Method (SONC -- Second Order Necessary Condition)





