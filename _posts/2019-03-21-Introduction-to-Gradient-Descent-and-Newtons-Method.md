---
layout:     post
title:      "Introduction to Gradient Descent and Newton's Method"
subtitle:   ""
date:       2019-03-21 12:00:00
author:     "LIN"
header-img: "img/in-post/post-MLOptimization-0.jpg"
tags:
    - Machine Learning
    - Optimization
    - Python
    - Gradient Descent
    - Newton's Method
catalog: true
mathjax: true
mathjax_autoNumber: true

---

## Introduction

For this blog I wanna give you some intuitions about how does optimization methods work. Without further ado let's clarify a question: 



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



We set initial point $w^0 = 1.2 $, step length $\alpha = 0.01$

```
iter = 0 | w = 1.2 | gradient: 66.19999999999993
iter = 1 | w = 1.1338 | gradient: 37.45960258879995
iter = 2 | w = 1.0963403974112 | gradient: 24.194635073345353
...
iter = 31 | w = 1.005010564582171 | gradient: 0.017226688501068566
iter = 32 | w = 1.00499333789367 | gradient: 0.013677433129146266
iter = 33 | w = 1.0049796604605408 | gradient: 0.010859695382237078
iter = 34 | w = 1.0049688007651585 | gradient: 0.008622611509167655
```

After 34 iterations we can get the global optimal point $w \approx 1, f(x) \approx -1.0025$



{% include plot/GD_01.html %}



What if we change the start point to $w=  -0.4$



```
iter = 0 | w = -0.4 | gradient: -202.60000000000002
iter = 1 | w = -0.1974 | gradient: -66.9368713696
iter = 2 | w = -0.1304631286304 | gradient: -38.19322823380614
...
iter = 36 | w = 0.005020417353844365 | gradient: -0.010988668449965933
iter = 37 | w = 0.005031406022294332 | gradient: -0.008856875366752504
```



{% include plot/GD_02.html %}



After 37 iterations, it reached local optimal $w\approx 0.005, f(x) \approx -0.0025$

I will talk about other optimization method and choose step length and start point in future blogs.



Overall, we can understand that:



$$w^k = w^{k-1} - \alpha\nabla g(w^{k-1})$$ 



$$\frac{w^k - w^{k-1}}{\alpha} = \nabla g(w^{k-1})$$



Gradient Descent is a way that by keeping gradient (tangent for 2-D problem), step length $\alpha$ and start point to update *"optimial"* solution of a **convex** model.



## Newton's Method (SONC -- Second Order Necessary Condition)

Newton's method AKA SONC, it means we use second order Taylor approximation to find optimial solution. 

Second-Order Taylor Approximation



$$f(x) \approx f(a) + \frac{f'(a)}{1!}(x - a) + \frac{f''(a)}{2!}(x-a)^2 \ \ \ \ \ \ \ (1)$$



$$f(x) \approx \frac{1}{2}f''(a)x^2 + [f'(a) - af''(a)]x + [f(a) -  af'(a) +\frac{1}{2}a^2f''(a)]  \ \ \ \ \ \ \ (2)$$



We keep use  $$f(x) = 100x^2(1-x)^2 - x$$, and still set initial point $w^0 = 1.2$



{% include plot/NT_00_0.html %}



You will notice that I make a black point and a new red point which become the next *start point* $w$. 



{% include plot/NT_00_1.html %}



The basic idea of Newton's Method is given a start point, find its Second-Order Taylor Approximation, then find the nearest stationary point of the approximation to update $w$. 

From equation $(2)$ 

$$\frac{d}{dx}f(x) \approx f''(a)x + f'(a) - af''(a) = 0 \ \ \ \ \ (3)$$

$$x = a - \frac{f'(a)}{f''(a)} \ \ \ \ \ (4)$$



> **Newtons's Method**
>
> **Input:** twice differentiable gunction $g$, and set an initaial point $w^0$
>
> **Repeat following derivation untill met stopping conditions**:
>
> $w^k = w^{k-1} - [\nabla^2g(w^{k-1})]^{-1} \nabla g(w^{k-1})$ 
>
> k += 1



This is the derivation process of Newton's method. We can apply this to find the optimal solution.



```
iter = 0 | w = 1.2 | gradient: 66.19999999999993
iter = 1 | w = 1.064344262295082 | gradient: 14.459521749617807
iter = 2 | w = 1.0131023128030214 | gradient: 1.7243646338414464
iter = 3 | w = 1.0051165101084196 | gradient: 0.03906280448109101
```



{% include plot/NT_00.html %}



Then we change start point at $w= -0.4$ 

```
iter = 0 | w = -0.4 | gradient: -202.60000000000002
iter = 1 | w = -0.16766055045871558 | gradient: -53.283325465105406
iter = 2 | w = -0.04514894913826514 | gradient: -11.289659538390302
iter = 3 | w = -0.001156095577173201 | gradient: -1.2320216676998228
iter = 4 | w = 0.004961528354188844 | gradient: -0.022415532619628586
```



{% include plot/NT_01.html %}



We may find compare with Gradient Descent, Newton's Method is much faster!!!

However here are some points we need to pay attentation to.



### Cons



1. Cannot guarantee convergency 



For Newton's method, it doesn't like Gradient Descent directed by gradient, it always tend to find the nearest **stationary point** point that means it may be trapped into **saddle point**.



```
iter = 0 | w = 0.6 | gradient: -10.600000000000023
iter = 1 | w = 0.6106 | gradient: -11.518840393599987
iter = 2 | w = 0.6221188403936 | gradient: -12.483420186573682
...
iter = 52 | w = 1.004865440928985 | gradient: -0.012662234052754684
iter = 53 | w = 1.0048781031630378 | gradient: -0.010055401587351298
iter = 54 | w = 1.0048881585646252 | gradient: -0.00798511133473312
```



{% include plot/GD_03.html %}



```
iter = 0 | w = 0.6 | gradient: -10.600000000000023
iter = 1 | w = 0.4795454545454543 | gradient: 1.0420313673929797
iter = 2 | w = 0.4900183490741459 | gradient: -0.0022327095634864236
```



{% include plot/NT_02.html %}



1. Computationally Expensive



Also we have to use two differentiatd function in each iteration. As to a high dimensional problem like



$$f(x, y , z) = 2x^2 + 2y^2 + z^2 - 2xy - 2xz - 6y + 7$$



We first have FONC:



$$\nabla f(x, y, z) = (4x-2y-2z,4y-2x-6, 2z-2x)$$ 



Then we have second-order Taylor Approximation (Hessian Matrix):



$$\nabla^2f = \begin{pmatrix} 4 &-2& -2 \\ -2 &4 &0 \\ -2 &0 & 2\end{pmatrix}$$



1. May cause low accuracy (compared with Gradient Descent)



For GD, we can set a small step size $\alpha$ or step a adaptative one to adjust the learning rate. But for 

## Summary

In this blog I talked about the basic understanding and give some intuitions of two parameters optimization methods: **Gradient Descent** and **Newton's Method**. 

**Gradient Descent:** 

> Pros: Accurate
>
> Cons: Slow

**Newton's Method:**

> Pros: Faster than GD
>
> Cons: Need to calculate Hessian Matrix in each iteration, May cause more noise

However,  application in real machine learn or data mining production circumstance is another more complicated problem. In the next few blogs, I will interpret how to deal with regression and classification.



------

Code: [GitHub](<https://github.com/linchrisdeng/ML_post/tree/master/ML_00_GD_NT>), [nbviewer](<https://nbviewer.jupyter.org/github/linchrisdeng/ML_post/blob/master/ML_00_GD_NT/ML_00_GD_NT.ipynb>)





