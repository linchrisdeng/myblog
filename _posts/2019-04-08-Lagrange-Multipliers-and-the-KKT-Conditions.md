---
layout:     post
title:      "Lagrange Multipliers and the KKT Conditions"
subtitle:   ""
date:       2019-04-08 12:00:00
author:     "LIN"
header-img: "img/in-post/KKT_03.jpg"
tags:
    - Optimization    
    - Convex
catalog: true
mathjax: true
mathjax_autoNumber: true

---

## Introduction

After the illustration of [convex constrained optimization](), we prepared to pursue the topic of lagrange multipliers and the KKT conditions. The goal of KKT is to find maximum or minimum of a function subject to some constraints. Before interpretation, let' s have a look of KKT.

Normal Form:


$$
\begin{array}{ll}
\text{min}\quad \quad f(\mathbf x)
\\
\text{s.t.} \\ \quad\quad\quad\begin{array}{l}{g_{i}(x) \leq 0 \quad i = 1,...,m} \\ {h_{j}(x)=0 \quad j = 1,...,l}\end{array}
\end{array}
$$


KKT conditions:

We associate each constraint (not including the sign constraints) with a ***Lagrangian multiplier***

- Lagrangian 


$$
L(\mathbf{x, \mu}) = f(\mathbf{x}) + \mu_ig_i(x) + \lambda_j(x)
$$


-  Main Condition


$$
\nabla f\left(x^{*}\right)=\sum_{i=1}^{m} \mu_{i} \nabla g_{i}\left(x^{*}\right)+\sum_{j=1}^{\ell} \lambda_{j} \nabla h_{j}\left(x^{*}\right)
$$


- Primal Feasibility


$$
\begin{array}{l}{g_{i}\left(x^{*}\right) \leq 0, \text { for } i=1, \ldots, m} \\ {h_{j}\left(x^{*}\right)=0, \text { for } j=1, \ldots, \ell}\end{array}
$$


- Dual Feasibility


$$
\mu_{i} \geq 0, \text { for } i=1, \ldots, m
$$

- Complementary Slackness


$$
\mu_{i} g_{i}\left(x^{*}\right)=0, \text { for } i=1, \dots, m
$$




## Lagrange Multiplier

From the normal KKT conditions we may find that **Lagrangian** is the most important part in KKT conditions, which give us the solution to a constrained optimization problem orrurs when it meet both objective function and constraints. I will explain this by giving a example:


$$
\text{max} \quad f(x,y) = x^2y
\\
\text{s.t.} \quad x^2 +y^2 =5
$$
The second-order gradient of it is $-4x^2$ which is not convex, but we maximize a concave function with a convex constraint means this is convex optimization problem. Following is the visualization of objective function.



<div>
    <a href="https://plot.ly/~linchrisdeng/67/?share_key=KOEOFSRpADqwRp4fN5ZnkU" target="_blank" title="KKT_Model0" style="display: block; text-align: center;"><img src="https://plot.ly/~linchrisdeng/67.png?share_key=KOEOFSRpADqwRp4fN5ZnkU" alt="KKT_Model0" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="linchrisdeng:67" sharekey-plotly="KOEOFSRpADqwRp4fN5ZnkU" src="https://plot.ly/embed.js" async></script>
</div>

Here I need to introduce the idea of **contour line** or iso-contour, which is used to describe the relationship between two variables and the value of objective function, for more details please check this [wiki](<https://en.wikipedia.org/wiki/Contour_line>).



<img src="/img/in-post/KKT_04.jpg" width="500">



Above is the contour line of out function $x^2y$, the blue circle is the constraint $x^2 +y^2 = 5$ or $x^2 + y^2 -5= 0$. 

Base on our understanding, the optimal solution will be laid on the tangent points 
between objective function and constrint. 

<img src="/img/in-post/KKT_06.jpg" width="500">



Then I added the gradient quiver  $x, y\in [-5,5]$ of the objective function:



<img src="/img/in-post/KKT_05.jpg" width="500">



Here there are one gradient for each tangent point, and this gradient is perpendicular to the point as well. In the meanwhile, our constraint also have gradients for the tangent point with the same direction as objective function. 

Following is plot of gradient, blue arrow for constraint, red for objective function.



 <img src="/img/in-post/KKT_07.jpg" width="500">



 <img src="/img/in-post/KKT_08.jpg" width="500">



For now, I think you may understand how we derive the **Lagrangian** and **Main condition**:


$$
\begin{array}{ll}
\text{max} \quad f(x,y) = x^2y
\\
\text{s.t.} \ \ \quad g(x, y)
\end{array}

\\
\text{where} \quad g(x, y) = x^2 +y^2 - 5
$$

$$
\Downarrow\\
\nabla f(x^*, y^*) = \lambda\nabla g(x^*, y^*)
$$



## Example

Here is another example:


$$
\begin{array}{ll}
\text{min} \quad 2x_1^2 + 2x_1x_2 + x_2^2 - 10x_1 - 10x_2
\\
\text{s.t.} \quad \quad \quad x_1^2 +x_2^2 \leq 5
\\
\quad \quad \quad \quad 3x_1 + x_2 \geq 3

\end{array}
$$


- Lagrangian:


$$
L(\mathbf {x, \mu}) = 2x_1^2 + 2x_1x_2 + x_2^2 - 10x_1  - 10x_2 +
\\ 
\quad \quad \quad \mu_1(x_1^2 + x_2^2 - 5) + \mu_2(3x_1 + x_2 -3)
$$


- Main Conditions:


$$
\begin{array}{ll} 
4 x_{1}+2 x_{2}-10+2 \mu_{1} x_{1}+3 \mu_{2} &=0 \\ 2 x_{1}+2 x_{2}-10+2 \mu_{1} x_{2}+\mu_{2} &=0 

\end{array}
$$


- Primal Feasibility:


$$
\begin{aligned}
x_{1}^{2}+x_{2}^{2} \leq 5
\\
3 x_{1}+x_{2} \geq 3
\end{aligned}
$$


- Dual Feasibility


$$
\mu_1 \geq0
\\
\mu_2 \leq 0
$$


- Complementarity:


$$
\begin{array}{ll} 
\mu_{1} \cdot\left(x_{1}^{2}+x_{2}^{2}-5\right) &=0 
\\ 
\mu_{2} \cdot\left(3 x_{1}+x_{2}-3\right) &=0 
\end{array}
$$




## KKT -- Hard-Margin SVM

Because of the merge of gradient, we derived the main condition of KKT conditions, and this is how we get soft-margin SVM 


$$
\begin{array}{ll}
\text{minimize}\  \ ||\mathbf{w}||_2
\\
\text{s.t} \ \ 
\quad \quad \quad \frac{1}{N}\sum^N_{i=1}\max \left(0,1-y_{i} \mathbf{w}^{T} \mathbf{x}_{i}\right)=0 \ \ \ i = 1,...N 
\end{array}

\\
\Downarrow
\\
L(\mathbf {w}, C)=\|\mathbf{w}\|_{2}+C \sum_{i=1}^{N} \max \left(1+e^{1-y_{i} \mathbf{w}^{T} \mathbf{x}_{i}}\right)
\\
\Downarrow
\\
L(\mathbf {w}, C)=\|\mathbf{w}\|_{2}+C \sum_{i=1}^{N} \log \left(1+e^{1-y_{i} \mathbf{w}^{T} \mathbf{x}_{i}}\right)
$$




## Summary 

KKT conditions is one of the most important knowledge in convex optimization, this is also where machine learning and statistical models thrived from. In this post I briefly reviewed the KKT conditions combine with convex optimization problems, in future post I will review some models that manipulated KKT and their applications.



-------

[GitHub](<https://github.com/linchrisdeng/ML_post/tree/master/ML_05_KKT>) | [nbviewer](<https://nbviewer.jupyter.org/github/linchrisdeng/ML_post/blob/master/ML_05_KKT/lagrangeVisualization.ipynb>)

