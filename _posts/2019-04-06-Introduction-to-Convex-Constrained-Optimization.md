---
layout:     post
title:      "Introduction to Convex Constrained Optimization"
subtitle:   ""
date:       2019-04-06 12:00:00
author:     "LIN"
header-img: "img/in-post/KKT_00.jpg"
tags:
    - Optimization    
    - Convex
catalog: true
mathjax: true
mathjax_autoNumber: true

---

## Introduction

[Last post](<https://linchrisdeng.github.io/2019/04/03/Why-SVM-(Support-Vector-Machine)/>), I discussed the SVM's soft-margin cost function without detailed interpretation , in this and next posts I will give an intuition of Duality of Convec Optimization problems and KKT conditions.



Before explain KKT, we need to think about why we need this tansformation. For **unconstrined convex optimization** like MSE, cross-entropy cost, etc., we will find the *optimal solution*, which fullfill our stop conditions: gradient == 0, enough iterations ... However, after introducing the *margin* to the  the original perceptron, we face a problem that how to optimize a cost function with constraints? Base on our limited experience and knowledge we all know that we need to summarize the constrained convex problem into a one-line convex cost function, the method we use is the Karush–Kuhn–Tucker **(KKT) conditions**, but before this let's illustrate what is **Convex**.



## Convex Problems

In previous posts, I have mentioned what is convex and how to implement *descent methods* to it.  Here we need to dive into the definition of **convex** with **constraints**

- Convex Combination

For any $\mathbf{x, y} \subseteq \mathbb{R}^{n}$, a *convex combination* of them is $\mathbf{z}=\lambda \mathbf{x}+(1-\lambda) \mathbf{y}, \lambda \in[0,1]$ 

- Convex Set

A set is convex if it contains all convex combinations of any two points $\mathbf{x,y }$ and for $x = \sum^k_i\lambda_ix_i$ where all $x_i \in S$ and $\lambda_i \geq0$ satisfying $\sum^k_i\lambda_i =1$. 



<img src="/img/in-post/KKT_01.jpg" width="300">

We can clearly find in non-convex set, between $\mathbf{x, y}$ not all points locates in this combination.



- Convex Function

  - $f$ is convex if :


  $$
  \forall x_{1}, x_{2} \in X, \forall \lambda \in[0,1] : \quad f\left(\lambda x_{1}+(1-\lambda) x_{2}\right) \leq \lambda f\left(x_{1}\right)+(1-\lambda) f\left(x_{2}\right)
  $$




  -  is strictly convex if:


$$
\forall x_{1}, x_{2} \in X, \forall \lambda \in[0,1] : \quad f\left(\lambda x_{1}+(1-\lambda) x_{2}\right) < \lambda f\left(x_{1}\right)+(1-\lambda) f\left(x_{2}\right)
$$



<img src="/img/in-post/KKT_02.jpg" width="300">



> **Convex** means stand on one point you can "see" any points on this set

## Function Convexity

If $f(x)$ is a convex second-order differentiable function then its *Hessian Matrix* must be **PSD** (Positive Semi-definite) throughout the defined region.

### Proposition

1. If $f_1, ..., f_m$ are convex functions, then $f(x) = \text{max}\{f_1(x), ...f_m(x)\}$ is a convex function. 

   eg: 

    $$|x|, \text{max}\{a_i^Tx+b_i\}$$ 



1. If $f_1, ..., f_m$ are concave functions, then $f(x) = \text{min}\{f_1(x), ...f_m(x)\}$ is a concave function.

2. If $\lambda_1,...,\lambda_m\geq0$ and functions $f_1,.,,,f_m$ are convex (concave), then $a_1f_1+  \cdots + a_mf_m$ is a convex (concave) function

   eg: 

$$x_1^2 + x_2^2, e^x + |x|$$



> **Convex Optimization:**
>
> 􏰀 **Minimize** a **convex** function over a **convex** feasible region
> 􏰀 **Maximize** a **concave** function over a **convex** feasible region



## Constraints Convexity

1. If we have constraint $g(x) ≤ 0$, and $g(x)$ is **convex**, then this is a **convex** constraint 

2. If we have constraint $g(x) ≥ 0$, and $g(x)$ is **concave**, then this is a **convex** constraint

3. **Linear** constraints are always **convex** constraints 

4. Sometimes, even if a constraint doesn’t appear to be in the above form, it still could be a convex constraint


For point 4, there is a example:

$$x^3 -  1 \leq 0$$

$x^3$  is obviously a convex function, however this constraint defines a convex feasible region ($x<1$) thus is a convex constraint.



## Convex Optimization

$$
\begin{array}{ll}
\text{min} \quad 2x_i^2 + x_1x_2 + x_2^2 - 5x_1 - 5x_2
\\
\text{s.t.} \quad \quad \quad \quad x_1^2 + x_2^2 \leq 5
\\
\quad \quad \quad \quad \quad 2x_1+x_2 \geq 5
\end{array}
$$



Above is a convex optimization problem, however if we change the first constraint to $x_1^2 +x_2^2 \geq 5$, it won't be a convex constraint. 


$$
\begin{array}{ll}
\text{max} \quad \quad  xyz
\\
\text{s.t.} \quad \quad  x+2y+3z \leq3
\\
\quad \quad \quad \quad   x, y, z \geq 0
\end{array}
$$


> Convex optimization: maximize a concave or minimize a convex in a convex region

This example, we unfortunately maximize a convex function, but we can transform this into a **concave** function:


$$
\begin{array}{ll}
\text{max} \quad \quad  \text {log}(xyz)
\\
\text{s.t.} \quad \quad  x+2y+3z \leq3
\\
\quad \quad \quad \quad   x, y, z \geq 0
\end{array}
$$


In real world optimization problem, transformation will be applied to more complex functions and constraints, in machine learning field, we don't need to worry too much about these *tricky* methods.

