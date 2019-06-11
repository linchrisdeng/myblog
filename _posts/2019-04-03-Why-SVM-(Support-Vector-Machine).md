---
layout:     post
title:      "Why SVM (Support Vector Machine)? "
subtitle:   ""
date:       2019-04-03 12:00:00
author:     "LIN"
header-img: "img/in-post/SVM_00.jpg"
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

After understanding the perceptron we may have an intuition of Linear Classification and raise some question about it. **Perceptron** seems not stable I am not saying it is unreliable but lack of enough limitations which means for an optimization problem weights varied during the process and stopped within same conditions, like:

<img src="/img/in-post/SVM_01.jpg" width="300">

and this will lead problem in predicting new data. Hence we need a more concrete and rigorous method -- SVM (Support Vector Machine)



## Hyperplane and Maximum Margin

We all know that for high-dimension classification, we divided different classes by insert a **hyperplane** and if we need a stable linear hyperplane we will *"fix"* something which is **margin**.

The margin is the gap between two classes and our hyperplane will be placed in the middle of the margin. The reason of setting a maximum margin is given the more space between classes, given the better prediction ability.



<img src="/img/in-post/SVM_02.jpg" width="300">



Above plot is the basic idea of a perfect hyperplane and its "twin friends" -- Maximum Margin. To maximize margins means we need to calculate the distance $\text{d}$ between two margins. 

We defind the perfect classification hyperplane and two margins as:


$$
\begin{array}{ll}
\text{w}^T\text{x} = 0 \ \ \text{(Hyperplane)}
\\
\text{w}^T\text{x} = +1 \ \ \text{(Positive Region)}
\\
\text{w}^T\text{x} = -1 \ \ \text{(Negative Region)}
\end{array}
$$


We all know that weights vector is always perpendicular to the hyperplane and parallel margins. 

Imagining there is a straight perpendicularl ine cross two margins and leave two points


$$
\text{w}^T\text{x}_1 =  +1 \  \  \text{or} \ \ (\text{x}_1, +1)
\\
\text{w}^T\text{x}_2 =  -1 \  \  \text{or} \ \ (\text{x}_2, -1)
$$


Hence, the distance of these two points is:


$$
\mathbf{w}^T\mathbf{x}_1 - \mathbf{w}^T\mathbf{x}_2 = \mathbf{w}^T(\mathbf{x}_1 -  \mathbf{x}_2) = 2
\\
\Downarrow
\\
||\mathbf{x}_1 - \mathbf{x}_2||_2 = \frac{2}{||\mathbf{w}||_2}
$$


Therefore aiming to find maximum margin is to find the minimal normal weight $\text{w}$ with classification ability.



## Hard-Margin SVM

In the meanwhile, we need to adjust the original cost function of *perceptron*:

For Perceptron


$$
\begin{array}{ll}
{\mathbf{w}^{T} \mathbf{x}_{i}>0} & {\text { if } y_{i}=+1 }
\\ 
{\mathbf{w}^{T} \mathbf{x}_{i}<0} & {\text { if } y_{i}=-1 }
\end{array}
\\
\Downarrow
\\
\begin{array}{ll}
{y_i\mathbf{w}^{T} \mathbf{x}_{i}>0} & {\text { correctly classified }}
\\ 
{y_i\mathbf{w}^{T} \mathbf{x}_{i}<0} & {\text { wrongly classified } }
\end{array}
$$


For SVM


$$
\begin{array}{ll}
\mathbf{w}^{T} \mathbf{x}_i \geq  +1 \ \ \text{if}\ y_i = +1
\\
\mathbf{w}^{T} \mathbf{x}_i \leq  -1 \ \ \text{if}\ y_i = -1
\end{array}
\\
\Downarrow
\\
\begin{array}{ll}
{1-y_i\mathbf{w}^{T} \mathbf{x}_{i}>0} & {\text { correctly classified }}
\\ 
{1-y_i\mathbf{w}^{T} \mathbf{x}_{i}<0} & {\text { wrongly classified } }
\end{array}
$$


Now we have the cost function of SVM


$$
g_i(\mathbf{w}) = \text{max}\left(0, 1-y_i\mathbf{w}^T\mathbf{x}_i\right) = 0
\\
\Downarrow
\\
g(\mathbf{w})=\frac{1}{N} \sum_{i=1}^{N} \max \left(0,1-y_{i} \mathbf{w}^{T} \mathbf{x}_{i}\right) = 0
$$


After understanding how to maxmize the margin, we can add the margin limitation to the cost function. This gives us the hard-margin SVM:


$$
\begin{array}{ll}
\text{minimize}\  \ ||\mathbf{w}||_2
\\
\text{s.t} \ \ 
\quad \quad \quad \frac{1}{N}\sum^N_{i=1}\max \left(0,1-y_{i} \mathbf{w}^{T} \mathbf{x}_{i}\right)=0 \ \ \ i = 1,...N 
\end{array}
$$


However, this hard-margin cost of SVM can only be used to train perfectly classified dataset due to its constraint, thus we need to relax this by using **KKT Condition**, which is   one of the most important knowledge in convex optimization. 



## KKT Condition

I will make this as an individual post because it contains too much content need to be well explained.

Here is the [Link](<https://linchrisdeng.github.io/2019/04/08/Lagrange-Multipliers-and-the-KKT-Conditions/>)

## Soft-Margin SVM

After completing the KKT conditions, we can relax the conditions of the hard-margin SVM and get soft-margin SVM:


$$
g(\mathbf{w}) = ||\mathbf{w}||_2 + C\sum^N_{i=1}\text{max}\left(0, 1-y_{i} \mathbf{w}^{T} \mathbf{x}_{i}\right) 
\\
\Downarrow
\\
g(\mathbf{w}) = ||\mathbf{w}||_2 +  C\sum^N_{i=1}\text{log}\left(1+e^{1-y_{i} \mathbf{w}^{T} \mathbf{x}_{i}}\right) 
$$


where $C$ means the penalty we given to misclassified, which determines the width of the  margin (large $C \ \Rightarrow $  smaller margin, vice versa)



<img src="/img/in-post/SVM_03.jpg" width="500">

Plots downloaded from [stackoverflow: Kent Munthe Caspersen](<https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel>) 

## Summary

In this post, an imperfect one, I talked about the the disadvantages of *Perceptron* and the 

connection between margin and its original cost function. I also solved the hard and soft margin SVM cost function. As to the next month, I will try to summarize the KKT conditions and some other convex optimization methods that can be used in Machine Learning. 



----------

### References

Here is a great interpretation of KKT condition in SVM from [Arthur Gretton
Gatsby Unit, CSML, UCL](<http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/Slides5A.pdf>)

