---
layout:     post
title:      "Linear Regression -- Parameter Estimation"
subtitle:   "Linear Regression"
date:       2019-03-26 12:00:00
author:     "LIN"
header-img: "img/in-post/post-MLOptimization-1.jpg"
tags:
    - Machine Learning
    - Optimization
    - Linear Regression
    - Python
catalog: true
mathjax: true
mathjax_autoNumber: true

---

## Introduction

We have reviewed **Gradient Descent** and **Newton's Method** in previous blog. It's time to apply optimization method into real problems -- **Linear Regression**.

### Notation

Input $X$ is a dataset with $N$ rows, $P+1$ columns:  


$$
X=\left( \begin{array}{c}{\mathbf{x}_{1}^{T}} \\ {\mathbf{x}_{2}^{T}} \\ {\vdots} \\ {\mathbf{x}_{n}^{T}}\end{array}\right)=\left( \begin{array}{cccc}{1} & {x_{11}} & {\cdots} & {x_{1 p}} \\ {1} & {x_{21}} & {\cdots} & {x_{2 p}} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {1} & {x_{n 1}} & {\cdots} & {x_{n p}}\end{array}\right)
$$


Output $Y$: 

$$Y = [y_1, y_2, ..., y_n]^T$$  

$N + 1$ parameters: 

$$W = [w_0, w_1, w_2, ..., w_n]^T$$  

Normal form of linear regression: 

$$y_p \approx w_0 + w_1x_{p, 1} + w_2x_{p,2} + \cdots + w_nx_{p,n}$$

or 

$$y_p = w_0 + w_1x_{i, 1} + w_2x_{i,2} + \cdots + w_nx_{i,n} + \varepsilon_i, \ \ \ \ \ \ \ i = 1,...n$$   

where $\varepsilon$ is residual (error) to the fitness line 

Full equation:

$$Y \approx W^TX$$ 



## Cost Function

Cost function is a way that we can measure the fitness of our regression.

Normally we use **Least Squares** for regression problem.

![least_squares](/img/in-post/ML_Optimization/least_squares.jpg)

We aim to minimize the distance between data and regression line.
$$
g(w) = \sum^N_{i = 1}(w^T x_i - y_i)^2
$$
Then we can try to expand the equation
$$
g(w) = \sum_{i = 1}^N((w^Tx_i)^2 + y_i^2 - 2w^Tx_iy_i)
$$
Regardless of the dataset, the least square cost function for linear regression is always convex.

It means for X with one dimension, least square will be a U curve. To compute the gradient of least squares, which gives:


$$
\nabla g(w) = 2\sum^N_{i = 1}(w^Tx_i)x_i - 2y_ix_i = 0
$$

$$
w^T = \sum^N_{i=1}y_ix_i(\sum^N_{i = 1}x_i^2)^{-1}
$$

$$
w = (X^TX)^{-1}X^TY
$$

```python
# import the dataset
data = np.asarray(pd.read_csv('student_debt_data.csv',header = None))
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)

# pad input with ones
o = np.ones((len(x),1))
x_new = np.concatenate((o,x),axis = 1)


w =np.matmul(inv(np.matmul(x_new.T, x_new)), np.matmul(x_new.T, y))      # weights learned by solving linear system
w
```

```output
array([[-1.60729045e+02],
       [ 8.03244175e-02]])
```



{%include /plot/Reg_00.html%}



------

Code: [Github](<https://github.com/linchrisdeng/ML_post/tree/master/ML_01_Regression>), [nbviewer](<https://nbviewer.jupyter.org/github/linchrisdeng/ML_post/blob/master/ML_01_Regression/ML_01_regression.ipynb>)

