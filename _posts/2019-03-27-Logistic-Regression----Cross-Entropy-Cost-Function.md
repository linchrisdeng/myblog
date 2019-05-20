---
layout:     post
title:      "Logistic Regression -- Cross Entropy Cost Function"
subtitle:   "{0, 1} class problem"
date:       2019-03-27 12:00:00
author:     "LIN"
header-img: "img/in-post/post-MLOptimization-2.jpg"
tags:
    - Machine Learning
    - Optimization
    - Logistic Regression
    - Python
    - Gradient Descent
    - Newton's Method
catalog: true
mathjax: true
mathjax_autoNumber: true

---

## Introduction

In this blog I will describe a popular and one of my favorite model **Logistic Regression.** And this is also the first classification method appeared in my blog, but we still call this is a *regression* rather than a *classification* method as it is a non-linear curve cross two-class.

Here is a small sample:

{% include /plot/logit_00.html %}

where we fit a linear regression by using


$$
w=\left(X^{T} X\right)^{-1} X^{T} Y
$$


However, we may find that linear regression cannot be used to classification problem, since we need a more descriptive method.



## Logistic Sigmoid Function

$$
\sigma(w^Tx)=\frac{1}{1+e^{-(w^Tx)}}
$$


$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs')
clf.fit(X, y)
y_new = clf.predict(X) 
## sklearn here aims to save time, detailed parameters optimization will be intereprted later

output_file("logit_01.html")
p3 = figure(plot_height = 300, plot_width = 300,
            tools = "")
p3.circle(x = x, y = y)
p3.toolbar.logo = None
p3.add_layout(hline)
p3.line(x = x, y = y_new, color = "red")
show(row(p2, p3))
```





{% include /plot/logit_01.html %}



> Here I use scikit-learn to estimate parameters aim to save some time



After setting threshold as 0.5 (if output > 0.5, output $\rightarrow$ 1, otherwise 0) and comparing two regressions, we can find **logistic sigmoid** increase the classification accuracy, only one plot was misclassified to 0.

Now we need to consider how to construct the cost function for logistic regression to improve the parameters prediction. The first method we use is the **least squares**.



## Least Squares Cost Function

$$
g(\mathbf{w})=\frac{1}{N} \sum_{i=1}^{N}\left(\sigma\left(\mathbf{w}^{T} \mathbf{x_i}\right)-y_{i}\right)^{2}
$$

```python
def sigmoid(t):
    return 1/(1 + np.exp(-t))
def sigmoid_least_squares(w_0, w_1):
    cost = 0
    for i in range(y.size):
        cost += (sigmoid(w_0 + w_1* x[i])- y[i])**2
    return cost/y.size

w_0 = np.linspace(20, -20, 90)
w_1 = np.linspace(20, -20, 90)
w0, w1 = np.meshgrid(w_0, w_1)
cost = sigmoid_least_squares(w0, w1)

from mpl_toolkits import mplot3d
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
%matplotlib widget
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = p3.Axes3D(fig)

for i in range(20, 180, 2):
    ax.plot_surface(w0,w1,cost, cmap='binary')
    # ax.contourf3D(w0,w1,cost)
    ax.set_xlabel('w_0')
    ax.set_ylabel('w_1')
    ax.set_zlabel('cost');

    ax.view_init(30, i)
    fig.add_axes(ax)
    filename='GIF/step'+str(i)+'.png'
    plt.savefig(filename, dpi=96)
#     p.show()

from IPython.display import IFrame
IFrame('GIF/LS_gif.gif', width=600, height=500)
```





<iframe src="https://giphy.com/embed/Y3qHdXJC0auQE69C06" width="480" height="360" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>





![](/img/in-post/post-MLOptimization-3.jpg)



This is not a convex problem, hence we cannot treat this least squares as linear regression. Then we give a derivative to the *least squares* cost


$$
\nabla g({\mathbf{w}})=2 \sum_{i=1}^{N}\left(\sigma\left({\mathbf{w}}^{T} {\mathbf{x_i}}\right)-y_{i}\right) \sigma\left(
{\mathbf{w}}^{T} {\mathbf{x_i}}\right)\left(1-\sigma\left({\mathbf{w}}^{T} {\mathbf{x_i}}\right)\right) {\mathbf{x}}_{i}
$$
Combining the gradient cost function and visualization, this a non-convex problem. And it may cause a problem that some parts of the cost function is too *flat* that means it will force the gradient or newton stop at some bad parameters. 

For this gradient function, it can easily return $0$ when $sigmoid = 1$ with large input value, let me try this Mean Square Error as cost function:

```python
alpha = 0.0001
g = 0
w = np.array([-20, 20])
iteration = 0
for t in range(20000):
    for i in range(x.size):
        grad = gradient((np.dot(w.T, X[i])), y[i])
        g += grad
#         print("gradient = {} | g = {}".format(grad, g))
#     if abs(grad) <= 0.001:
#         break
        
    w = w - alpha * g
#     print("iter = {0} | w = {1} | gradient = {2}".format(iteration, w, grad))
    iteration += 1
    
print("iter = {0} | w = {1} | gradient = {2}".format(iteration, w, grad))
```

```
iter = 20000 | w = [-19.99634777  20.00365223] | gradient = 0.0
```





{% include /plot/logit_03.html %}

You can notice that by setting a *KNOWN* start point will help us get a perfect classification. But what if we have a high-dimension problem? This means we need a better cost function.



## Cross Entropy Cost Function

This method called **Cross Entropy**, but I'd like to call **Cross Penalty**. 


$$
g_{i}(\mathbf{w})=\left\{\begin{array}{ll}{-\log \left(\sigma\left(\mathbf{w}^{T} \mathbf{x_i}\right)\right)} & {\text { if } y_{i}=1} \\ {-\log \left(1-\sigma\left(\mathbf{w}^{T}\mathbf{x_i}\right)\right)} & {\text { if } y_{i}=0}\end{array}\right.
$$


It's a little complex but it's crystal clear for visualization.



{% include /plot/logit_04.html %}



In case $y = 1$, when the sigmoid approaches to 1, cost approches to 0. Conversely, the cost grows to infinity as sigmoid approaches to 0. That's why I call this is a penalty function. We can summarize two cases into one formula:


$$
g_{i}(\mathbf{w}) = -y_i\log(\sigma(\mathbf{w}^{T} \mathbf{x_i})) - (1 - y_i)\log(1 - \sigma(\mathbf{w}^{T} \mathbf{x_i)})
$$


$$\Downarrow$$


$$
g(\mathbf{w}) = -\frac{1}{N}\sum^{N}_{i = 1}y_i\log(\sigma(\mathbf{w}^{T} \mathbf{x_i})) + (1 - y_i)\log(1 - \sigma(\mathbf{w}^{T} \mathbf{x_i)})
$$


$$\Downarrow$$


$$
g(\mathbf{w}) = -\frac{1}{N}\sum^{N}_{i = 1}y_i\mathbf{w}^{T} \mathbf{x_i} - \log(1 + e^{\mathbf{w}^{T} \mathbf{x_i}})
$$


We also get the gradient:


$$
\nabla g(\mathbf w) = -\frac{1}{N}\sum^N_{i = 1}(y_i - \sigma(\mathbf w^T\mathbf x_i))\mathbf x_i
$$


And second order gradient:


$$
\nabla^2 g(\mathbf w) = \frac{1}{N}\sum^N_{i =1}(\sigma(\mathbf w^T \mathbf x_i))(1-\sigma(\mathbf w^T \mathbf x_i))\mathbf x_i^T\mathbf x_i
$$


If the Hessian Matrix or Second-Order Taylor Approximation is non-negative or positive semi-definite, the original model would be **convex**

We can use 



$$\mathbf z^T\nabla^2g(w) \mathbf z \geq 0, \ where \ \mathbf  z \ is \ a \ non-zero \ vevtor$$



to prove this.

Hence, for cross-entropy cost function:


$$
\mathbf z^T\nabla^2g(w) \mathbf z = \frac{1}{N}\sum^N_{}(\sigma(\mathbf w^T \mathbf x_i))(1-\sigma(\mathbf w^T \mathbf x_i)) \mathbf z^T\mathbf x_i^T\mathbf x_i \mathbf z \\
\Downarrow \\
\frac{1}{N}\sum^N_{}(\sigma(\mathbf w^T \mathbf x_i))(1-\sigma(\mathbf w^T \mathbf x_i)) (\mathbf x_i \mathbf z)^2 \geq 0
$$


For more details, please read [Definiteness of a matrix](<https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Negative-definite.2C_semidefinite_and_indefinite_matrices>)



Thus logistic regression becomes a pretty straight forward convex-optimization problem, you can choose *Gradient Descent* or *Newton's Method* to find the global optimal.



{% include /plot/logit_05.html %}



I set all iteration == 5000, we can see for this small scale problem, **Newton's Method** is also the best. From now, we can build our own logistic regresion for binary classification. Next blog I will give a quick review to softmax -- logistic regression multi-classification. 



------

Code: [GItHub](<https://github.com/linchrisdeng/ML_post/tree/master/ML_02_Logistic_Regression>) & [nbviewer](<https://nbviewer.jupyter.org/github/linchrisdeng/ML_post/blob/master/ML_02_Logistic_Regression/ML_02_logit.ipynb>)

