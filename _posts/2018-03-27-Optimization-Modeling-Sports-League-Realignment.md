---
layout:     post
title:      "Optimization Modeling: Sports League Realignment"
subtitle:   ""
date:       2018-03-27 12:00:00
author:     "LIN"
header-img: "img/in-post/post-NHL-0.jpg"
tags:
    - Optimization
    - Model
    - AMPL
    - R
    - Sports
catalog: true
mathjax: true
mathjax_autoNumber: true
---



##  Introduction

For MLB, NBA, NFL, NHL these four big leagues in North America normally consists of 30 to 32 teams. These teams will play same number of home and away games in the same division, and against some teams in other conferences and divisions.

Take the National Hockey League (NHL) as an example, prior to 2013, the league was structured into two 15-team conferences, each subdivided into three 5-team. This kind of assignment is called **League Structure**.

![NHL 2008-2012](https://ws3.sinaimg.cn/large/006tNc79ly1g2vhqa9bd9j310o04u0tb.jpg)

But for 2013-2014 season, the league realign the number of subdivision to 4.

![NHL 2013-2014](https://ws1.sinaimg.cn/large/006tNc79ly1g2viahuipbj30f302uaa6.jpg)

Hence for this optimization problem we need to focus on how to build a model that will support the construction of a efficient league structure.

## League Structure Evaluation

The purpose of constructing a league structure is to minimize the total travel distance for a whole season. We set $S$ as the league structure, the measure or objective function $D(S)$ is defined as the total distances over all games between pairs $(i, j)$ teams or cities. Base on above, let $d(i,j) $ as distance between teams or home cities $(i, j)$ , $g(i, j)$ as the number of games between pair $(i, j)$ in a season. The objective can be defined as: 

$$D(S) = \sum_{(i,j)}d(i, j)g(i,j)$$



## Mixed Integer Programming (MIP)

Our realignment (league structure) is a common Mixed Integer Progarmming which require one additional condition that at least one of the variables can and only take integer value. For NHL teams whther to be assigned to one of subdivision is a binary (0 or 1) problem. 

The basic form of MIP: 

Minimize					

$$\beta^T X$$

subject to

$$AX \le b, X\geq 0$$

Where



$$X = (x_1, x_2, x_3, ..., x_n)^T, x_i \in X \ are \ restricted\  to\  integer $$

$$\beta = (\beta_1, \beta_2, \beta_3,..., \beta_n)^T$$

$$ A = \begin {pmatrix}a_{11} &...  &a_{1n}\\ \vdots &\ddots &\vdots \\a_{n1} &... &a_{nn} \end {pmatrix}$$

$$b = (b_1, b_2, b_3,..., b_n)^T$$



## Minimizing the total league travel distance MIP

- $T$ is a set of $n$ teams or cities
- $S$ is a set of $s$ divisions
- $D = (D_{ij} = d(i, j): i, j \in T)$ is a $n \times n$ matrix that contains distance between each pair of cities
- G is a $s \times s$ matrix. $G_{u, v}$ represent average number of games between division $u$ and $v$, it was calculated by the understanding the schedule of NHL, details could be seen at [Season structure of the NHL](<https://en.wikipedia.org/wiki/Season_structure_of_the_NHL>)
- Set $b$ as the number of teams assigned into each divisions



> **2008 - 2012** 
>
> NHL was a 2-conference, 6-division league where $n = 30, s = 6, b = [5, 5, 5, 5, 5, 5]$

$$ G = \begin{pmatrix} 3 & 2 & 2 & 0.6 & 0.6 & 0.6 \\ 2 & 3 & 2 & 0.6 & 0.6 & 0.6 \\ 2 & 2 & 3 & 0.6 & 0.6 & 0.6 \\ 0.6 & 0.6 & 0.6 & 3 & 2 & 2 \\ 0.6 & 0.6 & 0.6 & 2 & 3 & 2 \\   0.6 & 0.6 & 0.6 & 2 & 2 & 3 \\\end{pmatrix}$$





> **2013 - 2014**
>
> NHL had 4 divisions where $n = 30, s = 4, b = [7, 7, 8, 8]$

$$G = \begin{pmatrix} \frac{15}{7} & \frac{3}{2} & 1 & 1 \\ \frac{3}{2} & \frac{15}{7} &1 & 1 \\ 1 & 1 & \frac{14.5}{6} & \frac{3}{2} \\ 1 & 1 & \frac{3}{2} & \frac{14.5}{6} \end{pmatrix}$$



### Formulate MIP

Set $x_{iu} = 1$ if team i in division j else $x_{iu} = 0$. Hence, we can get objective function as

minimize

$$ F(x) = \sum_{i,j \in T} \sum_{u,v \in S} x_{iu} \cdot x_{jv} \cdot D_{ij} \cdot G_{uv}$$

> However, this may lead objective function $F(x) = 0$,  or one of the team will be assigned to multiple divisions, then we need a more rigorous term to define the assignment of each teams corresponding to each divisions.

Set $Y_{ijuv}$ as follows: for each pair of teams $(i,j)$ and each pair of divisions $(u, v)$, if $y_{ijuv} = 1$ means team $i$ is assigned to division $u$, team $j$ is assigned to division $v$ 

- $\sum_{u, v \in S} Y_{ijuv} = 1$ for each pair of cities $(i, j)$
- $\sum_{i, j \in T} Y_{ijuv} = b_u \cdot b_v$ for each pair of divisions $(u, v)$

For now we can summarize out MIP optimization:

minimize 

$$\sum_{i,j\in T; u,v\in S} Y_{ijuv} \cdot D_{ij}  \cdot G_{uv}$$

s.t.

$$\sum_{i \in T} x_{iu} = b_u \ for \ u \in S $$

> number of teams in each division must fullfill requirment

$$\sum_{u \in S}  x_{iu} = 1 \ for \  i \in T$$

> each team can only be assigned into one division

$$\sum_{u, v \in S} y_{ijuv} = 1 \ for \ each  \ pair \ of \ teams \ (i, j) $$

$$\sum_{i, j \in T} y_{ijuv} = b_u \cdot b_v\ for \ each \ pair \ of \ divisions \ (u, v)$$



> for any two divisions, will have $b_u \cdot b_v$ pairs of teams

$$y_{ijuv} \leq x_{iu} \ \ \ i,j \in T \ u,v \in S$$

$$y_{ijuv} \leq x_{jv} \ \ \ i,j \in T \ u,v \in S$$

> $y_{ijuv} = 1$ when both $x_{iu}, x_{jv} = 1$



## Application

Base on the above MIP, we can apply this to our own division schedule.

### Roller Derby

A organizing committee for a new Roller Derby sports league
is in the process of assigning 16 teams to four divisions in such a way that overall league travel is minimized. The 16 teams are located in the following cities.

| indianapolis,   in | columbus, oh   | cincinnati, oh | st. paul, mn     |
| :----------------: | -------------- | -------------- | ---------------- |
|   milwaukee, wi    | memphis, tn    | toledo, oh     | grand rapids, mi |
|  bloomington, il   | louisville, ky | rockford, il   | fort wayne, in   |
|  cedar rapids, ia  | topeka, ks     | green bay, wi  | sioux falls, sd  |

And what if we assigned **Fort Wanye, IN** and **Indianapolis, IN** into a same division. How  does this new constraint affect the total distance and alignment ?

Details can be seen in my [GitHub](https://github.com/linchrisdeng/Optimization-Modeling-Sports-League-Realignment) or [here](<https://htmlpreview.github.io/?https://github.com/linchrisdeng/Optimization-Modeling-Sports-League-Realignment/blob/master/Realignment.html#model>)

### Update

- In R, [DataScienceToolkit](<http://www.datasciencetoolkit.org/about>) will not support  `geocode` function `dsk` API anymore, please register, please register [Google GeoCoding API](<https://developers.google.com/maps/documentation/geocoding/start>)
- To solve this MIP Optimization problem I use [AMPL](<https://ampl.com/products/ampl/>) which is a optimization modeling system support (CPLEX, Gurobi, Xpress) solver, but it is not free to use except [30-day trial](<https://ampl.com/products/ampl/ampl-for-students/>) for students





## References

1. [Mix Integer Programming](<https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Mixed_Integer_Programming.pdf>)

2. Macdonald, B. and Pulleyblank, W., 2014. Realignment in the NHL, MLB, NFL, and NBA. *Journal of Quantitative Analysis in Sports*, *10*(2), pp.225-240.



