---
aliases: [Optimization Using Gradient Descent in Two Variables]
tags: []
created: 2025-11-27, 10:58:38
modified: 2025-11-28, 11:11:00
---

# Optimization Using Gradient Descent in Two Variables

In this lab, you will implement and visualize the gradient descent method optimizing some functions in two variables. You will have a chance to experiment with the initial parameters, and investigate the results and limitations of the method.

%%
在这个实验中，你将通过梯度下降方法实现两个变量函数的优化方法并且将其可视化。
你将尝试调整初始参数，并研究该方法的结果和局限性。
%%

# Table of Contents

- [ 1 - Function with One Global Minimum](#1)
- [ 2 - Function with Multiple Minima](#2)

## Packages

Run the following cell to load the packages you'll need.


```python
import numpy as np
import matplotlib.pyplot as plt
# Some functions defined specifically for this notebook.
from w2_tools import (plot_f_cont_and_surf, gradient_descent_two_variables, 
                      f_example_3, dfdx_example_3, dfdy_example_3, 
                      f_example_4, dfdx_example_4, dfdy_example_4)
# Magic command to make matplotlib plots interactive.
%matplotlib widget
```

<a name='1'></a>
## 1 - Function with One Global Minimum

Let's explore a simple example of a function in two variables $f\left(x, y\right)$ with one global minimum. Such a function was discussed in the videos, it is predefined and uploaded into this notebook as `f_example_3` with its partial derivatives `dfdx_example_3` and `dfdy_example_3`. At this stage, you do not need to worry about the exact expression for that function and its partial derivatives, so you can focus on the implementation of gradient descent and the choice of the related parameters. Run the following cell to plot the function.

%%
让我们探究一个简单的例子——只有一个全局最小值的双变量函数。
就像视频中讨论的函数那样，我们定义并导入了 notebook 三个内置函数—— `f_example_3` 和它的偏导数 `dfdx_example_3` and `dfdy_example_3`。
在这个阶段，你不需要担心函数和它的偏导数的表达式具体的实现，你应该关注如何实现梯度下降和如何选择相关的参数。
运行下面的单元格绘制这个函数。
%%


```python
plot_f_cont_and_surf([0, 5], [0, 5], [74, 85], f_example_3, cmap='coolwarm', view={'azim':-60,'elev':28})
```

> [!result]
(<Figure size 1000x500 with 2 Axes>,
 <Axes: xlabel='$x$', ylabel='$y$'>,
 <Axes3D: xlabel=' $x$ ', ylabel=' $y$ ', zlabel=' $f$ '>)
![C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_7_1.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_7_1.png)


To find the minimum, you can implement gradient descent starting from the initial point $\left(x_0, y_0\right)$ and making steps iteration by iteration using the following equations:
%%
要找到最小值，你需要实现梯度下降的开始的点 $\left(x_0, y_0\right)$ ，然后通过以下方程逐步迭代：
%%

$$x_1 = x_0 - \alpha \frac{\partial f}{\partial x}(x_0, y_0),$$ 
$$y_1 = y_0 - \alpha \frac{\partial f}{\partial y}(x_0, y_0),\tag{1}$$

where $\alpha>0$ is a learning rate. Number of iterations is also a parameter. The method is implemented with the following code:
%%
其中参数 $\alpha>0$ 是学习率。
迭代次数同样也是一个参数。
下面是它实现的方法。
%%

```python
def gradient_descent(dfdx, dfdy, x, y, learning_rate = 0.1, num_iterations = 100):
    for iteration in range(num_iterations):
        x, y = x - learning_rate * dfdx(x, y), y - learning_rate * dfdy(x, y)
    return x, y
```

Now to optimize the function, set up the parameters `num_iterations`, `learning_rate`, `x_initial`, `y_initial` and run gradient descent:

%%
现在开始优化这个函数，设置参数 `num_iterations`, `learning_rate`, `x_initial`, `y_initial`， 然后运行梯度下降。
%%

```python
num_iterations = 30; learning_rate = 0.25; x_initial = 0.5; y_initial = 0.6
print("Gradient descent result: x_min, y_min =", 
      gradient_descent(dfdx_example_3, dfdy_example_3, x_initial, y_initial, learning_rate, num_iterations)) 
```

> [!result]
Gradient descent result: x_min, y_min = (4.0, 4.0)


You can see the visualization running the following code. Note that gradient descent in two variables performs steps on the plane, in a direction opposite to the gradient vector $\begin{bmatrix}\frac{\partial f}{\partial x}(x_0, y_0) \\ \frac{\partial f}{\partial y}(x_0, y_0)\end{bmatrix}$ with the learning rate $\alpha$ as a scaling factor.
%%
您可以通过运行以下代码查看可视化效果。
注意，请注意，双变量梯度下降是在平面上执行步骤的，它的方向和梯度向量 $\begin{bmatrix}\frac{\partial f}{\partial x}(x_0, y_0) \\ \frac{\partial f}{\partial y}(x_0, y_0)\end{bmatrix}$ 相反，同样也是由学习率 $\alpha$ 缩放因子。
%%

By uncommenting different lines you can experiment with various sets of the parameter values and corresponding results. At the end of the animation, you can also click on the contour plot to choose the initial point and restart the animation automatically.

%%
你可以通过注释不同的行实现不同的配置以得到不同的结果。
动画结束的时候，您还可以点击等高线图选择初始点，动画将自动重新开始。
%%

Run a few experiments and try to explain what is actually happening in each of the cases.

%%
请跑一下实验，并尝试解释每种情况中究竟发生了什么。
%%

```python
num_iterations = 20; learning_rate = 0.25; x_initial = 0.5; y_initial = 0.6
# num_iterations = 20; learning_rate = 0.5; x_initial = 0.5; y_initial = 0.6
# num_iterations = 20; learning_rate = 0.15; x_initial = 0.5; y_initial = 0.6
# num_iterations = 20; learning_rate = 0.15; x_initial = 3.5; y_initial = 3.6

gd_example_3 = gradient_descent_two_variables([0, 5], [0, 5], [74, 85], 
                                              f_example_3, dfdx_example_3, dfdy_example_3, 
                                              gradient_descent, num_iterations, learning_rate, 
                                              x_initial, y_initial, 
                                              [0.1, 0.1, 81.5], 2, [4, 1, 171], 
                                              cmap='coolwarm', view={'azim':-60,'elev':28})
```

> [!result]
![C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_13_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_13_0.png)
<a name='2'></a>
## 2 - Function with Multiple Minima

Let's investigate a more complicated case of a function, which was also shown in the videos:

%%让我们研究更复杂的函数示例，如同课程视频展示那样%%

```python
plot_f_cont_and_surf([0, 5], [0, 5], [6, 9.5], f_example_4, cmap='terrain', view={'azim':-63,'elev':21})
```

> [!result]
(<Figure size 1000x500 with 2 Axes>,
 <Axes: xlabel='$x$', ylabel='$y$'>,
 <Axes3D: xlabel='$x$', ylabel='$y$', zlabel='$f$'>)
![C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_16_1.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_16_1.png)


You can find its global minimum point by using gradient descent with the following parameters:
%%
你可以使用下面的参数通过梯度下降找到全局最小点。
%%

```python
num_iterations = 100; learning_rate = 0.2; x_initial = 0.5; y_initial = 3

print("Gradient descent result: x_min, y_min =", 
      gradient_descent(dfdx_example_4, dfdy_example_4, x_initial, y_initial, learning_rate, num_iterations)) 
```

> [!result]
Gradient descent result: x_min, y_min = (0.5230322579358745, 0.5169891562802605)

However, the shape of the surface is much more complicated and not every initial point will bring you to the global minimum of this surface. Use the following code to explore various sets of parameters and the results of gradient descent.

%%
无论如何，这个函数的曲面的形状要复杂得多，而且并非每个初始点都能导向其全局最小值。
使用以下代码探索各种参数的各种组合以及梯度下降的结果。
%%


```python
# Converges to the global minimum point.
num_iterations = 30; learning_rate = 0.2; x_initial = 0.5; y_initial = 3
# Converges to a local minimum point.
# num_iterations = 20; learning_rate = 0.2; x_initial = 2; y_initial = 3
# Converges to another local minimum point.
# num_iterations = 20; learning_rate = 0.2; x_initial = 4; y_initial = 0.5

gd_example_4 = gradient_descent_two_variables([0, 5], [0, 5], [6, 9.5], 
                                              f_example_4, dfdx_example_4, dfdy_example_4, 
                                              gradient_descent, num_iterations, learning_rate, 
                                              x_initial, y_initial, 
                                              [2, 2, 6], 0.5, [2, 1, 63], 
                                              cmap='terrain', view={'azim':-63,'elev':21})
```

> [!result]
![C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_20_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_2_Optimization_Using_Gradient_Descent_in_Two_Variables_20_0.png)


You had a chance to experience the robustness and limitations of the gradient descent methods for a function in two variables. 
%%
你已经体验到梯度下降法在处理二元函数时所展现的鲁棒性与局限性。
%%
