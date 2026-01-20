---
aliases: [Optimization Using Gradient Descent in One Variable]
tags: []
created: 2025-11-27, 10:56:52
modified: 2025-11-27, 17:04:02
---

# Optimization Using Gradient Descent in One Variable

To understand how to optimize functions using gradient descent, start from simple examples - functions of one variable. In this lab, you will implement the gradient descent method for functions with single and multiple minima, experiment with the parameters and visualize the results. This will allow you to understand the advantages and disadvantages of the gradient descent method.

%%
想要理解如何利用梯度下降来优化函数，我们不妨从一些简单的例子着手，比如单变量函数。
在这个实验中你将会通过梯度下降方法实现求得函数的单个或多个极小值，调整参数并通过图形观察结果。
这可以让你了解梯度下降的优缺点。
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
from w2_tools import plot_f, gradient_descent_one_variable, f_example_2, dfdx_example_2
# Magic command to make matplotlib plots interactive.
%matplotlib widget
```
<a name='1'></a>
## 1 - Function with One Global Minimum

Function $f\left(x\right)=e^x - \log(x)$ (defined for $x>0$) is a function of one variable which has only one **minimum point** (called **global minimum**). However, sometimes that minimum point cannot be found analytically - solving the equation $\frac{df}{dx}=0$. It can be done using a gradient descent method.

%%
函数 $f\left(x\right)=e^x - \log(x)$ （且 $x>0$）是一个单变量函数，它只有一个**最小点**（这被称为**全局最小值**）。
无论如何，有些时候这个最小点不能通过解析法求解方程 $\frac{df}{dx}=0$。
此时可以使用梯度下降方法。
%%

To implement gradient descent, you need to start from some initial point $x_0$. Aiming to find a point, where the derivative equals zero, you want to move "down the hill". Calculate the derivative $\frac{df}{dx}(x_0)$ (called a **gradient**) and step to the next point using the expression:

%%
要实现梯度下降，最初你需要设置一个初始点 $x_0$。
我们的目的是找到一个点，让它的导数为零，你需要移动它，让它“下山”。
计算这个导数 $\frac{df}{dx}(x_0)$ （它被称为梯度），然后使用这个表达式步入下一个点。
%%

$$x_1 = x_0 - \alpha \frac{df}{dx}(x_0),\tag{1}$$
where $\alpha>0$ is a parameter called a **learning rate**. Repeat the process iteratively. The number of iterations $n$ is usually also a parameter.

%%
其中参数 $\alpha>0$ 被称为**学习率**，
重复这个迭代处理。
迭代的次数 $n$ 一般来说也是个参数。
%%

Subtracting $\frac{df}{dx}(x_0)$ you move "down the hill" against the increase of the function - toward the minimum point. So, $\frac{df}{dx}(x_0)$ generally defines the direction of movement. Parameter $\alpha$ serves as a scaling factor.

%%
在函数的递增方向上减去 $\frac{df}{dx}(x_0)$ ，意味着你正在沿着最小化数据点移动“下山”。
所以 $\frac{df}{dx}(x_0)$ 一般来说决定了移动的方向。
参数 $\alpha$ 则负责缩放因子。
%%

Now it's time to implement the gradient descent method and experiment with the parameters!

%%
现在开始实现梯度下降方法，并且对其调参。
%%

First, define function $f\left(x\right)=e^x - \log(x)$ and its derivative $\frac{df}{dx}\left(x\right)=e^x - \frac{1}{x}$:

%%
首先，定义函数 $f\left(x\right)=e^x - \log(x)$ 和它的导数 $\frac{df}{dx}\left(x\right)=e^x - \frac{1}{x}$:
%%


```python
def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x
```

Function $f\left(x\right)$ has one global minimum. Let's plot the function:

%%
函数 $f\left(x\right)$ 有一个全局的最小值，让我们绘制这个函数。
%%

```python
plot_f([0.001, 2.5], [-0.3, 13], f_example_1, 0.0)
```

> [!result] 
(<Figure size 800x400 with 1 Axes>, <Axes: xlabel='$x$', ylabel='$f$'>)
![C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_9_1.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_9_1.png)

Gradient descent can be implemented in the following function: 

%%
下面是梯度下降的实现函数：
%%

```python
def gradient_descent(dfdx, x, learning_rate = 0.1, num_iterations = 100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
    return x
```

Note that there are three parameters in this implementation: `num_iterations`, `learning_rate`, initial point `x_initial`. Model parameters for such methods as gradient descent are usually found experimentially. For now, just assume that you know the parameters that will work in this model - you will see the discussion of that later. To optimize the function, set up the parameters and call the defined function `gradient_descent`:

%%
注意它的实现的三个参数：`num_iterations`, `learning_rate` 和最初的点 `x_initial`。
梯度下降这类方法的模型参数一般来说需要通过实验来确定。
在这里，假设你知道这个模型的运行参数，稍后我们将讨论相关问题。
那么要优化这个函数，需要设置这些参数然后调用定义的函数 `gradient_descent`。
%%

```python
num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
print("Gradient descent result: x_min =", gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)) 
```

> [!result]
Gradient descent result: x_min = 0.5671434156768685


The code in following cell will help you to visualize and understand the gradient descent method deeper. After the end of the animation, you can click on the plot to choose a new initial point and investigate how the gradient descent method will be performed.

%%
下面的代码单元将让你看到并深刻理解梯度下降方法。
在动画完成之前，你可以点击绘图选择新的初始点，以便于研究梯度下降方法是如何执行的。
%%

You can see that it works successfully here, bringing it to the global minimum point!

%%
你可以看到它如何成功的运行，并最终抵达全局最低点！
%%

What if some of the parameters will be changed? Will the method always work? Uncomment the lines in the cell below and rerun the code to investigate what happens if other parameter values are chosen. Try to investigate and analyse the results. You can read some comments below.

%%
如果参数改变了会发生什么变化呢？这个方法依然可以工作吗？
取消下面单元格的注释，然后再次运行代码观察修改了参数值之后发生了什么。
尝试对结果进行调查和分析。
你可以在下面看到一些评论。
%%

*Notes related to this animation*: 
- Gradient descent is performed with some pauses between the iterations for visualization purposes. The actual implementation is much faster.%%梯度下降在迭代过程中会进行间歇性暂停，以便可视化。%%
- The animation stops when minimum point is reached with certain accuracy (it might be a smaller number of steps than `num_iterations`) - to avoid long runs of the code and for teaching purposes.%%未了避免代码的运行时间过长，同时兼顾教学的需求，当最小点到达特定精度的时候动画会停止停止（它可能比 `num_iterations` 的步数少）%%
- Please wait for the end of the animation before making any code changes or rerunning the cell. In case of any issues, you can try to restart the Kernel and rerun the notebook.%%在动画完成前不要修改或者重新运行这个单元格。如果有任何问题，你可以尝试重启内核然后再次运行这个 notebook。%%


```python
num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.3; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.5; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.04; x_initial = 1.6
# num_iterations = 75; learning_rate = 0.04; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.05
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.03
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.02

gd_example_1 = gradient_descent_one_variable([0.001, 2.5], [-0.3, 13], f_example_1, dfdx_example_1, 
                                   gradient_descent, num_iterations, learning_rate, x_initial, 0.0, [0.35, 9.5])
```

> [!result]
![C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_15_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_15_0.png)


Comments related to the choice of the parameters in the animation above:

%%
动画中参数选择的考量：
%%

- Choosing `num_iterations = 25`, `learning_rate = 0.1`, `x_initial = 1.6` you get to the minimum point successfully. Even a little bit earlier - on the iteration 21, so for this choice of the learning rate and initial point, the number of iterations could have been taken less than `25` to save some computation time.%%你将成功得到最小化的点，即使只是早了一点点——迭代了 21 ，所以选择这个学习率和初始点，这个迭代的数会少于 25 次同时节省一些计算时间。%%

- Increasing the `learning_rate` to `0.3` you can see that the method converges even faster - you need less number of iterations. But note that the steps are larger and this may cause some problems.%%你可以看到这个方法收敛的速度更快——你需要的迭代次数更少。但是由于步长较大，可能会引发一些问题。%%

- Increasing the `learning_rate` further to `0.5` the method doesn't converge anymore! You steped too far away from the minimum point. So, be careful - increasing `learning_rate` the method may converge significantly faster... or not converge at all.%%使用该方法不再收敛，当心——随着学习率的增加这个方法可能收敛得更快，或者是根本不会收敛。%%

- To be "safe", you may think, why not to decrease `learning_rate`?! Take it `0.04`, keeping the rest of the parameters the same. The model will not run enough number of iterations to converge!%%将学习率修改为 `0.04` 并保持其他参数不变，模型在迭代次数不足的情况下无法收敛。%%

- Increasing `num_iterations`, say to `75`, the model will converge but slowly. This would be more "expensive" computationally.%%模型收敛了，但是很慢。这会让计算上更加“昂贵”。%%

- What if you get back to the original parameters `num_iterations = 25`, `learning_rate = 0.1`, but choose some other `x_initial`, e.g. `0.05`? The function is steeper at that point, thus the gradient is larger in absolute value, and the first step is larger. But it will work - you will get to the minimum point.%%函数的起始点更加陡峭，因此梯度的绝对值更大，并且第一步更长。但是它依然可以正常工作，你同样可以得到最小点。%%

- If you take `x_initial = 0.03` the function is even steeper, making the first step significantly larger. You are risking "missing" the minimum point.%%这个函数更加陡峭，第一步明显更大。你会有“丢失”最小点的风险。%%

- Taking `x_initial = 0.02` the method doesn't converge anymore...%%该方法同样不再收敛%%

This is a very simple example, but hopefully, it gives you an idea of how important is the choice of the initial parameters.

%%
这是一个非常简单的例子，但是非常有帮助，它给你一个想法，如何选择这个重要的初始化变量。
%%
<a name='2'></a>
## 2 - Function with Multiple Minima

Now you can take a slightly more complicated example - a function in one variable, but with multiple minima. Such an example was shown in the videos, and you can plot the function with the following code:

%%
现在你可以处理稍微复杂一点的例子——单变量函数，但是有多个最小点。
视频中展示了这样的例子，你可以通过下面的代码绘制这个函数。
%%

```python
plot_f([0.001, 2], [-6.3, 5], f_example_2, -6)
```

> [!result]
(<Figure size 800x400 with 1 Axes>, <Axes: xlabel='$x$', ylabel='$f$'>)
![C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_19_1.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_19_1.png)


Function `f_example_2` and its derivative `dfdx_example_2` are pre-defined and uploaded into this notebook. At this stage, while you are mastering the optimization method, do not worry about the corresponding expressions, just concentrate on the gradient descent and the related parameters for now.

%%
函数 `f_example_2` 和它的导数 `dfdx_example_2` 首先定义和传入了这个 notebook。
现阶段，在您掌握优化方法时，暂且无需顾虑对应的表达式，只需将重心放在梯度下降及其相关参数上。
%%

Use the following code to run gradient descent with the same `learning_rate` and `num_iterations`, but with a different starting point:

%%
使用下面的代码使用了相同的 `learning_rate` 和 `num_iterations` 以运行梯度下降，但是使用了不同的初始点。
%%

```python
print("Gradient descent results")
print("Global minimum: x_min =", gradient_descent(dfdx_example_2, x=1.3, learning_rate=0.005, num_iterations=35)) 
print("Local minimum: x_min =", gradient_descent(dfdx_example_2, x=0.25, learning_rate=0.005, num_iterations=35)) 
```

> [!result]
Gradient descent results
Global minimum: x_min = 1.7751686214270586
Local minimum: x_min = 0.7585728671820583


The results are different. Both times the point did fall into one of the minima, but in the first run it was a global minimum, while in the second run it got "stuck" in a local one. To see the visualization of what is happening, run the code below. You can uncomment the lines to try different sets of parameters or click on the plot to choose the initial point (after the end of the animation).

%%
结果是不同的。
两次实现，点确实都落在了某个极小值上，但是第一次运行它得到了一个全局最小值，但是在第二次运行它被“卡”在了一个局部最小值上。
运行下面的代码看看视频，究竟发生了什么。
你可以取消注释行尝试不同的参数或者点积图像选择初始化的点。
%%


```python
num_iterations = 35; learning_rate = 0.005; x_initial = 1.3
# num_iterations = 35; learning_rate = 0.005; x_initial = 0.25
# num_iterations = 35; learning_rate = 0.01; x_initial = 1.3

gd_example_2 = gradient_descent_one_variable([0.001, 2], [-6.3, 5], f_example_2, dfdx_example_2, 
                                      gradient_descent, num_iterations, learning_rate, x_initial, -6, [0.1, -0.5])
```

> [!result]
![C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_23_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_23_0.png)


You can see that gradient descent method is robust - it allows you to optimize a function with a small number of calculations, but it has some drawbacks. The efficiency of the method depends a lot on the choice of the initial parameters, and it is a challenge in machine learning applications to choose the "right" set of parameters to train the model!

%%
可以看到梯度下降法是稳健的，它可以让你用少量的计算来优化函数，但是它也有一些缺点。
这个方法的效率很大程度决定于选择的最初的参数，并且它也是机器学习应用选择正确的参数训练模型的挑战！
%%
