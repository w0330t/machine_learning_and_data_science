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


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    File ~/Projects/machine_learning_calculus_w2_lib/.venv/lib/python3.12/site-packages/matplotlib/backends/registry.py:407, in BackendRegistry.resolve_gui_or_backend(self, gui_or_backend)
        406 try:
    --> 407     return self.resolve_backend(gui_or_backend)
        408 except Exception:  # KeyError ?


    File ~/Projects/machine_learning_calculus_w2_lib/.venv/lib/python3.12/site-packages/matplotlib/backends/registry.py:369, in BackendRegistry.resolve_backend(self, backend)
        368 if gui is None:
    --> 369     raise RuntimeError(f"'{backend}' is not a recognised backend name")
        371 return backend, gui if gui != "headless" else None


    RuntimeError: 'widget' is not a recognised backend name

    
    During handling of the above exception, another exception occurred:


    RuntimeError                              Traceback (most recent call last)

    Cell In[1], line 6
          4 from w2_tools import plot_f, gradient_descent_one_variable, f_example_2, dfdx_example_2
          5 # Magic command to make matplotlib plots interactive.
    ----> 6 get_ipython().run_line_magic('matplotlib', 'widget')


    File ~/Projects/machine_learning_calculus_w2_lib/.venv/lib/python3.12/site-packages/IPython/core/interactiveshell.py:2504, in InteractiveShell.run_line_magic(self, magic_name, line, _stack_depth)
       2502     kwargs['local_ns'] = self.get_local_scope(stack_depth)
       2503 with self.builtin_trap:
    -> 2504     result = fn(*args, **kwargs)
       2506 # The code below prevents the output from being displayed
       2507 # when using magics with decorator @output_can_be_silenced
       2508 # when the last Python token in the expression is a ';'.
       2509 if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):


    File ~/Projects/machine_learning_calculus_w2_lib/.venv/lib/python3.12/site-packages/IPython/core/magics/pylab.py:103, in PylabMagics.matplotlib(self, line)
         98     print(
         99         "Available matplotlib backends: %s"
        100         % _list_matplotlib_backends_and_gui_loops()
        101     )
        102 else:
    --> 103     gui, backend = self.shell.enable_matplotlib(args.gui)
        104     self._show_matplotlib_backend(args.gui, backend)


    File ~/Projects/machine_learning_calculus_w2_lib/.venv/lib/python3.12/site-packages/IPython/core/interactiveshell.py:3787, in InteractiveShell.enable_matplotlib(self, gui)
       3784     import matplotlib_inline.backend_inline
       3786 from IPython.core import pylabtools as pt
    -> 3787 gui, backend = pt.find_gui_and_backend(gui, self.pylab_gui_select)
       3789 if gui != None:
       3790     # If we have our first gui selection, store it
       3791     if self.pylab_gui_select is None:


    File ~/Projects/machine_learning_calculus_w2_lib/.venv/lib/python3.12/site-packages/IPython/core/pylabtools.py:349, in find_gui_and_backend(gui, gui_select)
        347 else:
        348     gui = _convert_gui_to_matplotlib(gui)
    --> 349     backend, gui = backend_registry.resolve_gui_or_backend(gui)
        351 gui = _convert_gui_from_matplotlib(gui)
        352 return gui, backend


    File ~/Projects/machine_learning_calculus_w2_lib/.venv/lib/python3.12/site-packages/matplotlib/backends/registry.py:409, in BackendRegistry.resolve_gui_or_backend(self, gui_or_backend)
        407     return self.resolve_backend(gui_or_backend)
        408 except Exception:  # KeyError ?
    --> 409     raise RuntimeError(
        410         f"'{gui_or_backend}' is not a recognised GUI loop or backend name")


    RuntimeError: 'widget' is not a recognised GUI loop or backend name


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


```python
plot_f([0.001, 2.5], [-0.3, 13], f_example_1, 0.0)
```




    (<Figure size 800x400 with 1 Axes>, <Axes: xlabel='$x$', ylabel='$f$'>)




    
![png](C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_files/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_9_1.png)
    


Gradient descent can be implemented in the following function: 


```python
def gradient_descent(dfdx, x, learning_rate = 0.1, num_iterations = 100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
    return x
```

Note that there are three parameters in this implementation: `num_iterations`, `learning_rate`, initial point `x_initial`. Model parameters for such methods as gradient descent are usually found experimentially. For now, just assume that you know the parameters that will work in this model - you will see the discussion of that later. To optimize the function, set up the parameters and call the defined function `gradient_descent`:


```python
num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
print("Gradient descent result: x_min =", gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)) 
```

    Gradient descent result: x_min = 0.5671434156768685


The code in following cell will help you to visualize and understand the gradient descent method deeper. After the end of the animation, you can click on the plot to choose a new initial point and investigate how the gradient descent method will be performed.

You can see that it works successfully here, bringing it to the global minimum point!

What if some of the parameters will be changed? Will the method always work? Uncomment the lines in the cell below and rerun the code to investigate what happens if other parameter values are chosen. Try to investigate and analyse the results. You can read some comments below.

*Notes related to this animation*: 
- Gradient descent is performed with some pauses between the iterations for visualization purposes. The actual implementation is much faster.
- The animation stops when minimum point is reached with certain accuracy (it might be a smaller number of steps than `num_iterations`) - to avoid long runs of the code and for teaching purposes.
- Please wait for the end of the animation before making any code changes or rerunning the cell. In case of any issues, you can try to restart the Kernel and rerun the notebook.


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


    
![png](C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_files/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_15_0.png)
    


Comments related to the choice of the parameters in the animation above:

- Choosing `num_iterations = 25`, `learning_rate = 0.1`, `x_initial = 1.6` you get to the minimum point successfully. Even a little bit earlier - on the iteration 21, so for this choice of the learning rate and initial point, the number of iterations could have been taken less than `25` to save some computation time.

- Increasing the `learning_rate` to `0.3` you can see that the method converges even faster - you need less number of iterations. But note that the steps are larger and this may cause some problems.

- Increasing the `learning_rate` further to `0.5` the method doesn't converge anymore! You steped too far away from the minimum point. So, be careful - increasing `learning_rate` the method may converge significantly faster... or not converge at all.

- To be "safe", you may think, why not to decrease `learning_rate`?! Take it `0.04`, keeping the rest of the parameters the same. The model will not run enough number of iterations to converge!

- Increasing `num_iterations`, say to `75`, the model will converge but slowly. This would be more "expensive" computationally.

- What if you get back to the original parameters `num_iterations = 25`, `learning_rate = 0.1`, but choose some other `x_initial`, e.g. `0.05`? The function is steeper at that point, thus the gradient is larger in absolute value, and the first step is larger. But it will work - you will get to the minimum point.

- If you take `x_initial = 0.03` the function is even steeper, making the first step significantly larger. You are risking "missing" the minimum point.

- Taking `x_initial = 0.02` the method doesn't converge anymore...

This is a very simple example, but hopefully, it gives you an idea of how important is the choice of the initial parameters.

<a name='2'></a>
## 2 - Function with Multiple Minima

Now you can take a slightly more complicated example - a function in one variable, but with multiple minima. Such an example was shown in the videos, and you can plot the function with the following code:


```python
plot_f([0.001, 2], [-6.3, 5], f_example_2, -6)
```




    (<Figure size 800x400 with 1 Axes>, <Axes: xlabel='$x$', ylabel='$f$'>)




    
![png](C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_files/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_19_1.png)
    


Function `f_example_2` and its derivative `dfdx_example_2` are pre-defined and uploaded into this notebook. At this stage, while you are mastering the optimization method, do not worry about the corresponding expressions, just concentrate on the gradient descent and the related parameters for now.

Use the following code to run gradient descent with the same `learning_rate` and `num_iterations`, but with a different starting point:


```python
print("Gradient descent results")
print("Global minimum: x_min =", gradient_descent(dfdx_example_2, x=1.3, learning_rate=0.005, num_iterations=35)) 
print("Local minimum: x_min =", gradient_descent(dfdx_example_2, x=0.25, learning_rate=0.005, num_iterations=35)) 
```

    Gradient descent results
    Global minimum: x_min = 1.7751686214270586
    Local minimum: x_min = 0.7585728671820583


The results are different. Both times the point did fall into one of the minima, but in the first run it was a global minimum, while in the second run it got "stuck" in a local one. To see the visualization of what is happening, run the code below. You can uncomment the lines to try different sets of parameters or click on the plot to choose the initial point (after the end of the animation).


```python
num_iterations = 35; learning_rate = 0.005; x_initial = 1.3
# num_iterations = 35; learning_rate = 0.005; x_initial = 0.25
# num_iterations = 35; learning_rate = 0.01; x_initial = 1.3

gd_example_2 = gradient_descent_one_variable([0.001, 2], [-6.3, 5], f_example_2, dfdx_example_2, 
                                      gradient_descent, num_iterations, learning_rate, x_initial, -6, [0.1, -0.5])
```


    
![png](C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_files/C2_W2_Lab_1_Optimization_Using_Gradient_Descent_in_One_Variable_23_0.png)
    


You can see that gradient descent method is robust - it allows you to optimize a function with a small number of calculations, but it has some drawbacks. The efficiency of the method depends a lot on the choice of the initial parameters, and it is a challenge in machine learning applications to choose the "right" set of parameters to train the model!


```python

```
