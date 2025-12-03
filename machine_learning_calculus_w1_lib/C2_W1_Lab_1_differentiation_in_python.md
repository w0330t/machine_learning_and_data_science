---
aliases: ["Differentiation in Python: Symbolic, Numerical and Automatic"]
tags: []
created: 2025-11-18, 22:43:52
modified: 2025-11-21, 14:35:48
---

# Differentiation in Python: Symbolic, Numerical and Automatic

In this lab you explore which tools and libraries are available in Python to compute derivatives. You will perform symbolic differentiation with `SymPy` library, numerical with `NumPy` and automatic with `JAX` (based on `Autograd`). Comparing the speed of calculations, you will investigate the computational efficiency of those three methods.

%%
在这个实验中你可以探索有哪些在使用 Python 计算微分的工具和库。
你将会使用 `SymPy` 执行符号微分，使用 NumPy 进行数值微分，并使用 JAX（基于 Autograd）进行自动微分。
通过比较计算速度，你将考察这三种方法的计算效率。
%%
# Table of Contents
- [ 1 - Functions in Python](#1)
- [ 2 - Symbolic Differentiation](#2)
  - [ 2.1 - Introduction to Symbolic Computation with `SymPy`](#2.1)
  - [ 2.2 - Symbolic Differentiation with `SymPy`](#2.2)
  - [ 2.3 - Limitations of Symbolic Differentiation](#2.3)
- [ 3 - Numerical Differentiation](#3)
  - [ 3.1 - Numerical Differentiation with `NumPy`](#3.1)
  - [ 3.2 - Limitations of Numerical Differentiation](#3.2)
- [ 4 - Automatic Differentiation](#4)
  - [ 4.1 - Introduction to `JAX`](#4.1)
  - [ 4.2 - Automatic Differentiation with `JAX` ](#4.2)
- [ 5 - Computational Efficiency of Symbolic, Numerical and Automatic Differentiation](#5)

<a name='1'></a>
## 1 - Functions in Python

This is just a reminder how to define functions in Python. A simple function $f\left(x\right) = x^2$, it can be set up as:

%%
这里只是提醒你如何定义 Python 的函数。
一个简单的函数 $f(x)=x^2$，它可以这样定义。
%%

```python
def f(x):
    return x**2

print(f(3))
```

> [!result]
> 9


You can easily find the derivative of this function analytically. You can set it up as a separate function:

%%
你可以轻松地分析性地找到此函数的导数。然后将其设置为一个单独的函数：
%%

```python
def dfdx(x):
    return 2*x

print(dfdx(3))
```

> [!result]
6

Since you have been working with the `NumPy` arrays, you can apply the function to each element of an array:
%%
既然您一直在使用 `NumPy` 数组，您可以将该函数应用于数组的每个元素：
%%

```python
import numpy as np

x_array = np.array([1, 2, 3])

print("x: \n", x_array)
print("f(x) = x**2: \n", f(x_array))
print("f'(x) = 2x: \n", dfdx(x_array))
```

> [!result]
x: 
 [1 2 3]
f(x) = x**2: 
 [1 4 9]
f'(x) = 2x: 
 [2 4 6]



Now you can apply those functions `f` and `dfdx` to an array of a larger size. The following code will plot function and its derivative (you don't have to understand the details of the `plot_f1_and_f2` function at this stage):

%%
现在你可以将函数 `f` 和 `dfdx` 应用在更大尺寸的数组上了。
下面的代码可以绘制函数和它的导数。
%%


```python
import matplotlib.pyplot as plt

# Output of plotting commands is displayed inline within the Jupyter notebook.
%matplotlib inline

def plot_f1_and_f2(f1, f2=None, x_min=-5, x_max=5, label1="f(x)", label2="f'(x)"):
    x = np.linspace(x_min, x_max,100)

    # Setting the axes at the centre.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, f1(x), 'r', label=label1)
    if not f2 is None:
        # If f2 is an array, it is passed as it is to be plotted as unlinked points.
        # If f2 is a function, f2(x) needs to be passed to plot it.        
        if isinstance(f2, np.ndarray):
            plt.plot(x, f2, 'bo', markersize=3, label=label2,)
        else:
            plt.plot(x, f2(x), 'b', label=label2)
    plt.legend()

    plt.show()
    
plot_f1_and_f2(f, dfdx)
```

> [!result]
![C2_W1_Lab_1_differentiation_in_python_11_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W1_Lab_1_differentiation_in_python_11_0.png)


In real life the functions are more complicated and it is not possible to calculate the derivatives analytically every time. Let's explore which tools and libraries are available in Python for the computation of derivatives without manual derivation.

%%
在现实生活中，函数往往会更加复杂，不可能每次都通过解析求导来计算。
让我们使用 Python 的工具和库计算导数而非手动求导。
%%
<a name='2'></a>
## 2 - Symbolic Differentiation

**Symbolic computation** deals with the computation of mathematical objects that are represented exactly, not approximately (e.g. $\sqrt{2}$ will be written as it is, not as $1.41421356237$). For differentiation it would mean that the output will be somehow similar to if you were computing derivatives by hand using rules (analytically). Thus, symbolic differentiation can produce exact derivatives.

%%
**符号计算**用于处理精确的数学计算，它不做近似处理（比如 $\sqrt{2}$ 直接写作 $\sqrt{2}$，而不是 $1.41421356237$）。
对于微分而言它意味着输出结果会类似于手工计算的规则。
因此，符号计算可以精确的进行求导。
%%
<a name='2.1'></a>
### 2.1 - Introduction to Symbolic Computation with `SymPy`

Let's explore symbolic differentiation in Python with commonly used `SymPy` library.

%%
让我们使用常用的 `SymPy` 进行符号微分
%%

If you want to compute the approximate decimal value of $\sqrt{18}$, you could normally do it in the following way:

%%
如果你想计算 $\sqrt{18}$ 的近似值，一般来说你会像下面这么写：
%%

```python
import math

math.sqrt(18)
```

> [!result]
4.242640687119285

The output $4.242640687119285$ is an approximate result. You may recall that $\sqrt{18} = \sqrt{9 \cdot 2} = 3\sqrt{2}$ and see that it is pretty much impossible to deduct it from the approximate result. But with the symbolic computation systems the roots are not approximated with a decimal number but rather only simplified, so the output is exact:

%%
这个 $4.242640687119285$ 是一个近似的结果。
你可能还记得，$\sqrt{18} = \sqrt{9 \cdot 2} = 3\sqrt{2}$，并且看到这个精确值几乎不可能从近似值推导出来。
但使用符号计算系统的时候，根不会被近似为小数，而是只进行简化，因此输出是精确的：
%%


```python
# This format of module import allows to use the sympy functions without sympy. prefix.
from sympy import *

# This is actually sympy.sqrt function, but sympy. prefix is omitted.
sqrt(18)
```

> [!result]
$\displaystyle 3\sqrt{2}$

Numerical evaluation of the result is available, and you can set number of the digits to show in the approximated output:
%%
提供结果的数值评估，您可以设置近似输出中显示的小数位数：
%%

```python
N(sqrt(18),8)
```

> [!result]
$\displaystyle 4.2426407$

In `SymPy` variables are defined using **symbols**. In this particular library they need to be predefined (a list of them should be provided). Have a look in the cell below, how the symbolic expression, correspoinding to the mathematical expression $2x^2 - xy$, is defined:

%%
使用 **symbols** 定义 `SymPy` 的变量。
使用这个库的话，需要预先定义（定义一个 list）。
请看下方单元格中，如何定义对应于数学表达式 $2x^2 - xy$ 的符号表达式：
%%


```python
# List of symbols.
x, y = symbols('x y')
# Definition of the expression.
expr = 2 * x**2 - x * y
expr
```

> [!result]
$\displaystyle 2 x^{2} - x y$

Now you can perform various manipulations with this expression: add or subtract some terms, multiply by other expressions etc., just like if you were doing it by hands:

%%
现在你可以对这个表达式进行各种操作：添加或者减去一些项，乘以其他的表达式等等，如同自己计算一般。
%%


```python
expr_manip = x * (expr + x * y + x**3)
expr_manip
```

> [!result]
$\displaystyle x \left(x^{3} + 2 x^{2}\right)$


You can also expand the expression:
%%
你同样可以展开这个表达式
%%

```python
expand(expr_manip)
```

> [!result]
$\displaystyle x^{4} + 2 x^{3}$

Or factorise it:
%%
或者对其做因式分解：
%%

```python
factor(expr_manip)
```

> [!result]
$\displaystyle x^{3} \left(x + 2\right)$

To substitute particular values for the variables in the expression, you can use the following code:

%%
想要将数值代入表达式，可以用下面的代码
%%

```python
expr.evalf(subs={x:-1, y:2})
```

> [!result]
$\displaystyle 4.0$


This can be used to evaluate a function $f\left(x\right) = x^2$:

%%
这同样可以用于对函数 $f\left(x\right) = x^2$ 进行计算：
%%

```python
f_symb = x ** 2
f_symb.evalf(subs={x:3})
```

> [!result]
$\displaystyle 9.0$

You might be wondering now, is it possible to evaluate the symbolic functions for each element of the array? At the beginning of the lab you have defined a `NumPy` array `x_array`:

%%
你现在可能想知道，能否对数组的每个元素进行符号函数求值？
在本实验开始的时候，你已经定义了一个 NumPy 数组 `x_array`：
%%

```python
print(x_array)
```

> [!result]
 [1 2 3]


Now try to evaluate function `f_symb` for each element of the array. You will get an error:

%%
现在尝试通过函数 `f_symb` 计算这个数组的每个元素。
你将得到一个错误提示
%%

```python
try:
    f_symb(x_array)
except TypeError as err:
    print(err)
```

> [!result] 
'Pow' object is not callable

It is possible to evaluate the symbolic functions for each element of the array, but you need to make a function `NumPy` -friendly first:

%%
它其实是可以通过符号函数求解这个数组的每个元素，但是你需要先让函数兼容 `NumPy`。
%%

```python
from sympy.utilities.lambdify import lambdify

f_symb_numpy = lambdify(x, f_symb, 'numpy')
```

The following code should work now:
%%
下面的代码现在就能正常运行了。
%%


```python
print("x: \n", x_array)
print("f(x) = x**2: \n", f_symb_numpy(x_array))
```

> [!result]
x: 
 [1 2 3]
f(x) = x**2: 
 [1 4 9]


`SymPy` has lots of great functions to manipulate expressions and perform various operations from calculus. More information about them can be found in the official documentation [here](https://docs.sympy.org/).
%%
`SymPy` 提供了很多强大的函数来操作表达式和进行各种微积分运算。
更多信息可以去参阅官方文档[这里](https://docs.sympy.org/)。 
%%

<a name='2.2'></a>
### 2.2 - Symbolic Differentiation with `SymPy`

Let's try to find a derivative of a simple power function using `SymPy`:
%%
我们来尝试用 `SymPy` 求解一个简单幂函数的导数：
%%

```python
diff(x**3,x)
```

> [!result]
$\displaystyle 3 x^{2}$


Some standard functions can be used in the expression, and `SymPy` will apply required rules (sum, product, chain) to calculate the derivative:
%%
一些标准的函数可以直接写入表达式，`SymPy` 将应用规则来计算导数。
%%

```python
dfdx_composed = diff(exp(-2*x) + 3*sin(3*x), x)
dfdx_composed
```

> [!result]
$\displaystyle 9 \cos{\left(3 x \right)} - 2 e^{- 2 x}$


Now calculate the derivative of the function `f_symb` defined in [2.1](#2.1) and make it `NumPy`-friendly:
%%
现在定义 2.1 的兼容 `NumPy` 的函数 `f_symb` 来计算导数。
%%

```python
dfdx_symb = diff(f_symb, x)
dfdx_symb_numpy = lambdify(x, dfdx_symb, 'numpy')
```

Evaluate function `dfdx_symb_numpy` for each element of the `x_array`:
%%
使用函数 `dfdx_symb_numpy` 求得 `x_array` 的每个元素：
%%

```python
print("x: \n", x_array)
print("f'(x) = 2x: \n", dfdx_symb_numpy(x_array))
```

> [!result]
x: 
 [1 2 3]
f'(x) = 2x: 
 [2 4 6]

You can apply symbolically defined functions to the arrays of larger size. The following code will plot function and its derivative, you can see that it works:
%%
你可以象征性的定义一些函数来处理大尺寸的数组，下面的代码会绘制函数和它的导数，你可以看看它是怎么工作的。
%%

```python
plot_f1_and_f2(f_symb_numpy, dfdx_symb_numpy)
```

> [!result]
![C2_W1_Lab_1_differentiation_in_python_11_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W1_Lab_1_differentiation_in_python_11_0.png)

<a name='2.3'></a>
### 2.3 - Limitations of Symbolic Differentiation

Symbolic Differentiation seems to be a great tool. But it also has some limitations. Sometimes the output expressions are too complicated and even not possible to evaluate. For example, find the derivative of the function
%%
符号微分看上去是一个强大的工具。
但是它同样有一些局限性。
有时候，输出表达式过于复杂，甚至无法求值。
比如下面这个函数的求导。
%%
$$\left|x\right| = \begin{cases} x, \ \text{if}\ x > 0\\  -x, \ \text{if}\ x < 0 \\ 0, \ \text{if}\ x = 0\end{cases}$$
Analytically, its derivative is:
%%
从解析的角度来看，它的导数是
%%
$$\frac{d}{dx}\left(\left|x\right|\right) = \begin{cases} 1, \ \text{if}\ x > 0\\  -1, \ \text{if}\ x < 0\\\ \text{does not exist}, \ \text{if}\ x = 0\end{cases}$$

Have a look the output from the symbolic differentiation:
%%
看看符号微分的结果：
%%

```python
dfdx_abs = diff(abs(x),x)
dfdx_abs
```

> [!result]
$\displaystyle \frac{\left(\operatorname{re}{\left(x\right)} \frac{d}{d x} \operatorname{re}{\left(x\right)} + \operatorname{im}{\left(x\right)} \frac{d}{d x} \operatorname{im}{\left(x\right)}\right) \operatorname{sign}{\left(x \right)}}{x}$

Looks complicated, but it would not be a problem if it was possible to evaluate. But check, that for $x=-2$ instead of the derivative value $-1$ it outputs some unevaluated expression:
%%
看上去非常复杂，但是如果它可以求值那么它就不是一个问题。
但是请注意，如果使用 x=-2 时，结果不是-1 而是一些未求值的表达式。
%%

```python
dfdx_abs.evalf(subs={x:-2})
```

> [!result]
$\displaystyle - \left. \frac{d}{d x} \operatorname{re}{\left(x\right)} \right|_{\substack{ x=-2 }}$


And in the `NumPy` friendly version it also will give an error:
%%
在 `NumPy` 兼容的版本下它同样会报错。
%%

```python
dfdx_abs_numpy = lambdify(x, dfdx_abs,'numpy')

try:
    dfdx_abs_numpy(np.array([1, -2, 0]))
except NameError as err:
    print(err)
```

> [!fail] Result
PrintMethodNotImplementedError            Traceback (most recent call last)
（略过）
PrintMethodNotImplementedError: Unsupported by <class 'sympy.printing.numpy.NumPyPrinter'>: <class 'sympy.core.function.Derivative'>
Printer has no method: _print_Derivative_re
Set the printer option 'strict' to False in order to generate partially printed code.

In fact, there are problems with the evaluation of the symbolic expressions wherever there is a "jump" in the derivative (e.g. function expressions are different for different intervals of $x$), like it happens with $\frac{d}{dx}\left(\left|x\right|\right)$. 
%%
实际上，但凡时导数存在“跳跃”的地方，符号表达式的求值都会出现问题。（函数表达式随 $x$ 的不同区间而异）。
就像 $\frac{d}{dx}\left(\left|x\right|\right)$ 那样。
%%

Also, you can see in this example, that you can get a very complicated function as an output of symbolic computation. This is called **expression swell**, which results in unefficiently slow computations. You will see the example of that below after learning other differentiation libraries in Python.
%%
同样的，你可以看到在这个例子中，你可以得到非常复杂的函数作为符号计算的输出结果。
这个被称为**表达式膨胀**，会导致计算效率低下且速度缓慢。
你将在学习完 Python 中其他求导库后，在下方看到相关示例。
%%
<a name='3'></a>
## 3 - Numerical Differentiation

This method does not take into account the function expression. The only important thing is that the function can be evaluated in the nearby points $x$ and $x+\Delta x$, where $\Delta x$ is sufficiently small. Then $\frac{df}{dx}\approx\frac{f\left(x + \Delta x\right) - f\left(x\right)}{\Delta x}$, which can be called a **numerical approximation** of the derivative. 
%%
这个方法没有考虑函数表达式。
这里唯一重要的事情是这个函数可以求出邻近点 $x$ and $x+\Delta x$, 其中 $\Delta x$ 足够小。
使得 $\frac{df}{dx}\approx\frac{f\left(x + \Delta x\right) - f\left(x\right)}{\Delta x}$，这被称为导数的**数值近似**。
%%

Based on that idea there are different approaches for the numerical approximations, which somehow vary in the computation speed and accuracy. However, for all of the methods the results are not accurate - there is a round off error. At this stage there is no need to go into details of various methods, it is enough to investigate one of the numerial differentiation functions, available in `NumPy` package.
%%
基于这一思想，数值逼近有多种不同的实现途径，这些方法在计算速度和准确性上各有不同。
无论如何，所有的方法得到的结果都是不精准的，它们都存在舍入误差。
现阶段无需深入探讨各种方法的细节，使用 `NumPy` 包已经可以足够让我们探索一个数值的微分函数了。
%%
<a name='3.1'></a>
### 3.1 - Numerical Differentiation with `NumPy`

You can call function `np.gradient` to find the derivative of function $f\left(x\right) = x^2$ defined above. The first argument is an array of function values, the second defines the spacing $\Delta x$ for the evaluation. Here pass it as an array of $x$ values, the differences will be calculated automatically. You can find the documentation [here](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).
%%
你可以使用函数 `np.gradient` 来求得上面这个 $f\left(x\right) = x^2$ 函数的导数。
第一个参数是一个函数值的数组，第二个定义了用于评估间距的 $\Delta x$。
这里将它作为 x 值的数组传递，它们的差异将自动计算。你可以参考文档 [这里](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html)。
%%

```python
x_array_2 = np.linspace(-5, 5, 100)
dfdx_numerical = np.gradient(f(x_array_2), x_array_2)

plot_f1_and_f2(dfdx_symb_numpy, dfdx_numerical, label1="f'(x) exact", label2="f'(x) approximate")
```

> [!result]
![C2_W1_Lab_1_differentiation_in_python_62_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W1_Lab_1_differentiation_in_python_62_0.png)

对上面的代码总结一下
- `x_array_2` 是一个从 $-5$ 到 $5$ 的平均分布的数组，作为**函数**的 **$x$ 轴**。
- `f(x_array_2)` 是上面定义的 $f(x)=x^2$，作为**函数**的 **$y$ 轴**。
- `np.gradient` 的作用是将函数求导，**结果为上面这根直线的 $y$ 轴**，`x_array_2` 作为对应这根直线的 $x$ 轴，

Try to do numerical differentiation for more complicated function:
%%
尝试对更复杂的函数做数值微分。
%%

```python
def f_composed(x):
    return np.exp(-2*x) + 3*np.sin(3*x)

plot_f1_and_f2(lambdify(x, dfdx_composed, 'numpy'), np.gradient(f_composed(x_array_2), x_array_2),
              label1="f'(x) exact", label2="f'(x) approximate")
```

> [!result]
> ![C2_W1_Lab_1_differentiation_in_python_64_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W1_Lab_1_differentiation_in_python_64_0.png)


The results are pretty impressive, keeping in mind that it does not matter at all how the function was calculated - only the final values of it!
%%
这些结果让人印象深刻，请记住，函数是如何计算的完全无关紧要——重要的是它的最终数值！
%%
<a name='3.2'></a>
### 3.2 - Limitations of Numerical Differentiation

Obviously, the first downside of the numerical differentiation is that it is not exact. However, the accuracy of it is normally enough for machine learning applications. At this stage there is no need to evaluate errors of the numerical differentiation.
%%
显而易见的，第一个缺点是数值微分是不精确的。
无论如何，这个准确性应付一般的机器学习应用已经足够了。
在这个阶段不需要评估数值微分的误差。
%%

Another problem is similar to the one which appeared in the symbolic differentiation: it is inaccurate at the points where there are "jumps" of the derivative. Let's compare the exact derivative of the absolute value function and with numerical approximation:
%%
另一个问题与符号微分中出现的问题类似：在导数存在“跳跃”的点时，它的准确性较差。
让我们比较绝对值函数的精确导数与数值逼近的结果。
%%


```python
def dfdx_abs(x):
    if x > 0:
        return 1
    else:
        if x < 0:
            return -1
        else:
            return None

plot_f1_and_f2(np.vectorize(dfdx_abs), np.gradient(abs(x_array_2), x_array_2))
```

> [!result]
![C2_W1_Lab_1_differentiation_in_python_68_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W1_Lab_1_differentiation_in_python_68_0.png)

You can see that the results near the "jump" are $0.5$ and $-0.5$, while they should be $1$ and $-1$. These cases can give significant errors in the computations.
%%
你可以看到这个“跳跃”的结果是 $0.5$ 和 $-0.5$，它们应该是 $1$ 和 $-1$。
这些情况可能在计算中产生显著误差。
%%

But the biggest problem with the numerical differentiation is slow speed. It requires function evalutation every time.  In machine learning models there are hundreds of parameters and there are hundreds of derivatives to be calculated, performing full function evaluation every time slows down the computation process. You will see the example of it below.
%%
但是更大的问题在于数值微分太慢了。
它每次都需要函数求值。
在机器学习模型中有数百个参数和数百个求导需要计算，每次执行完整的求值都会减慢计算过程。
你可以在下方看到相应的示例。
%%
<a name='4'></a>
## 4 - Automatic Differentiation

**Automatic differentiation** (autodiff) method breaks down the function into common functions ($sin$, $cos$, $log$, power functions, etc.), and constructs the computational graph consisting of the basic functions. Then the chain rule is used to compute the derivative at any node of the graph. It is the most commonly used approach in machine learning applications and neural networks, as the computational graph for the function and its derivatives can be built during the construction of the neural network, saving in future computations.
%%
**自动微分**法可以将函数分解为常见的函数（如正弦、余弦、对数、幂函数等），然后将基本函数构成计算图。
然后运用链式法则即可求得图中任意节点的导数。
这是机器学习应用和神经网络中最常用的方法，因为在构建神经网络时，可以同时建立函数及其导数的计算图，从而节省后续计算成本。
%%

The main disadvantage of it is implementational difficulty. However, nowadays there are libraries that are convenient to use, such as [MyGrad](https://mygrad.readthedocs.io/en/latest/index.html), [Autograd](https://autograd.readthedocs.io/en/latest/) and [JAX](https://jax.readthedocs.io/en/latest/). `Autograd` and `JAX` are the most commonly used in the frameworks to build neural networks. `JAX` brings together `Autograd` functionality for optimization problems, and `XLA` (Accelerated Linear Algebra) compiler for parallel computing.
%%
其主要缺点是实施难度大。
如今有一些方便易用的库，比如 [MyGrad](https://mygrad.readthedocs.io/en/latest/index.html), [Autograd](https://autograd.readthedocs.io/en/latest/) 和 [JAX](https://jax.readthedocs.io/en/latest/)。
`Autograd` 和 `JAX` 是构建神经网络最常用的框架。
`JAX` 集成了 `Autograd` 的优化问题解决能力，以及用于并行计算的 `XLA`（加速线性代数）编译器。
%%

The syntax of `Autograd` and `JAX` are slightly different. It would be overwhelming to cover both at this stage. In this notebook you will be performing automatic differentiation using one of them: `JAX`.
%%
`Autograd` 和 `JAX` 的语法有些不同。
现阶段要同时顾及两者会让人应接不暇。
在这个 notebook 中你将要使用 `JAX` 执行自动微分。
%%
<a name='4.1'></a>
### 4.1 - Introduction to `JAX`

To begin with, load the required libraries. From `jax` package you need to load just a couple of functions for now (`grad` and `vmap`). Package `jax.numpy` is a wrapped `NumPy`, which pretty much replaces `NumPy` when `JAX` is used. It can be loaded as `np` as if it was an original `NumPy` in most of the cases. However, in this notebook you'll upload it as `jnp` to distinguish them for now.
%%
最初需要载入库。
目前我们只需从 jax 软件包中导入两个函数即可：grad 和 vmap。
`jax.numpy` 已经包含了 `NumPy`，使用 `JAX` 完全可以替代 `NmuPy`。
大多数情况下，它可以像原版 NumPy 一样通过 `np` 加载。
无论如何，在这个 notebook 中你将暂时将其定义为 `jnp` 以作区分。
%%


```python
from jax import grad, vmap
import jax.numpy as jnp
```

Create a new `jnp` array and check its type.
%%
创建新的 `jnp` 数组然后确认变量类型。
%%


```python
x_array_jnp = jnp.array([1., 2., 3.])

print("Type of NumPy array:", type(x_array))
print("Type of JAX NumPy array:", type(x_array_jnp))
# Please ignore the warning message if it appears.
```

> [!result]
Type of NumPy array: <class 'numpy.ndarray'>
Type of JAX NumPy array: <class 'jaxlib._jax.ArrayImpl'>


The same array can be created just converting previously defined `x_array = np.array([1, 2, 3])`, although in some cases `JAX` does not operate with integers, thus the values need to be converted to floats. You will see an example of it below.
%%
相同的数组可以通过之前定义的 `x_array = np.array([1, 2, 3])` 变量直接转换来进行创建，尽管一些情况下， `JAX` 不支持整数运算，因此这些值需要转为浮点型。
下面是个例子：
%%


```python
x_array_jnp = jnp.array(x_array.astype('float32'))
print("JAX NumPy array:", x_array_jnp)
print("Type of JAX NumPy array:", type(x_array_jnp))
```

> [!result]
JAX NumPy array: [1. 2. 3.]
Type of JAX NumPy array: <class 'jaxlib._jax.ArrayImpl'>

Note, that `jnp` array has a specific type `jaxlib.xla_extension.DeviceArray`. In most of the cases the same operators and functions are applicable to them as in the original `NumPy`, for example:
%%
那个 `jnp` 数组有一个特定的类型 `jaxlib.xla_extension.DeviceArray`。
在大多数时候它们的运算符和函数都和原生的 `NumPy` 相同，比如：
%%

```python
print(x_array_jnp * 2)
print(x_array_jnp[2])
```
> [!result]
[2. 4. 6.]
3.0

But sometimes working with `jnp` arrays the approach needs to be changed. In the following code, trying to assign a new value to one of the elements, you will get an error:
%%
但有时，使用 `jnp` 数组时需要改变方法.
在下面的代码中尝试给数组分配新的值，你将获得一个错误提示。
%%


```python
try:
    x_array_jnp[2] = 4.0
except TypeError as err:
    print(err)
```

> [!result]
JAX arrays are immutable and do not support in-place item assignment. Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html


To assign a new value to an element in the `jnp` array you need to apply functions `.at[i]`, stating which element to update, and `.set(value)` to set a new value. These functions also operate **out-of-place**, the updated array is returned as a new array and the original array is not modified by the update.
%%
为 `jnp` 数组分配新的值需要使用函数 `.at[i]` 指明要更新哪些元素，然后使用 `.set(value)` 设置新的值。
这些函数同样以**非就地**方式操作，更新后的数组作为新数组返回，而原始数组在更新过程中不会被修改。
%%

```python
y_array_jnp = x_array_jnp.at[2].set(4.0)
print(y_array_jnp)
```

> [!result]
[1. 2. 4.]


Although, some of the `JAX` functions will work with arrays defined with `np` and `jnp`. In the following code you will get the same result in both lines:
%%
尽管如此，有些 `JAX` 函数可以同时处理使用 `np` 和 `jnp` 定义的数组。
比如下面的代码都会输出相同的结果。
%%

```python
print(jnp.log(x_array))
print(jnp.log(x_array_jnp))
```

> [!result]
[0.        0.6931472 1.0986123]
[0.        0.6931472 1.0986123]

This is probably confusing - which `NumPy` to use then? Usually when `JAX` is used, only `jax.numpy` gets imported as `np`, and used instead of the original one.
%%
这可能会让人有一些困惑——究竟用哪个 `NumPy` ？
一般来说如果使用 `JAX`，只需要导入 `jax.numpy` 即可，用它来替代原生的 `NumPy`。
%%

 <a name='4.2'></a>
### 4.2 - Automatic Differentiation with `JAX` 

Time to do automatic differentiation with `JAX`. The following code will calculate the derivative of the previously defined function $f\left(x\right) = x^2$ at the point $x = 3$:
%%
是时候使用 `JAX` 进行自动微分了。
下面的代码将计算之间定义的函数 $f\left(x\right) = x^2$ 在 $x = 3$ 时的导数：
%%

```python
print("Function value at x = 3:", f(3.0))
print("Derivative value at x = 3:",grad(f)(3.0))
```

> [!result]
Function value at x = 3: 9.0
Derivative value at x = 3: 6.0


Very easy, right? Keep in mind, please, that this cannot be done using integers. The following code will output an error:
%%
非常简单对吗？
保持专注，这里不能使用整数。
下面的代码输出了一个错误。
%%

```python
try:
    grad(f)(3)
except TypeError as err:
    print(err)
```

> [!result]
grad requires real- or complex-valued inputs (input dtype that is a sub-dtype of np.inexact), but got int32. If you want to use Boolean- or integer-valued inputs, use vjp or set allow_int to True.


Try to apply the `grad` function to an array, calculating the derivative for each of its elements: 
%%
尝试将数组传入 `grad` 函数，用于计算每个元素的导数。
%%

```python
try:
    grad(f)(x_array_jnp)
except TypeError as err:
    print(err)
```

> [!result]
Gradient only defined for scalar-output functions. Output had shape: (3,).

There is some broadcasting issue there. You don't need to get into more details of this at this stage, function `vmap` can be used here to solve the problem.
%%
这里有一些广播的问题，在现阶段你并不需要了解细节。
函数 `vmap` 可以解决这里的问题。
%%

*Note*: Broadcasting is covered in the Course 1 of this Specialization "Linear Algebra". You can also review it in the documentation [here](https://numpy.org/doc/stable/user/basics.broadcasting.html#:~:text=The%20term%20broadcasting%20describes%20how,that%20they%20have%20compatible%20shapes.).
%%
注：广播（broadcasting）在专项课程的第一门课《线性代数》中有所涵盖。
你也可以查看 `NumPy` 的文档 [这里](https://numpy.org/doc/stable/user/basics.broadcasting.html#:~:text=The%20term%20broadcasting%20describes%20how,that%20they%20have%20compatible%20shapes.).
%%

```python
dfdx_jax_vmap = vmap(grad(f))(x_array_jnp)
print(dfdx_jax_vmap)
```

> [!result]
[2. 4. 6.]

Great, now `vmap(grad(f))` can be used to calculate the derivative of function `f` for arrays of larger size and you can plot the output:
%%
很好，现在 `vmap(grad(f))` 可以用于计算更大尺寸数组的函数 `f` 导数，并且你可以绘制输出：
%%


```python
plot_f1_and_f2(f, vmap(grad(f)))
```

> [!result]
![C2_W1_Lab_1_differentiation_in_python_11_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W1_Lab_1_differentiation_in_python_11_0.png)


In the following code you can comment/uncomment lines to visualize the common derivatives. All of them are found using `JAX` automatic differentiation. The results look pretty good!
%%
在以下代码中，您可以注释或取消注释行以查看常见导数的可视化效果。这些导数均通过 JAX 自动微分技术求得，结果看起来相当不错！
%%


```python
def g(x):
#     return x**3
#     return 2*x**3 - 3*x**2 + 5
#     return 1/x
#     return jnp.exp(x)
#     return jnp.log(x)
#     return jnp.sin(x)
#     return jnp.cos(x)
    return jnp.abs(x)
#     return jnp.abs(x)+jnp.sin(x)*jnp.cos(x)

plot_f1_and_f2(g, vmap(grad(g)))
```

> [!result]
> ![C2_W1_Lab_1_differentiation_in_python_101_0.png](https://obsidian-image.wwtt.xyz/2025/11/C2_W1_Lab_1_differentiation_in_python_101_0.png)




<a name='5'></a>
## 5 - Computational Efficiency of Symbolic, Numerical and Automatic Differentiation

In sections [2.3](#2.3) and [3.2](#3.2) low computational efficiency of symbolic and numerical differentiation was discussed. Now it is time to compare speed of calculations for each of three approaches. Try to find the derivative of the same simple function $f\left(x\right) = x^2$ multiple times, evaluating it for an array of a larger size, compare the results and time used:
%%
在 2.3 节和 3.2 节中，我们讨论了符号微分和数值微分计算效率低下的问题。
现在是时候开始比较三种方法的计算速度了。
尝试多次求导同一个简单函数 f(x) = x²，将其在一个更大尺寸的数组上进行求值，并比较结果和所用时间：
%%


```python
import timeit, time

x_array_large = np.linspace(-5, 5, 1000000)

tic_symb = time.time()
res_symb = lambdify(x, diff(f(x),x),'numpy')(x_array_large)
toc_symb = time.time()
time_symb = 1000 * (toc_symb - tic_symb)  # Time in ms.

tic_numerical = time.time()
res_numerical = np.gradient(f(x_array_large),x_array_large)
toc_numerical = time.time()
time_numerical = 1000 * (toc_numerical - tic_numerical)

tic_jax = time.time()
res_jax = vmap(grad(f))(jnp.array(x_array_large.astype('float32')))
toc_jax = time.time()
time_jax = 1000 * (toc_jax - tic_jax)

print(f"Results\nSymbolic Differentiation:\n{res_symb}\n" + 
      f"Numerical Differentiation:\n{res_numerical}\n" + 
      f"Automatic Differentiation:\n{res_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_symb} ms\n" + 
      f"Numerical Differentiation:\n{time_numerical} ms\n" + 
      f"Automatic Differentiation:\n{time_jax} ms")
```

> [!result]
Results
Symbolic Differentiation:
[-10.       -9.99998  -9.99996 ...   9.99996   9.99998  10.     ]
Numerical Differentiation:
[-9.99999 -9.99998 -9.99996 ...  9.99996  9.99998  9.99999]
Automatic Differentiation:
[-10.       -9.99998  -9.99996 ...   9.99996   9.99998  10.     ]
> 
Time
Symbolic Differentiation:
2.6586055755615234 ms
Numerical Differentiation:
41.49174690246582 ms
Automatic Differentiation:
6.301164627075195 ms


The results are pretty much the same, but the time used is different. Numerical approach is obviously inefficient when differentiation needs to be performed many times, which happens a lot training machine learning models. Symbolic and automatic approach seem to be performing similarly for this simple example. But if the function becomes a little bit more complicated, symbolic computation will experiance significant expression swell and the calculations will slow down.
%%
结果看上去几乎是相同的，但是使用时间的区别有点大了。
当需要多次执行微分运算时，数值方法显然效率低下，在机器学习模型中经常发生这样的事情。
符号方法和自动方法在这个示例中看上去表现相似。
但是如果函数变得更加复杂一点的话，符号计算将面临**表达式显著膨胀**的问题，从而导致计算速度变慢。
%%

*Note*: Sometimes the execution time results may vary slightly, especially for automatic differentiation. You can run the code above a few time to see different outputs. That does not influence the conclusion that numerical differentiation is slower. `timeit` module can be used more efficiently to evaluate execution time of the codes, but that would unnecessary overcomplicate the codes here.
%%
*注意*：有时候运行时间的结果会有所不同，特别是自动微分。
你可以多运行几次上面的代码看看不同的输出。
这并不会影响数值微分较慢的结论。
`timeit` 模块能更高效地评估代码的执行时间，但那会不必要地过度复杂化这里的代码。
%%

Try to define some polynomial function, which should not be that hard to differentiate, and compare the computation time for its differentiation symbolically and automatically:
%%
试着定义一个多项式函数，其求导过程不应过于复杂。接着，比较其符号求导与自动求导各自所需的计算时间：
%%

```python
def f_polynomial_simple(x):
    return 2*x**3 - 3*x**2 + 5

def f_polynomial(x):
    for i in range(3):
        x = f_polynomial_simple(x)
    return x

tic_polynomial_symb = time.time()
res_polynomial_symb = lambdify(x, diff(f_polynomial(x),x),'numpy')(x_array_large)
toc_polynomial_symb = time.time()
time_polynomial_symb = 1000 * (toc_polynomial_symb - tic_polynomial_symb)

tic_polynomial_jax = time.time()
res_polynomial_jax = vmap(grad(f_polynomial))(jnp.array(x_array_large.astype('float32')))
toc_polynomial_jax = time.time()
time_polynomial_jax = 1000 * (toc_polynomial_jax - tic_polynomial_jax)

print(f"Results\nSymbolic Differentiation:\n{res_polynomial_symb}\n" + 
      f"Automatic Differentiation:\n{res_polynomial_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_polynomial_symb} ms\n" +  
      f"Automatic Differentiation:\n{time_polynomial_jax} ms")
```

> [!result]
Results
Symbolic Differentiation:
[2.88570423e+24 2.88556400e+24 2.88542377e+24 ... 1.86202587e+22
 1.86213384e+22 1.86224181e+22]
Automatic Differentiation:
[2.8857043e+24 2.8855642e+24 2.8854241e+24 ... 1.8620253e+22 1.8621349e+22
 1.8622416e+22]
> 
Time
Symbolic Differentiation:
282.2086811065674 ms
Automatic Differentiation:
35.06922721862793 ms


Again, the results are similar, but automatic differentiation is times faster. 
%%
再一次，结果差不多，但是自动微分要快得多。
%%

With the increase of function computation graph, the efficiency of automatic differentiation compared to other methods raises, because autodiff method uses chain rule!
%%
随着函数计算图的增大，自动微分相较于其他方法的效率优势愈发明显，这得益于自动微分运用了链式法则！
%%

Congratulations! Now you are equiped with Python tools to perform differentiation.
%%
恭喜！现在您已掌握使用 Python 工具执行微分的方法。
%%
