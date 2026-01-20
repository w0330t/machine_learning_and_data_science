---
aliases: [Classification with Perceptron]
tags: []
created: 2025-12-05, 10:02:32
modified: 2025-12-05, 18:40:06
---

# Classification with Perceptron

In this lab, you will use a single perceptron neural network model to solve a simple classification problem. 
%%
在这个实验中，你将使用简单的感知神经网络模型解决简单的分类问题。
%%
# Table of Contents

- [ 1 - Simple Classification Problem](#1)
- [ 2 - Single Perceptron Neural Network with Activation Function](#2)
  - [ 2.1 - Neural Network Structure](#2.1)
  - [ 2.2 - Dataset](#2.2)
  - [ 2.3 - Define Activation Function](#2.3)
- [ 3 - Implementation of the Neural Network Model](#3)
  - [ 3.1 - Defining the Neural Network Structure](#3.1)
  - [ 3.2 - Initialize the Model's Parameters](#3.2)
  - [ 3.3 - The Loop](#3.3)
  - [ 3.4 - Integrate parts 3.1, 3.2 and 3.3 in nn_model() and make predictions](#3.4)
- [ 4 - Performance on a Larger Dataset](#4)

## Packages

Let's first import all the packages that you will need during this lab.


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# A function to create a dataset.
from sklearn.datasets import make_blobs 

# Output of plotting commands is displayed inline within the Jupyter notebook.
%matplotlib inline 

# Set a seed so that the results are consistent.
np.random.seed(3)
```

<a name='1'></a>
## 1 - Simple Classification Problem

**Classification** is the problem of identifying which of a set of categories an observation belongs to. In case of only two categories it is called a **binary classification problem**. Let's see a simple example of it.
%%
分类是指将观察对象归入一组类别中的某一类的问题。
案例中只有两个类型的被称为**二元分类问题**。
让我们看看这个简单的例子。
%%

Imagine that you have a set of sentences which you want to classify as "happy" and "angry". And you identified that the sentences contain only two words: *aack* and *beep*. For each of the sentences (data point in the given dataset) you count the number of those two words ($x_1$ and $x_2$) and compare them with each other. If there are more "beep" ($x_2 > x_1$), the sentence should be classified as "angry", if not ($x_2 <= x_1$), it is a "happy" sentence. Which means that there will be some straight line separating those two classes.
%%
想象你有一组句子，你想将它们分类为表达“快乐”和“愤怒”。
然后你发现句子中仅包含两个词：*aack* 和 *beep*。
每个句子（即数据点）统计了两个单词的数量（$x_1$ 和 $x_2$）,然后对它们相互比较。
如果句子中有更多的“beep”($x_2 > x_1$)，这个句子需要分类到“愤怒”，与之相反（$x_2 <= x_1$），它是一个表达“快乐”的句子。
这意味这这里将有一条直线将它们分割为两组。
%%

Let's take a very simple set of $4$ sentenses: 
- "Beep!" 
- "Aack?" 
- "Beep aack..." 
- "!?"

Here both $x_1$ and $x_2$ will be either $0$ or $1$. You can plot those points in a plane, and see the points (observations) belong to two classes, "angry" (red) and "happy" (blue), and a straight line can be used as a decision boundary to separate those two classes. An example of such a line is plotted. 
%%
这里的 $x_{1}$ 和 $x_2$ 都是 0 或者 1。
你可以在平面上绘制它们的数据点，然后看看这些数据点（观察点），有 “愤怒”（红色）和 “快乐”（蓝色） 两个类别，并且可以用一条直线可以作为决策边界来分离这两个类别。
绘制示例如下。
%%

```python
fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
x_line = np.arange(xmin, xmax, 0.1)
# Data points (observations) from two classes.
ax.scatter(0, 0, color="b")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="b")
ax.scatter(1, 1, color="b")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# One of the lines which can be used as a decision boundary to separate two classes.
ax.plot(x_line, x_line + 0.5, color="black")
plt.plot()
```

> [!result]
>     []
![C2_W3_Lab_2_Classification_with_Perceptron_6_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Lab_2_Classification_with_Perceptron_6_1.png)

This particular line is chosen using common sense, just looking at the visual representation of the observations. Such classification problem is called a problem with **two linearly separable classes**.

这根线是凭借常识选定的，仅凭观察结果的视觉呈现即可确定。
这类分类问题被称为具有**两个线性可分类别的问题**。

The line $x_1-x_2+0.5 = 0$ (or $x_2 = x_1 + 0.5$) can be used as a separating line for the problem. All of the points $(x_1, x_2)$ above this line, such that $x_1-x_2+0.5 < 0$ (or $x_2 > x_1 + 0.5$), will be considered belonging to the red class, and below this line $x_1-x_2+0.5 > 0$ ($x_2 < x_1 + 0.5$) - belonging to the blue class. So the problem can be rephrased: in the expression $w_1x_1+w_2x_2+b=0$ find the values for the parameters $w_1$, $w_2$ and the threshold $b$, so that the line can serve as a decision boundary.
%%
这根线 $x_1-x_2+0.5 = 0$ (或者 $x_2 = x_1 + 0.5$) 可以作为该问题的分割线。
所有在这根线上面的点，比如 $x_1-x_2+0.5 < 0$ (or $x_2 > x_1 + 0.5$)，它们都将视为红色的类别，然后在这根线下面的点比如 $x_1-x_2+0.5 > 0$ ($x_2 < x_1 + 0.5$) 都属于蓝色的类别。
所以这个问题可以这样表述：
在表达式 $w_1x_1+w_2x_2+b=0$ 中寻找参数 $w_1$, $w_2$ 和截距 $b$，使得这条线能够作为决策边界。
%%

In this simple example you could solve the problem of finding the decision boundary just looking at the plot: $w_1 = 1$, $w_2 = -1$, $b = 0.5$. But what if the problem is more complicated? You can use a simple neural network model to do that! Let's implement it for this example and then try it for more complicated problem.
%%
在这个简单的示例中你可以通过绘图找到这个问题的决策边界：$w_1 = 1$, $w_2 = -1$, $b = 0.5$。
但是如果这个问题更复杂呢？
简单的神经网络模型可以做到！
让我们实现这个示例，然后尝试用它解决更复杂的问题。
%%
<a name='2'></a>
## 2 - Single Perceptron Neural Network with Activation Function

You already have constructed and trained a neural network model with one **perceptron**. Here a similar model can be used, but with an activation function. Then a single perceptron basically works as a threshold function.
%%
你已经准备好构建和训练单层感知器的神经网络模型了。
这里可以使用类似的模型，但带有一个激活函数。
那么，单个感知器基本上就像一个阈值函数那样工作。
%%

<a name='2.1'></a>
### 2.1 - Neural Network Structure

The neural network components are shown in the following scheme:


![nn_model_classification_1_layer.png](https://obsidian-image.wwtt.xyz/2025/12/nn_model_classification_1_layer.png)


Similarly to the previous lab, the input layer contains two nodes $x_1$ and $x_2$. Weight vector $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ and bias ($b$) are the parameters to be updated during the model training. First step in the forward propagation is the same as in the previous lab. For every training example $x^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)}\end{bmatrix}$:
%%
和上一个实验类似，它的输入层包含两个节点—— $x_1$ 和 $x_2$。
在模型训练的时候不断更新权重向量 $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 和偏置 ($b$) 。
对于每一个训练样本 $x^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)}\end{bmatrix}$ ，第一步和上一个实验一样进行前向传播：
%%
$$z^{(i)} = w_1x_1^{(i)} + w_2x_2^{(i)} + b = Wx^{(i)} + b.\tag{1}$$


But now you cannot take a real number $z^{(i)}$ into the output as you need to perform classification. It could be done with a discrete approach: compare the result with zero, and classify as $0$ (blue) if it is below zero and $1$ (red) if it is above zero. Then define cost function as a percentage of incorrectly identified classes and perform backward propagation.
%%
但是现在你不能拿真实的数 $z^{(i)}$ 作为输出，你需要执行分类。
它可以用离散方法来实现：将结果与零比较，如果小于 0 则分类为 0（blue），如果大于 0 则分类为 1（read）。
%%

This extra step in the forward propagation is actually an application of an **activation function**. It would be possible to implement the discrete approach described above (with unit step function) for this problem, but it turns out that there is a continuous approach that works better and is commonly used in more complicated neural networks. So you will implement it here: single perceptron with sigmoid activation function.
%%
这个在前向传播里额外的步骤实际上是**激活函数**的一种应用。
对于这个问题，固然可以采用上述（包含单位阶跃函数）的离散方法，但实际上，存在一种效果更好且在更复杂神经网络中广泛使用的连续方法。
所以你将在这里实现它——具用 Sigmoid 激活函数的单层感知器。
%%
Sigmoid activation function is defined as

$$a = \sigma\left(z\right) = \frac{1}{1+e^{-z}}.\tag{2}$$

Then a threshold value of $0.5$ can be used for predictions: $1$ (red) if  $a > 0.5$ and $0$ (blue) otherwise. Putting it all together, mathematically the single perceptron neural network with sigmoid activation function can be expressed as:
%%
将阈值设置为 0.5 进行预测：如果 a 大于 0.5 则为 1（红色），反之为 0（蓝色）。
总而言之，数学上这个含有 sigmoid 激活函数的单层感知神经网络的表达式如下：
%%
$$\begin{align}
z^{(i)} &=  W x^{(i)} + b,\\
a^{(i)} &= \sigma\left(z^{(i)}\right).\\\tag{3}
\end{align}$$

If you have $m$ training examples organised in the columns of ($2 \times m$) matrix $X$, you can apply the activation function element-wise. So the model can be written as:
%%
如果有 $m$ 个训练样本，将它们组成形状为 $2 \times m$ 的矩阵 $X$，这样你可以对每个元素应用激活函数。
所以这个模型也可以写作如下：
%%
$$
\begin{align}
Z &=  W X + b,\\
A &= \sigma\left(Z\right),\\\tag{4}
\end{align}
$$

where $b$ is broadcasted to the vector of a size ($1 \times m$). 
%%
其中 b 同样会广播为形状 $1 \times m$ 的矩阵。
%%

When dealing with classification problems, the most commonly used cost function is the **log loss**, which is described by the following equation:

%%
处理分类问题的时候，一般来说成本函数使用**对数损失**，它可以用下面的方程进行描述：
%%

$$\mathcal{L}\left(W, b\right) = \frac{1}{m}\sum_{i=1}^{m} L\left(W, b\right) = \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small -y^{(i)}\log\left(a^{(i)}\right) - (1-y^{(i)})\log\left(1- a^{(i)}\right)  \large  \right) \small,\tag{5}$$

where $y^{(i)} \in \{0,1\}$ are the original labels and $a^{(i)}$ are the continuous output values of the forward propagation step (elements of array $A$).
%%
其中 $y^{(i)} \in \{0,1\}$ 为原式的标签， $a^{(i)}$ 是一个前向传播步骤的连续输出值（数组 A 的元素）。
%%

You want to minimize the cost function during the training. To implement gradient descent, calculate partial derivatives using chain rule:
%%
在训练期间你的目的是最小化成本函数。
实现梯度下降，使用链式法则计算偏导数。
%%

$$\begin{align}
\frac{\partial \mathcal{L} }{ \partial w_1 } &= 
\frac{1}{m}\sum_{i=1}^{m} \frac{\partial L }{ \partial a^{(i)}}
\frac{\partial a^{(i)} }{ \partial z^{(i)}}\frac{\partial z^{(i)} }{ \partial w_1},\\
\frac{\partial \mathcal{L} }{ \partial w_2 } &= 
\frac{1}{m}\sum_{i=1}^{m} \frac{\partial L }{ \partial a^{(i)}}
\frac{\partial a^{(i)} }{ \partial z^{(i)}}\frac{\partial z^{(i)} }{ \partial w_2},\tag{6}\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\sum_{i=1}^{m} \frac{\partial L }{ \partial a^{(i)}}
\frac{\partial a^{(i)} }{ \partial z^{(i)}}\frac{\partial z^{(i)} }{ \partial b}.
\end{align}$$

As discussed in the videos, $\frac{\partial L }{ \partial a^{(i)}}\frac{\partial a^{(i)} }{ \partial z^{(i)}} = \left(a^{(i)} - y^{(i)}\right)$, $\frac{\partial z^{(i)}}{ \partial w_1} = x_1^{(i)}$, $\frac{\partial z^{(i)}}{ \partial w_2} = x_2^{(i)}$ and $\frac{\partial z^{(i)}}{ \partial b} = 1$. Then $(6)$ can be rewritten as:
%%
正如视频中讨论那样，$\frac{\partial L }{ \partial a^{(i)}}\frac{\partial a^{(i)} }{ \partial z^{(i)}} = \left(a^{(i)} - y^{(i)}\right)$, $\frac{\partial z^{(i)}}{ \partial w_1} = x_1^{(i)}$, $\frac{\partial z^{(i)}}{ \partial w_2} = x_2^{(i)}$ 和 $\frac{\partial z^{(i)}}{ \partial b} = 1$.代入式子 $(6)$ 后得到下面的式子：
%%

$$\begin{align}
\frac{\partial \mathcal{L} }{ \partial w_1 } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(a^{(i)} - y^{(i)}\right)x_1^{(i)},\\
\frac{\partial \mathcal{L} }{ \partial w_2 } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(a^{(i)} - y^{(i)}\right)x_2^{(i)},\tag{7}\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(a^{(i)} - y^{(i)}\right).
\end{align}$$

Note that the obtained expressions $(7)$ are exactly the same as in the section $3.2$ of the previous lab, when multiple linear regression model was discussed. Thus, they can be rewritten in a matrix form:
%%
注意式子 7 其实和上一个实验中 3.2 的表达式时相同的，这在多元线性回归模型中讨论过了。
所以它们同样可以重写成矩阵：
%%

$$\begin{align}
\frac{\partial \mathcal{L} }{ \partial W } &= 
\begin{bmatrix} \frac{\partial \mathcal{L} }{ \partial w_1 } & 
\frac{\partial \mathcal{L} }{ \partial w_2 }\end{bmatrix} = \frac{1}{m}\left(A - Y\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b } &= \frac{1}{m}\left(A - Y\right)\mathbf{1}.
\tag{8}
\end{align}
$$

where $\left(A - Y\right)$ is an array of a shape ($1 \times m$), $X^T$ is an array of a shape ($m \times 2$) and $\mathbf{1}$ is just a ($m \times 1$) vector of ones.
%%
其中 $\left(A - Y\right)$ 是一个形状为 $1 \times m$ 的数组，$X^T$ 是一个形状为 $m \times 2$ 的数组，而 1 则仅仅是一个形状为 $m \times 1$ 的全一向量。
%%
Then you can update the parameters:
%%
然后你就可以更新参数了
%%

$$\begin{align}
W &= W - \alpha \frac{\partial \mathcal{L} }{ \partial W },\\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b },
\tag{9}\end{align}
$$
where $\alpha$ is the learning rate. Repeat the process in a loop until the cost function stops decreasing.
%%
其中 $\alpha$ 是学习率。
重复运行这个循环直到成本函数停止下降。
%%
Finally, the predictions for some example $x$ can be made taking the output $a$ and calculating $\hat{y}$ as

%%
最后，对于示例 $x$ 的预测可以通过取输出 $a$ 并计算 $\hat{y}$ 来实现，具体如下：
%%

$$\hat{y} = \begin{cases} 1 & \mbox{if } a > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{10}$$

<a name='2.2'></a>
### 2.2 - Dataset

Let's get the dataset you will work on. The following code will create $m=30$ data points $(x_1, x_2)$, where $x_1, x_2 \in \{0,1\}$ and save them in the `NumPy` array `X` of a shape $(2 \times m)$ (in the columns of the array). The labels ($0$: blue, $1$: red) will be calculated so that $y = 1$ if $x_1 = 0$ and $x_2 = 1$, in the rest of the cases $y=0$. The labels will be saved in the array `Y` of a shape $(1 \times m)$.

%%
首先处理数据集。
下面的代码创建了 $m=30$ 的数据点 $(x_1, x_2)$，其中 $x_1, x_2 \in \{0,1\}$，并且将它们保存在 Numpy 数组 $X$ 中，这个数组的形状为 $(2 \times m)$（x 为它们的列）。
标签 ($0$: blue, $1$: red) 的处理流程是当 $x_1 = 0$ and $x_2 = 1$ 时标签  $y = 1$，其他情况下则 $y=0$。
标签将保存在数组 $Y$ 中，它的形状是 $(1 \times m)$。
%%

```python
m = 30

X = np.random.randint(0, 2, (2, m))
Y = np.logical_and(X[0] == 0, X[1] == 1).astype(int).reshape((1, m))

print('Training dataset X containing (x1, x2) coordinates in the columns:')
print(X)
print('Training dataset Y containing labels of two classes (0: blue, 1: red)')
print(Y)

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have m = %d training examples!' % (X.shape[1]))
```

> [!result]
>     Training dataset X containing (x1, x2) coordinates in the columns:
>     [[0 0 1 1 0 0 0 1 1 1 0 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 0]
>      [0 1 0 1 1 0 1 0 0 1 1 0 0 1 0 1 0 1 1 1 1 0 1 0 0 1 1 1 0 0]]
>     Training dataset Y containing labels of two classes (0: blue, 1: red)
>     [[0 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 0 0]]
>     The shape of X is: (2, 30)
>     The shape of Y is: (1, 30)
>     I have m = 30 training examples!


<a name='2.3'></a>
### 2.3 - Define Activation Function

The sigmoid function $(2)$ for a variable $z$ can be defined with the following code:


```python
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
print("sigmoid(-2) = " + str(sigmoid(-2)))
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(3.5) = " + str(sigmoid(3.5)))
```

> [!result]
>     sigmoid(-2) = 0.11920292202211755
>     sigmoid(0) = 0.5
>     sigmoid(3.5) = 0.9706877692486436


It can be applied to a `NumPy` array element by element:


```python
print(sigmoid(np.array([-2, 0, 3.5])))
```

> [!result]
>     [0.11920292 0.5        0.97068777]


<a name='3'></a>
## 3 - Implementation of the Neural Network Model

Implementation of the described neural network will be very similar to the previous lab. The differences will be only in the functions `forward_propagation` and `compute_cost`!

%%
实现所描述的神经函数和上一个实验非常的相似，不同的地方只有函数 `forward_propagation` and `compute_cost`！
%%
<a name='3.1'></a>
### 3.1 - Defining the Neural Network Structure

Define two variables:
- `n_x`: the size of the input layer
- `n_y`: the size of the output layer

%%
定义两个变量：
- `n_x`: 输入层的尺寸
- `n_y`: 输出层的尺寸
%%

using shapes of arrays `X` and `Y`.


```python
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))
```

> [!result]
>     The size of the input layer is: n_x = 2
>     The size of the output layer is: n_y = 1


<a name='3.2'></a>
### 3.2 - Initialize the Model's Parameters

Implement the function `initialize_parameters()`, initializing the weights array of shape $(n_y \times n_x) = (1 \times 1)$ with random values and the bias vector of shape $(n_y \times 1) = (1 \times 1)$ with zeros.

%%
构建函数 `initialize_parameters()`，初始化形状为 $(n_y \times n_x) = (1 \times 1)$，值为随机值的权重数组，初始化形状为 $(n_y \times 1) = (1 \times 1)$ 且值为 0 的偏置向量。
%%

```python
def initialize_parameters(n_x, n_y):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """
    
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))

    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))
```

    W = [[-0.00768836 -0.00230031]]
    b = [[0.]]


<a name='3.3'></a>
### 3.3 - The Loop

Implement `forward_propagation()` following the equation $(4)$ in the section [2.1](#2.1):

$$
\begin{align}
Z &=  W X + b,\\
A &= \sigma\left(Z\right).
\end{align}$$


```python
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A -- The output
    """
    W = parameters["W"]
    b = parameters["b"]
    
    # Forward Propagation to calculate Z.
    Z = np.matmul(W, X) + b
    A = sigmoid(Z)

    return A

A = forward_propagation(X, parameters)

print("Output vector A:", A)
```

> [!result]
>     Output vector A: [[0.5        0.49942492 0.49807792 0.49750285 0.49942492 0.5
>       0.49942492 0.49807792 0.49807792 0.49750285 0.49942492 0.49807792
>       0.49807792 0.49750285 0.5        0.49750285 0.49807792 0.49942492
>       0.49942492 0.49942492 0.49942492 0.49807792 0.49750285 0.5
>       0.5        0.49942492 0.49750285 0.49942492 0.5        0.5       ]]


Your weights were just initialized with some random values, so the model has not been trained yet. 

%%
你的权重已经初始化为一些随机值了，所以模型尚未开始训练。
%%

Define a cost function $(5)$ which will be used to train the model:

%%
定义成本函数 $(5)$，这将用于训练这个模型：
%%

$$\mathcal{L}\left(W, b\right)  = \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small -y^{(i)}\log\left(a^{(i)}\right) - (1-y^{(i)})\log\left(1- a^{(i)}\right)  \large  \right) \small.$$


```python
def compute_cost(A, Y):
    """
    Computes the log loss cost function
    
    Arguments:
    A -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- log loss
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
    cost = 1/m * np.sum(logprobs)
    
    return cost

print("cost = " + str(compute_cost(A, Y)))
```

> [!result]
>     cost = 0.6916391611507908


Calculate partial derivatives as shown in $(8)$:

$$
\begin{align}
\frac{\partial \mathcal{L} }{ \partial W } &= \frac{1}{m}\left(A - Y\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b } &= \frac{1}{m}\left(A - Y\right)\mathbf{1}.
\end{align}
$$

```python
def backward_propagation(A, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    A -- the output of the neural network of shape (n_y, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity. 
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(A, X, Y)

print("dW = " + str(grads["dW"]))
print("db = " + str(grads["db"]))
```

> [!result]
>     dW = [[ 0.21571875 -0.06735779]]
>     db = [[0.16552706]]


Update parameters as shown in $(9)$:

$$
\begin{align}
W &= W - \alpha \frac{\partial \mathcal{L} }{ \partial W },\\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b }.\end{align}
$$
```python
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    learning_rate -- learning rate parameter for gradient descent
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]
    
    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters_updated = update_parameters(parameters, grads)

print("W updated = " + str(parameters_updated["W"]))
print("b updated = " + str(parameters_updated["b"]))
```

> [!result]
>     W updated = [[-0.26655087  0.07852904]]
>     b updated = [[-0.19863247]]


<a name='3.4'></a>
### 3.4 - Integrate parts 3.1, 3.2 and 3.3 in nn_model() and make predictions

Build your neural network model in `nn_model()`.


```python
def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_y)
    
    # Loop
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A".
        A = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A, Y". Outputs: "cost".
        cost = compute_cost(A, Y)
        
        # Backpropagation. Inputs: "A, X, Y". Outputs: "grads".
        grads = backward_propagation(A, X, Y)
    
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```


```python
parameters = nn_model(X, Y, num_iterations=50, learning_rate=1.2, print_cost=True)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))
```

> [!result]
>    Cost after iteration 0: 0.693480
>    Cost after iteration 1: 0.608586
>    Cost after iteration 2: 0.554475
>    Cost after iteration 3: 0.513124
>    Cost after iteration 4: 0.478828
>    Cost after iteration 5: 0.449395
>    Cost after iteration 6: 0.423719
>    Cost after iteration 7: 0.401089
>    Cost after iteration 8: 0.380986
>    Cost after iteration 9: 0.363002
>    Cost after iteration 10: 0.346813
>    Cost after iteration 11: 0.332152
>    Cost after iteration 12: 0.318805
>    Cost after iteration 13: 0.306594
>    Cost after iteration 14: 0.295369
>    Cost after iteration 15: 0.285010
>    Cost after iteration 16: 0.275412
>    Cost after iteration 17: 0.266489
>    Cost after iteration 18: 0.258167
>    Cost after iteration 19: 0.250382
>    Cost after iteration 20: 0.243080
>    Cost after iteration 21: 0.236215
>    Cost after iteration 22: 0.229745
>    Cost after iteration 23: 0.223634
>    Cost after iteration 24: 0.217853
>    Cost after iteration 25: 0.212372
>    Cost after iteration 26: 0.207168
>    Cost after iteration 27: 0.202219
>    Cost after iteration 28: 0.197505
>    Cost after iteration 29: 0.193009
>    Cost after iteration 30: 0.188716
>    Cost after iteration 31: 0.184611
>    Cost after iteration 32: 0.180682
>    Cost after iteration 33: 0.176917
>    Cost after iteration 34: 0.173306
>    Cost after iteration 35: 0.169839
>    Cost after iteration 36: 0.166507
>    Cost after iteration 37: 0.163303
>    Cost after iteration 38: 0.160218
>    Cost after iteration 39: 0.157246
>    Cost after iteration 40: 0.154382
>    Cost after iteration 41: 0.151618
>    Cost after iteration 42: 0.148950
>    Cost after iteration 43: 0.146373
>    Cost after iteration 44: 0.143881
>    Cost after iteration 45: 0.141471
>    Cost after iteration 46: 0.139139
>    Cost after iteration 47: 0.136881
>    Cost after iteration 48: 0.134694
>    Cost after iteration 49: 0.132574
>     W = [[-3.57177421  3.24255633]]
>     b = [[-1.58411051]]


You can see that after about $40$ iterations the cost function does keep decreasing, but not as much. It is a sign that it might be reasonable to stop training there. The final model parameters can be used to find the boundary line and for making predictions. Let's visualize the boundary line.
%%
你可以看到在 40 个迭代后，成本函数虽然还在下降，但是幅度已经非常小了。
这表明，或许是时候停止在那里的训练了。
最终模型的参数可以找到这个边界线并进行预测。
让我们看看这条线。
%%

```python
def plot_decision_boundary(X, Y, parameters):
    W = parameters["W"]
    b = parameters["b"]

    fig, ax = plt.subplots()
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));
    
    x_line = np.arange(np.min(X[0,:]),np.max(X[0,:])*1.1, 0.1)
    ax.plot(x_line, - W[0,0] / W[0,1] * x_line + -b[0,0] / W[0,1] , color="black")
    plt.plot()
    plt.show()
    
plot_decision_boundary(X, Y, parameters)
```

> [!result]
![C2_W3_Lab_2_Classification_with_Perceptron_45_0.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Lab_2_Classification_with_Perceptron_45_0.png)


And make some predictions:


```python
def predict(X, parameters):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (blue: False / red: True)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A = forward_propagation(X, parameters)
    predictions = A > 0.5
    
    return predictions

X_pred = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 1]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")
```

> [!result]
>     Coordinates (in the columns):
>     [[1 1 0 0]
>      [0 1 0 1]]
>     Predictions:
>     [[False False False  True]]


Pretty good for such a simple neural network!

<a name='4'></a>
## 4 - Performance on a Larger Dataset

Construct a larger and more complex dataset with the function `make_blobs` from the `sklearn.datasets` library:
%%
使用函数 `make_blobs` 构建一个更大更复杂的数据集，这里我们用了 `sklearn.datasets` 库：
%%

```python
# Dataset
n_samples = 1000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9]), 
                             cluster_std=1.4,
                             random_state=0)

X_larger = np.transpose(samples)
Y_larger = labels.reshape((1,n_samples))

plt.scatter(X_larger[0, :], X_larger[1, :], c=Y_larger, cmap=colors.ListedColormap(['blue', 'red']));
```

> [!result]
> ![C2_W3_Lab_2_Classification_with_Perceptron_50_0.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Lab_2_Classification_with_Perceptron_50_0.png)


And train your neural network for $100$ iterations.

然后通过迭代 100 词的神经网络进行训练

```python
parameters_larger = nn_model(X_larger, Y_larger, num_iterations=100, learning_rate=1.2, print_cost=False)
print("W = " + str(parameters_larger["W"]))
print("b = " + str(parameters_larger["b"]))
```

> [!result]
>     W = [[1.01643208 1.13651775]]
>     b = [[-10.65346577]]
> 
> 
>     /tmp/ipykernel_99753/3146768556.py:17: RuntimeWarning: divide by zero encountered in log
>       logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
>     /tmp/ipykernel_99753/3146768556.py:17: RuntimeWarning: invalid value encountered in multiply
>       logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)


Plot the decision boundary:

%%
绘制决策边界
%%

```python
plot_decision_boundary(X_larger, Y_larger, parameters_larger)
```

> [!result]
![C2_W3_Lab_2_Classification_with_Perceptron_54_0.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Lab_2_Classification_with_Perceptron_54_0.png)


Try to change values of the parameters `num_iterations` and `learning_rate` and see if the results will be different.
%%
尝试修改参数值 `num_iterations` 和 `learning_rate` 看看结果有什么变化。
%%

Congrats on finishing the lab!
