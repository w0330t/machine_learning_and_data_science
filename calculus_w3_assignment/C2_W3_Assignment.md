---
aliases: [Table of Contents]
tags: []
created: 2025-12-12, 07:55:37
modified: 2025-12-12, 08:20:20
---

Welcome to your week three programming assignment. You are ready to build a neural network with two layers and train it to solve a classification problem. 
%%
欢迎来到你的第三周编程作业。
你已经准备好构建一个双层的神经网络，训练它以解决分类问题。
%%
**After this assignment, you will be able to:**

- Implement a neural network with two layers to a classification problem
- Implement forward propagation using matrix multiplication
- Perform backward propagation
%%
- 实现一个用于解决分类问题的双层神经网络。
- 实现矩阵乘法的前向传播
- 执行反向传播
%%

# Table of Contents

- [ 1 - Classification Problem](#1)
- [ 2 - Neural Network Model with Two Layers](#2)
  - [ 2.1 - Neural Network Model with Two Layers for a Single Training Example](#2.1)
  - [ 2.2 - Neural Network Model with Two Layers for Multiple Training Examples](#2.2)
  - [ 2.3 - Cost Function and Training](#2.3)
  - [ 2.4 - Dataset](#2.4)
  - [ 2.5 - Define Activation Function](#2.5)
    - [ Exercise 1](#ex01)
- [ 3 - Implementation of the Neural Network Model with Two Layers](#3)
  - [ 3.1 - Defining the Neural Network Structure](#3.1)
    - [ Exercise 2](#ex02)
  - [ 3.2 - Initialize the Model's Parameters](#3.2)
    - [ Exercise 3](#ex03)
  - [ 3.3 - The Loop](#3.3)
    - [ Exercise 4](#ex04)
    - [ Exercise 5](#ex05)
    - [ Exercise 6](#ex06)
  - [ 3.4 - Integrate parts 3.1, 3.2 and 3.3 in nn_model()](#3.4)
    - [ Exercise 7](#ex07)
    - [ Exercise 8](#ex08)
- [ 4 - Optional: Other Dataset](#4)

## Packages

First, import all the packages you will need during this assignment.


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
## 1 - Classification Problem

In one of the labs this week, you trained a neural network with a single perceptron, performing forward and backward propagation. That simple structure was enough to solve a "linear" classification problem - finding a straight line in a plane that would serve as a decision boundary to separate two classes.
%%
在本周的第一个实验中，你训练了一个仅含单个感知机的神经网络，并且运行了前向和后向传播。
那个简单的结构足以解决一个“线性”分类问题——在平面上找到一个条线，用这条线作为区分两个类别的决策边界。
%%
Imagine that now you have a more complicated problem: you still have two classes, but one line will not be enough to separate them.
%%
设想现在你面临一个更复杂的问题：你仍然有两个类别，但仅用一条线已不足以将它们分开。
%%


```python
fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
x_line = np.arange(xmin, xmax, 0.1)
# Data points (observations) from two classes.
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="b")
ax.scatter(1, 0, color="b")
ax.scatter(1, 1, color="r")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# Example of the lines which can be used as a decision boundary to separate two classes.
ax.plot(x_line, -1 * x_line + 1.5, color="black")
ax.plot(x_line, -1 * x_line + 0.5, color="black")
plt.plot()
```


> [!result]
>     []
![C2_W3_Assignment_8_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Assignment_8_1.png)
    


This logic can appear in many applications. For example, if you train a model to predict whether you should buy a house knowing its size and the year it was built. A big new house will not be affordable, while a small old house will not be worth buying. So, you might be interested in either a big old house, or a small new house.
%%
这种逻辑可能会出现在很多应用中出现。
举一个例子，如果你训练一个模型，根据房屋的面积和建造年份来预测是否应该购买。
一个新的大房子你可能无法负担，但是一个小的旧房子又不值得购买。
所以你可能对一个旧的大房子感兴趣，或者是一个新的小房子。
%%
The one perceptron neural network is not enough to solve such classification problem. Let's look at how you can adjust that model to find the solution.
%%
仅含有一个感知机的神经网络不能解决这样的分类问题。
让我们看看你如何调整这个模型找到这个答案。
%%
In the plot above, two lines can serve as a decision boundary. Your intuition might tell you that you should also increase the number of perceptrons. And that is absolutely right! You need to feed your data points (coordinates $x_1$, $x_2$) into two nodes separately and then unify them somehow with another one to make a decision. 
%%
在上面的绘图中，两根线可以作为其决策边界。
你的直觉可能告诉你应该增加感知机的数量来解决这个问题。
这绝对是明智的选择！
你需要将你的数据点（坐标 $x_{1},x_{2}$）分别输入两个节点中，然后将它们统一起来，以便做出决定。
%%
Now let's figure out the details, build and train your first multi-layer neural network!
%%
现在我们来详细探讨一下，构建和训练你的第一个多层神经网络！
%%

<a name='2'></a>
## 2 - Neural Network Model with Two Layers

<a name='2.1'></a>
### 2.1 - Neural Network Model with Two 
Layers for a Single Training Example

![nn_model_2_layers.png](https://obsidian-image.wwtt.xyz/2025/12/nn_model_2_layers.png)

The input and output layers of the neural network are the same as for one perceptron model, but there is a **hidden layer** now in between them. The training examples $x^{(i)}=\begin{bmatrix}x_1^{(i)} \\ x_2^{(i)}\end{bmatrix}$ from the input layer of size $n_x = 2$ are first fed into the hidden layer of size $n_h = 2$. They are simultaneously fed into the first perceptron with weights $W_1^{[1]}=\begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]}\end{bmatrix}$, bias  $b_1^{[1]}$; and into the second perceptron with weights $W_2^{[1]}=\begin{bmatrix}w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$, bias $b_2^{[1]}$. The integer in the square brackets $^{[1]}$ denotes the layer number, because there are two layers now with their own parameters and outputs, which need to be distinguished. 
%%
这个神经网络的输入和输出层和单个感知机模型是一样的，但是这里有一个**隐藏层**夹在它们的中间。
来自形状为 $n_x = 2$ 输入层的训练样本 $x^{(i)}=\begin{bmatrix}x_1^{(i)} \\ x_2^{(i)}\end{bmatrix}$ 首先被送入大小为 $n_h = 2$ 的隐藏层。
它们同时被送入两个感知机，第一个感知机的权重为 $W_1^{[1]}=\begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]}\end{bmatrix}$，偏置为 $b_1^{[1]}$；第二个感知机的权重为 $W_2^{[1]}=\begin{bmatrix}w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$，偏置为 $b_2^{[1]}$。
右上角的反括号中的数字 $^{[1]}$ 表示层号，因为这里有两层，它们分别都有自己的参数和输出需要加以区分。
%%

$$
\begin{align}
z_1^{[1](i)} &= w_{1,1}^{[1]} x_1^{(i)} + w_{2,1}^{[1]} x_2^{(i)} + b_1^{[1]} = W_1^{[1]}x^{(i)} + b_1^{[1]},\\
z_2^{[1](i)} &= w_{1,2}^{[1]} x_1^{(i)} + w_{2,2}^{[1]} x_2^{(i)} + b_2^{[1]} = W_2^{[1]}x^{(i)} + b_2^{[1]}.\tag{1}
\end{align}
$$

These expressions for one training example $x^{(i)}$ can be rewritten in a matrix form :
%%
这个表达式可以将训练样本 $x^{(i)}$ 重写成一个矩阵：
%%
$$z^{[1](i)} = W^{[1]} x^{(i)} + b^{[1]},\tag{2}$$

where 

-  $z^{[1](i)} = \begin{bmatrix}z_1^{[1](i)} \\ z_2^{[1](i)}\end{bmatrix}$ is vector of size $\left(n_h \times 1\right) = \left(2 \times 1\right)$; 

- $W^{[1]} = \begin{bmatrix}W_1^{[1]} \\ W_2^{[1]}\end{bmatrix} = \begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]} \\ w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$ is matrix of size $\left(n_h \times n_x\right) = \left(2 \times 2\right)$;

-  $b^{[1]} = \begin{bmatrix}b_1^{[1]} \\ b_2^{[1]}\end{bmatrix}$ is vector of size $\left(n_h \times 1\right) = \left(2 \times 1\right)$.
%%
其中
- $z^{[1](i)} = \begin{bmatrix}z_1^{[1](i)} \\ z_2^{[1](i)}\end{bmatrix}$ 是一个尺寸为 $\left(n_h \times 1\right) = \left(2 \times 1\right)$ 的向量；
- $W^{[1]} = \begin{bmatrix}W_1^{[1]} \\ W_2^{[1]}\end{bmatrix} = \begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]} \\ w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$ 是一个尺寸为 $\left(n_h \times n_x\right) = \left(2 \times 2\right)$ 的矩阵；
-  $b^{[1]} = \begin{bmatrix}b_1^{[1]} \\ b_2^{[1]}\end{bmatrix}$ 是一个尺寸为 $\left(n_h \times 1\right) = \left(2 \times 1\right)$ 的向量。
%%

Next, the hidden layer activation function needs to be applied for each of the elements in the vector $z^{[1](i)}$. Various activation functions can be used here and in this model you will take the sigmoid function $\sigma\left(x\right) = \frac{1}{1 + e^{-x}}$. Remember that its derivative is $\frac{d\sigma}{dx} = \sigma\left(x\right)\left(1-\sigma\left(x\right)\right)$. The output of the hidden layer is a vector of size $\left(n_h \times 1\right) = \left(2 \times 1\right)$:
%%
下一步，隐藏层里的激活函数需要应用在向量 $z^{[1](i)}$ 的每个元素中。
各种激活函数都可以应用在这里，但在模型中采用了 sigmoid 函数 $\sigma\left(x\right) = \frac{1}{1 + e^{-x}}$。
记住它的导数是 $\frac{d\sigma}{dx} = \sigma\left(x\right)\left(1-\sigma\left(x\right)\right)$。
隐藏层输出的向量形状是 $\left(n_h \times 1\right) = \left(2 \times 1\right)$:
%%

$$a^{[1](i)} = \sigma\left(z^{[1](i)}\right) = 
\begin{bmatrix}\sigma\left(z_1^{[1](i)}\right) \\ \sigma\left(z_2^{[1](i)}\right)\end{bmatrix}.\tag{3}$$

Then the hidden layer output gets fed into the output layer of size $n_y = 1$. This was covered in the previous lab, the only difference are: $a^{[1](i)}$ is taken instead of $x^{(i)}$ and layer notation $^{[2]}$ appears to identify all parameters and outputs:
%%
隐藏层输出的值形状为 $n_y = 1$，它会进入输出层。
这在上一个实验中已经讨论过了，这里唯一的不同是 $a^{[1](i)}$ 替代了 $x^{(i)}$，并且使用了层标记 $^{[2]}$ 来表示所有的参数和输出值：
%%
$$z^{[2](i)} = w_1^{[2]} a_1^{[1](i)} + w_2^{[2]} a_2^{[1](i)} + b^{[2]}= W^{[2]} a^{[1](i)} + b^{[2]},\tag{4}$$


-  $z^{[2](i)}$ and $b^{[2]}$ are scalars for this model, as $\left(n_y \times 1\right) = \left(1 \times 1\right)$; 

-  $W^{[2]} = \begin{bmatrix}w_1^{[2]} & w_2^{[2]}\end{bmatrix}$ is vector of size $\left(n_y \times n_h\right) = \left(1 \times 2\right)$.

%%
其中
-  $z^{[2](i)}$ 和 $b^{[2]}$ 是这个模型的标量, 形状为 $\left(n_y \times 1\right) = \left(1 \times 1\right)$; 
-  $W^{[2]} = \begin{bmatrix}w_1^{[2]} & w_2^{[2]}\end{bmatrix}$ 是向量，形状为 $\left(n_y \times n_h\right) = \left(1 \times 2\right)$.
%%

Finally, the same sigmoid function is used as the output layer activation function:
%%
最后，同样采用 sigmoid 函数作为输出层的激活函数：
%%
$$a^{[2](i)} = \sigma\left(z^{[2](i)}\right).\tag{5}$$

Mathematically the two layer neural network model for each training example $x^{(i)}$ can be written with the expressions $(2) - (5)$. Let's rewrite them next to each other for convenience:
%%
在数学上，双层的神经网络模型每个训练样本 $x^{(i)}$ 都可以写作表达式 $(2) - (5)$。
为了方便起见，让我们把它们写在一起。
%%
$$
\begin{align}
z^{[1](i)} &= W^{[1]} x^{(i)} + b^{[1]},\\
a^{[1](i)} &= \sigma\left(z^{[1](i)}\right),\\
z^{[2](i)} &= W^{[2]} a^{[1](i)} + b^{[2]},\\
a^{[2](i)} &= \sigma\left(z^{[2](i)}\right).\\
\end{align}
\tag{6}
$$
Note, that all of the parameters to be trained in the model are without $^{(i)}$ index - they are independent on the input data.

%%
注意，模型中所有待训练的参数均不带 $^{(i)}$ 索引，这意味着它们独立于输入数据。
%%

Finally, the predictions for some example $x^{(i)}$ can be made taking the output $a^{[2](i)}$ and calculating $\hat{y}$ as: $\hat{y} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5, \\ 0 & \mbox{otherwise }. \end{cases}$.
%%
最后，对于样本 $x^{(i)}$ 的预测，可以通过输出 $a^{[2](i)}$ 并计算 $\hat{y}$ 来完成。
%%

<a name='2.2'></a>
### 2.2 - Neural Network Model with Two Layers for Multiple Training Examples

Similarly to the single perceptron model, $m$ training examples can be organised in a matrix $X$ of a shape ($2 \times m$), putting $x^{(i)}$ into columns. Then the model $(6)$ can be rewritten in terms of matrix multiplications:
%%
类似单感知机模型，$m$ 个训练样本可以组织成一个形状为 $(2 × m)$ 的矩阵 $X$，将 $x^{(i)}$ 放入这个矩阵的列中。
然后式子（6）可以用矩阵乘法的形式进行重写：
%%
$$
\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]},\\
A^{[1]} &= \sigma\left(Z^{[1]}\right),\\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]},\\
A^{[2]} &= \sigma\left(Z^{[2]}\right),\\
\end{align}
\tag{7}
$$
where $b^{[1]}$ is broadcasted to the matrix of size $\left(n_h \times m\right) = \left(2 \times m\right)$ and $b^{[2]}$ to the vector of size $\left(n_y \times m\right) = \left(1 \times m\right)$. It would be a good exercise for you to have a look at the expressions $(7)$ and check that sizes of the matrices will actually match to perform required multiplications.
%%
其中 $b^{[1]}$ 会被广播为矩阵，其形状为 $\left(n_h \times m\right) = \left(2 \times m\right)$，同样的 $b^{[2]}$ 会变为向量，其形状为 $\left(n_y \times m\right) = \left(1 \times m\right)$。
仔细观察这些表达式 $(7)$，然后检查检查矩阵的形状是否匹配以执行所需的乘法运算。
%%

You have derived expressions to perform forward propagation. Time to evaluate your model and train it.

%%
你已经推导出前向传播的表达式，是时候评估你的模型并且训练它了。
%%

<a name='2.3'></a>
### 2.3 - Cost Function and Training

For the evaluation of this simple neural network you can use the same cost function as for the single perceptron case - log loss function. Originally initialized weights were just some random values, now you need to perform training of the model: find such set of parameters $W^{[1]}$, $b^{[1]}$, $W^{[2]}$, $b^{[2]}$, that will minimize the cost function.
%%
评估这个简单的神经网络，你同样可以使用单感知机的成本函数——对数损失函数。
最初的权重仅仅是一些随机值，现在你需要开始训练模型，寻找一组适合的参数 $W^{[1]}$, $b^{[1]}$, $W^{[2]}$, $b^{[2]}$，以最小化这个成本函数。
%%
Like in the previous example of a single perceptron neural network, the cost function can be written as:
%%
就像上一个单感知机神经网络的例子那样，这个成本函数可以写作：
%%
$$\mathcal{L}\left(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\right) = \frac{1}{m}\sum_{i=1}^{m} L\left(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\right) =  \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small - y^{(i)}\log\left(a^{[2](i)}\right) - (1-y^{(i)})\log\left(1- a^{[2](i)}\right)  \large  \right), \small\tag{8}$$

where $y^{(i)} \in \{0,1\}$ are the original labels and $a^{[2](i)}$ are the continuous output values of the forward propagation step (elements of array $A^{[2]}$).
%%
其中 $y^{(i)} \in \{0,1\}$ 是原始的标签，而 $a^{[2](i)}$ 是是前向传播的连续输出值。（数组 $A^{[2]}$ 的元素）。
%%
To minimize it, you can use gradient descent, updating the parameters with the following expressions:
%%
要对它最小化，你可以使用梯度下降，像下面表达式那样更新参数：
%%
$$
\begin{align}
W^{[1]} &= W^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[1]} },\\
b^{[1]} &= b^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[1]} },\\
W^{[2]} &= W^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[2]} },\\
b^{[2]} &= b^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[2]} },\\
\end{align}
\tag{9}
$$
where $\alpha$ is the learning rate.
%%
其中 $\alpha$ 是学习率
%%

To perform training of the model you need to calculate now $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$, $\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$, $\frac{\partial \mathcal{L} }{ \partial W^{[2]}}$, $\frac{\partial \mathcal{L} }{ \partial b^{[2]}}$. 
%%
要训练这个模型你现在需要计算 $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$、 $\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$、$\frac{\partial \mathcal{L} }{ \partial W^{[2]}}$ 和 $\frac{\partial \mathcal{L} }{ \partial b^{[2]}}$。
%%

Let's start from the end of the neural network. You can rewrite here the corresponding expressions for $\frac{\partial \mathcal{L} }{ \partial W }$ and $\frac{\partial \mathcal{L} }{ \partial b }$ from the single perceptron neural network:
%%
让我们从这个神经网络的末端开始。
你可以在这里写下 $\frac{\partial \mathcal{L} }{ \partial W }$ 和 $\frac{\partial \mathcal{L} }{ \partial b }$ 单感知机神经网络对应的表达式。
%%

$$\begin{align}
\frac{\partial \mathcal{L} }{ \partial W } &= 
\frac{1}{m}\left(A-Y\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\left(A-Y\right)\mathbf{1},\\
\end{align}
$$
where $\mathbf{1}$ is just a ($m \times 1$) vector of ones. Your one perceptron is in the second layer now, so $W$ will be exchanged with $W^{[2]}$, $b$ with $b^{[2]}$, $A$ with $A^{[2]}$, $X$ with $A^{[1]}$:
%%
其中 $1$ 只是一个形状为 ($m \times 1$) 的全一向量。
你的单感知机现在在第二层，所以 $W$ 将替换为 $W^{[2]}$， $b$ 替换为 $b^{[2]}$， $A$ 替换为 $A^{[2]}$，$X$ 替换为 $A^{[1]}$:
%%
$$
\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1}.\\
\end{align}
\tag{10}
$$

Let's now find $\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = \begin{bmatrix}\frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,1}^{[1]}} \\\frac{\partial \mathcal{L} }{ \partial w_{1,2}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,2}^{[1]}} \end{bmatrix}$. It was shown in the videos that

%%
现在让我们找到 $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$ 对应的矩阵。
之前在视频中也说过。
%%

$$\frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}}=\frac{1}{m}\sum_{i=1}^{m} \left( 
\left(a^{[2](i)} - y^{(i)}\right) 
w_1^{[2]} 
\left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_1^{(i)}\right)\tag{11}$$

If you do this accurately for each of the elements $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$, you will get the following matrix:
%%
如果你想要计算矩阵 $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$ 每个元素的准确值，那么你将得到这个矩阵：
%%
$$\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = \begin{bmatrix}
\frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,1}^{[1]}} \\
\frac{\partial \mathcal{L} }{ \partial w_{1,2}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,2}^{[1]}} \end{bmatrix}$$
$$= \frac{1}{m}\begin{bmatrix}
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_2^{(i)}\right)  \\
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_2^{(i)}\right)\end{bmatrix}\tag{12}$$

Looking at this, you can notice that all terms and indices somehow are very consistent, so it all can be unified into a matrix form. And that's true! $\left(W^{[2]}\right)^T = \begin{bmatrix}w_1^{[2]} \\ w_2^{[2]}\end{bmatrix}$ of size $\left(n_h \times n_y\right) = \left(2 \times 1\right)$ can be multiplied with the vector $A^{[2]} - Y$ of size $\left(n_y \times m\right) = \left(1 \times m\right)$, resulting in a matrix of size $\left(n_h \times m\right) = \left(2 \times m\right)$:
%%
你能注意到所有项和指数都是一致的，所以它们可以统一成一个矩阵形式。
形状为 $\left(n_h \times n_y\right) = \left(2 \times 1\right)$ 的 $\left(W^{[2]}\right)^T = \begin{bmatrix}w_1^{[2]} \\ w_2^{[2]}\end{bmatrix}$ 可以和形状为 $\left(n_y \times m\right) = \left(1 \times m\right)$ 的向量 $A^{[2]} - Y$ 相乘，得到形状为 $\left(n_h \times m\right) = \left(2 \times m\right)$ 的矩阵。
%%
$$\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)=
\begin{bmatrix}w_1^{[2]} \\ w_2^{[2]}\end{bmatrix}
\begin{bmatrix}\left(a^{[2](1)} - y^{(1)}\right) &  \cdots & \left(a^{[2](m)} - y^{(m)}\right)\end{bmatrix}
=\begin{bmatrix}
\left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]} & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]} \\
\left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]} & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \end{bmatrix}$$

Now taking matrix $A^{[1]}$ of the same size $\left(n_h \times m\right) = \left(2 \times m\right)$,
%%
现在来看看相同形状 $\left(n_h \times m\right) = \left(2 \times m\right)$ 的矩阵 $A^{[1]}$
%%
$$A^{[1]}
=\begin{bmatrix}
a_1^{[1](1)} & \cdots & a_1^{[1](m)} \\
a_2^{[1](1)} & \cdots & a_2^{[1](m)} \end{bmatrix},$$

you can calculate:
%%
你可以这样计算
%%
$$A^{[1]}\cdot\left(1-A^{[1]}\right)
=\begin{bmatrix}
a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right) & \cdots & a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right) \\
a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right) & \cdots & a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right) \end{bmatrix},$$

where "$\cdot$" denotes **element by element** multiplication.
%%
其中"$\cdot$"表示元素与元素相乘
%%

With the element by element multiplication,
%%
逐个元素相乘
%%
$$\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)=\begin{bmatrix}
\left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]}\left(a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]}\left(a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right)\right) \\
\left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]}\left(a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \left(a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right)\right) \end{bmatrix}.$$

If you perform matrix multiplication with $X^T$ of size $\left(m \times n_x\right) = \left(m \times 2\right)$, you will get matrix of size $\left(n_h \times n_x\right) = \left(2 \times 2\right)$:
%%
如果你用 $X^T$ （形状为 $\left(m \times n_x\right) = \left(m \times 2\right)$）执行矩阵乘法，你会得到大小为 $\left(n_h \times n_x\right) = \left(2 \times 2\right)$ 的矩阵：
%%
$$\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T = 
\begin{bmatrix}
\left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]}\left(a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]}\left(a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right)\right) \\
\left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]}\left(a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \left(a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right)\right) \end{bmatrix}
\begin{bmatrix}
x_1^{(1)} & x_2^{(1)} \\
\cdots & \cdots \\
x_1^{(m)} & x_2^{(m)}
\end{bmatrix}$$
$$=\begin{bmatrix}
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1 - a_1^{[1](i)}\right) \right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_2^{(i)}\right)  \\
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_2^{(i)}\right)\end{bmatrix}$$

This is exactly like in the expression $(12)$! So, $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$ can be written as a mixture of multiplications:
%%
正如表达式 $(12)$ 描述的那样，所以 $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$ 可以写作矩阵乘法：
%%
$$\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T\tag{13},$$

where "$\cdot$" denotes element by element multiplications.
%%
其中"$\cdot$"还是元素乘以元素
%%

Vector $\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$ can be found very similarly, but the last terms in the chain rule will be equal to $1$, i.e. $\frac{\partial z_1^{[1](i)}}{ \partial b_1^{[1]}} = 1$. Thus,
%%
向量 $\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$ 的计算过程非常的相似，但是它在链式法则的最后一项为 1，即 $\frac{\partial z_1^{[1](i)}}{ \partial b_1^{[1]}} = 1$，由此可得：
%%
$$\frac{\partial \mathcal{L} }{ \partial b^{[1]}} = \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1},\tag{14}$$

where $\mathbf{1}$ is a ($m \times 1$) vector of ones.
%%
其中 $1$ 是一个形状为 $m \times 1$ 的全一向量
%%
Expressions $(10)$, $(13)$ and $(14)$ can be used for the parameters update $(9)$ performing backward propagation:
%%
公式 $(10)$、$(13)$ 和 $(14)$ 算出了梯度，我们需要把这些梯度带入到公式 $(9)$ 中，从而完成反向传播阶段的参数更新。
%%
$$
\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1},\\
\frac{\partial \mathcal{L} }{ \partial W^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1},\\
\end{align}
\tag{15}
$$
where $\mathbf{1}$ is a ($m \times 1$) vector of ones.
%%
其中 $1$ 还是一个形状为 $m \times 1$ 的全一向量
%%

So, to understand deeply and properly how neural networks perform and get trained, **you do need knowledge of linear algebra and calculus joined together**! But do not worry! All together it is not that scary if you do it step by step accurately with understanding of maths.
%%
深入且恰当地理解神经网络如何运作与训练，你需要线性代数和微积分的知识进行结合！
不必担心，只要按部就班，准确理解数学，整体来看并没那么可怕。
%%
Time to implement this all in the code!
%%
是时候去实现所有的代码了！
%%

<a name='2.4'></a>
### 2.4 - Dataset

First, let's get the dataset you will work on. The following code will create $m=2000$ data points $(x_1, x_2)$ and save them in the `NumPy` array `X` of a shape $(2 \times m)$ (in the columns of the array). The labels ($0$: blue, $1$: red) will be saved in the `NumPy` array `Y` of a shape $(1 \times m)$.
%%
首先，我们来获取您将要处理的数据集。
下面的代码创建了 $m=2000$ 的数据点，然后将它们保存在这个形状为 $(2 \times m)$，类型为 `NumPy` 数组的变量 `X` 中（$2$ 行 $m$ 列，也就是 2 行 2000 列）。
它的标签（0 为蓝色，1 为红色）保存在形状为 $(1 \times m)$，类型为 `NumPy` 数组的变量 `Y` 中（1 行 $m$ 列）。
%%


```python
m = 2000
samples, labels = make_blobs(n_samples=m,
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]),
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1, m))

plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have m = %d training examples!' % (m))
```

> [!result]
>     The shape of X is: (2, 2000)
>     The shape of Y is: (1, 2000)
>     I have m = 2000 training examples!
![C2_W3_Assignment_19_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Assignment_19_1.png) 


<a name='2.5'></a>
### 2.5 - Define Activation Function

<a name='ex01'></a>
#### Exercise 1

Define sigmoid activation function $\sigma\left(z\right) =\frac{1}{1+e^{-z}}$.


```python
def sigmoid(z):
    ### START CODE HERE ### (~ 1 line of code)
    res = 1/(1+np.exp(-z))
    ### END CODE HERE ###

    return res
```


```python
print("sigmoid(-2) = " + str(sigmoid(-2)))
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(3.5) = " + str(sigmoid(3.5)))
```

> [!result]
>     sigmoid(-2) = 0.11920292202211755
>     sigmoid(0) = 0.5
>     sigmoid(3.5) = 0.9706877692486436

<a name='3'></a>
## 3 - Implementation of the Neural Network Model with Two Layers

<a name='3.1'></a>
### 3.1 - Defining the Neural Network Structure

<a name='ex02'></a>
#### Exercise 2

Define three variables:
- `n_x`: the size of the input layer
- `n_h`: the size of the hidden layer (set it equal to 2 for now)
- `n_y`: the size of the output layer

%%
定义三个变量
- `n_x`：输入层的形状。
- `n_h`：隐藏层的形状（暂时将其设为 2）。
- `n_y`：输出层的形状。
%%

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hint</b></font>
</summary>
<p>
<ul>
    Use shapes of X and Y to find n_x and n_y:
    <li>the size of the input layer n_x equals to the size of the input vectors placed in the columns of the array X,</li>
    <li>the outpus for each of the data point will be saved in the columns of the the array Y.</li>
</ul>
</p>


```python
# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (~ 3 lines of code)
    # Size of input layer.
    n_x = X.shape[0]
    # Size of hidden layer.
    n_h = 2
    # Size of output layer.
    n_y = Y.shape[0]
    ### END CODE HERE ###
    return (n_x, n_h, n_y)
```


```python
(n_x, n_h, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
```

> [!result]
>     The size of the input layer is: n_x = 2
>     The size of the hidden layer is: n_h = 2
>     The size of the output layer is: n_y = 1

<a name='3.2'></a>
### 3.2 - Initialize the Model's Parameters

<a name='ex03'></a>
#### Exercise 3

Implement the function `initialize_parameters()`.

**Instructions**:
- Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
- You will initialize the weights matrix with random values. 
    - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
- You will initialize the bias vector as zeros. 
    - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.

%%
- 保证你的参数形状正确。如果需要请参考神经网络计算图。
- 你需要用随机值初始化权重矩阵；使用 `np.random.randn(a,b) * 0.01` 来对形状为 (a,b) 的矩阵进行随机化。
- 你需要将偏置初始化为全零的向量；使用 `np.zeros((a,b))` 来对形状为 (a,b) 的矩阵填充 0。
%%


```python
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    ### START CODE HERE ### (~ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
```


```python
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

> [!result]
>     W1 = [[ 0.01788628  0.0043651 ]
>           [ 0.00096497 -0.01863493]]
>     b1 = [[0.]
>           [0.]]
>     W2 = [[-0.00277388 -0.00354759]]
>     b2 = [[0.]]
<a name='3.3'></a>
### 3.3 - The Loop

<a name='ex04'></a>
#### Exercise 4

Implement `forward_propagation()`.

**Instructions**:
- Look above at the mathematical representation $(7)$ of your classifier (section [2.2](#2.2)):
$$
\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]},\\
A^{[1]} &= \sigma\left(Z^{[1]}\right),\\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]},\\
A^{[2]} &= \sigma\left(Z^{[2]}\right).\\
\end{align}
$$
- The steps you have to implement are:
    1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
    2. Implement Forward Propagation. Compute `Z1` multiplying matrices `W1`, `X` and adding vector `b1`. Then find `A1` using the `sigmoid` activation function. Perform similar computations for `Z2` and `A2`.


%%
你需要实现的步骤包括：
1. 从字典 "parameters" 取出每个参数。
2. 实现前向传播。计算 `Z1`，即矩阵 `W1` 和 `X` 相乘后加上向量 `b1`。然后通过 `sigmoid` 激活函数求得 `A1`。对 `Z2` 和 `A2` 同样进行类似的计算。
%%


```python
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- the sigmoid output of the second activation
    cache -- python dictionary containing Z1, A1, Z2, A2 
    (that simplifies the calculations in the back propagation step)
    """
    # Retrieve each parameter from the dictionary "parameters".
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    # Implement forward propagation to calculate A2.
    ### START CODE HERE ### (~ 4 lines of code)
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    ### END CODE HERE ###

    assert(A2.shape == (n_y, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache
```


```python
A2, cache = forward_propagation(X, parameters)

print(A2)
```

> [!result]
>     [[0.49920157 0.49922234 0.49921223 ... 0.49921215 0.49921043 0.49920665]]


Remember, that your weights were just initialized with some random values, so the model has not been trained yet.

%%
请记得，你的权重只是一些之前初始化生成的随机值，所以模型尚未开始训练。
%%

<a name='ex05'></a>
#### Exercise 5

Define a cost function $(8)$ which will be used to train the model:

$$\mathcal{L}\left(W, b\right)  = \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small - y^{(i)}\log\left(a^{(i)}\right) - (1-y^{(i)})\log\left(1- a^{(i)}\right)  \large  \right) \small.$$


```python
def compute_cost(A2, Y):
    """
    Computes the cost function as a log loss

    Arguments:
    A2 -- The output of the neural network of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- log loss

    """
    # Number of examples.
    m = Y.shape[1]

    ### START CODE HERE ### (~ 2 lines of code)
    logloss = - Y * np.log(A2) - (1 - Y) * np.log(1 - A2)
    cost = 1 / m * np.sum(logloss)
    ### END CODE HERE ###

    assert(isinstance(cost, float))

    return cost
```


```python
print("cost = " + str(compute_cost(A2, Y)))
```

> [!result]
>     cost = 0.6931477703826823


Calculate partial derivatives as shown in $(15)$:

%%
计算式子 $(15)$ 列出的偏导数
%%
$$
\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1},\\
\frac{\partial \mathcal{L} }{ \partial W^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1}.\\
\end{align}
$$

```python
def backward_propagation(parameters, cache, X, Y):
    """
    Implements the backward propagation, calculating gradients

    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- python dictionary containing Z1, A1, Z2, A2
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)

    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate partial derivatives denoted as dW1, db1, dW2, db2 for simplicity. 
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

grads = backward_propagation(parameters, cache, X, Y)

print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))
```

>[!result]
>     dW1 = [[-1.49856632e-05  1.67791519e-05]
>      [-2.12394543e-05  2.43895135e-05]]
>     db1 = [[5.11207671e-07]
>      [7.06236219e-07]]
>     dW2 = [[-0.00032641 -0.0002606 ]]
>     db2 = [[-0.00078732]]


<a name='ex06'></a>
#### Exercise 6

Implement `update_parameters()`.

**Instructions**:
- Update parameters as shown in $(9)$ (section [2.3](#2.3)):
$$
\begin{align}
W^{[1]} &= W^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[1]} },\\
b^{[1]} &= b^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[1]} },\\
W^{[2]} &= W^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[2]} },\\
b^{[2]} &= b^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[2]} }.\\
\end{align}
$$
- The steps you have to implement are:
    1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
    2. Retrieve each derivative from the dictionary "grads" (which is the output of `backward_propagation()`) by using `grads[".."]`.
    3. Update parameters.

%%
实现步骤：
1. 从字典 "parameters" 取出每个参数。
2. 从字典 “grads” 中获取导数。
3. 根据上面的公式更新参数。
%%


```python
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule

    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients
    learning_rate -- learning rate for gradient descent

    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads".
    ### START CODE HERE ### (~ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ### END CODE HERE ###

    # Update rule for each parameter.
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
```


```python
parameters_updated = update_parameters(parameters, grads)

print("W1 updated = " + str(parameters_updated["W1"]))
print("b1 updated = " + str(parameters_updated["b1"]))
print("W2 updated = " + str(parameters_updated["W2"]))
print("b2 updated = " + str(parameters_updated["b2"]))
```

>[!result]
>     W1 updated = [[ 0.01790427  0.00434496]
>                   [ 0.00099046 -0.01866419]]
>     b1 updated = [[-6.13449205e-07]
>                   [-8.47483463e-07]]
>     W2 updated = [[-0.00238219 -0.00323487]]
>     b2 updated = [[0.00094478]]

<a name='3.4'></a>
### 3.4 - Integrate parts 3.1, 3.2 and 3.3 in nn_model()

<a name='ex07'></a>
#### Exercise 7

Build your neural network model in `nn_model()`.

**Instructions**: The neural network model has to use the previous functions in the right order.

%%
神经网络模型的构建必须按正确顺序调用前面的函数。
%%


```python
# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters.
    ### START CODE HERE ### (~ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###

    # Loop.
    for i in range(0, num_iterations):

        ### START CODE HERE ### (~ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(A2, Y)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```


```python
parameters = nn_model(X, Y, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]
```

>[!result]
>     Cost after iteration 0: 0.693148
>     Cost after iteration 1: 0.693147
>     Cost after iteration 2: 0.693147
>     Cost after iteration 3: 0.693147
>     Cost after iteration 4: 0.693147
>     Cost after iteration 5: 0.693147
>     Cost after iteration 6: 0.693147
>     Cost after iteration 7: 0.693147
>     Cost after iteration 8: 0.693147
>     Cost after iteration 9: 0.693147
>     Cost after iteration 10: 0.693147
>     Cost after iteration 11: 0.693147
>     Cost after iteration 12: 0.693147
>     Cost after iteration 13: 0.693147
>     Cost after iteration 14: 0.693147
>     Cost after iteration 15: 0.693147
>     Cost after iteration 16: 0.693147
>     Cost after iteration 17: 0.693147
>     Cost after iteration 18: 0.693147
>     Cost after iteration 19: 0.693147
>     Cost after iteration 20: 0.693147
>     ...
>     Cost after iteration 2988: 0.187304
>     Cost after iteration 2989: 0.187302
>     Cost after iteration 2990: 0.187301
>     Cost after iteration 2991: 0.187288
>     Cost after iteration 2992: 0.187280
>     Cost after iteration 2993: 0.187286
>     Cost after iteration 2994: 0.187299
>     Cost after iteration 2995: 0.187344
>     Cost after iteration 2996: 0.187401
>     Cost after iteration 2997: 0.187530
>     Cost after iteration 2998: 0.187680
>     Cost after iteration 2999: 0.188011
>     W1 = [[ 1.97400732 -1.68735181]
>           [ 2.21244075 -1.96207282]]
>     b1 = [[-4.83489386]
>           [ 6.31854573]]
>     W2 = [[-7.25665649  7.11965784]]
>     b2 = [[-3.44475406]]


The final model parameters can be used to find the boundary line and for making predictions.

%%
模型的最终参数可以用于预测，同时它可以确定边界线的位置。
%%

<a name='ex08'></a>
#### Exercise 8

Computes probabilities using forward propagation, and make classification to 0/1 using 0.5 as the threshold.

%%
利用前向传播计算预测值，然后使用0.5为阈值进行 0/1 分类。
%%


```python
# GRADED FUNCTION: predict

def predict(X, parameters):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (blue: 0 / red: 1)
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    A2, _ = forward_propagation(X, parameters)
    predictions = A2 >= 0.5
    ### END CODE HERE ###

    return predictions
```


```python
X_pred = np.array([[2, 8, 2, 8], [2, 8, 8, 2]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")
```

>[!result]
>     Coordinates (in the columns):
>     [[2 8 2 8]
>      [2 8 8 2]]
>     Predictions:
>     [[ True  True False False]]


Let's visualize the boundary line. Do not worry if you don't understand the function `plot_decision_boundary` line by line - it simply makes prediction for some points on the plane and plots them as a contour plot (just two colors - blue and red).

%%
让我们来看看这个边界线。
如果你不理解函数`plot_decision_boundary`，请不要担心，它只是对平面上的一些点进行预测，并将它们绘制为等高线图（只有两种颜色-蓝色和红色）。
%%


```python
def plot_decision_boundary(predict, parameters, X, Y):
    # Define bounds of the domain.
    min1, max1 = X[0, :].min()-1, X[0, :].max()+1
    min2, max2 = X[1, :].min()-1, X[1, :].max()+1
    # Define the x and y scale.
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # Create all of the lines and rows of the grid.
    xx, yy = np.meshgrid(x1grid, x2grid)
    # Flatten each grid to a vector.
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((1, len(r1))), r2.reshape((1, len(r2)))
    # Vertical stack vectors to create x1,x2 input for the model.
    grid = np.vstack((r1,r2))
    # Make predictions for the grid.
    predictions = predict(grid, parameters)
    # Reshape the predictions back into a grid.
    zz = predictions.reshape(xx.shape)
    # Plot the grid of x, y and z values as a surface.
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral.reversed())
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

# Plot the decision boundary.
plot_decision_boundary(predict, parameters, X, Y)
plt.title("Decision Boundary for hidden layer size " + str(n_h))
```

> [!result]
>     Text(0.5, 1.0, 'Decision Boundary for hidden layer size 2')
![C2_W3_Assignment_64_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Assignment_64_1.png)



That's great, you can see that more complicated classification problems can be solved with two layer neural network!

%%
太棒了，你可以看到，更复杂的分类问题也可以通过两层神经网络来解决！
%%

<a name='4'></a>
## 4 - Optional: Other Dataset

Build a slightly different dataset:

%%
构建一个略有差异的数据集：
%%


```python
n_samples = 2000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0)] = 0
labels[(labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 1
X_2 = np.transpose(samples)
Y_2 = labels.reshape((1,n_samples))

plt.scatter(X_2[0, :], X_2[1, :], c=Y_2, cmap=colors.ListedColormap(['blue', 'red']));
```

> [!result]
![C2_W3_Assignment_68_0.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Assignment_68_0.png)



Notice that when building your neural network, a number of the nodes in the hidden layer could be taken as a parameter. Try to change this parameter and investigate the results:

%%
请注意，在构建神经网络时，隐藏层中的一些节点可以作为参数。尝试更改此参数并研究结果：
%%


```python
parameters_2 = nn_model(X_2, Y_2, n_h=1, num_iterations=3000, learning_rate=1.2, print_cost=False)

# This function will call predict function 
plot_decision_boundary(predict, parameters_2, X_2, Y_2)
plt.title("Decision Boundary")
```

> [!result]
>     Text(0.5, 1.0, 'Decision Boundary')
![C2_W3_Assignment_70_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Assignment_70_1.png)


```python
parameters_2 = nn_model(X_2, Y_2, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=False)

# This function will call predict function
plot_decision_boundary(predict, parameters_2, X_2, Y_2)
plt.title("Decision Boundary")
```

> [!result]
>     Text(0.5, 1.0, 'Decision Boundary')
![C2_W3_Assignment_71_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Assignment_71_1.png)



```python
parameters_2 = nn_model(X_2, Y_2, n_h=15, num_iterations=3000, learning_rate=1.2, print_cost=False)

# This function will call predict function
plot_decision_boundary(predict, parameters_2, X_2, Y_2)
plt.title("Decision Boundary")
```

> [!result]
>     Text(0.5, 1.0, 'Decision Boundary')
![C2_W3_Assignment_72_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Assignment_72_1.png)


You can see that there are some misclassified points - real-world datasets are usually linearly inseparable, and there will be a small percentage of errors. More than that, you do not want to build a model that fits too closely, almost exactly to a particular set of data - it may fail to predict future observations. This problem is known as **overfitting**.

%%
你可以看到一些错误的分类点。
现实世界的数据集通常线性不可分，并且存在一定比例的错误。
更重要的是，你不能构建一个过于拟合的模型，几乎完全符合特定数据集是可能无法预测未来其他的观察结果。
这个问题被称为**过拟合**。
%%

Congrats on finishing this programming assignment!
