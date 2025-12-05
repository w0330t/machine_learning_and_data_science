---
aliases: [Regression with Perceptron]
tags: []
created: 2025-12-03, 09:02:39
modified: 2025-12-05, 18:25:41
---

# Regression with Perceptron

In the week 2 assignment, you implemented the gradient descent method to build a linear regression model, predicting sales given a TV marketing budget. In this lab, you will construct a neural network corresponding to the same simple linear regression model. Then you will train the network, implementing the gradient descent method. After that you will increase the complexity of the neural network to build a multiple linear regression model, predicting house prices based on their size and quality.

%%
在第二周的作业里，你使用一个梯度下降的方法构建了一个线性模型，用电视营销的成本预测了销量。
在这个实验中，你将使用和上一次一样的简单回归模型构建一个对应的神经网络。
然后，你将采用梯度下降法来训练网络。
然后，您将增加神经网络的复杂性，以构建一个多元线性回归模型。
基于它们的尺寸和质量预测房屋的价格。
%%

*Note*: The same models were discussed in Course 1 "Linear Algebra" week 3 assignment, but model training with backward propagation was omitted.

%%
*注意：*
同样的模型在“线性代数”章节第三周的作业中已经讨论过，但省略了采用反向传播的模型训练过程。
%%

# Table of Contents

- [ 1 - Simple Linear Regression](#1)
  - [ 1.1 - Simple Linear Regression Model](#1.1)
  - [ 1.2 - Neural Network Model with a Single Perceptron and One Input Node](#1.2)
  - [ 1.3 - Dataset](#1.3)
- [ 2 - Implementation of the Neural Network Model for Linear Regression](#2)
  - [ 2.1 - Defining the Neural Network Structure](#2.1)
  - [ 2.2 - Initialize the Model's Parameters](#2.2)
  - [ 2.3 - The Loop](#2.3)
  - [ 2.4 - Integrate parts 2.1, 2.2 and 2.3 in nn_model() and make predictions](#2.4)
- [ 3 - Multiple Linear Regression](#3)
  - [ 3.1 - Multipe Linear Regression Model](#3.1)
  - [ 3.2 - Neural Network Model with a Single Perceptron and Two Input Nodes](#3.2)
  - [ 3.3 - Dataset](#3.3)
  - [ 3.4 - Performance of the Neural Network Model for Multiple Linear Regression](#3.4)

## Packages

Let's first import all the required packages.


```python
import numpy as np
import matplotlib.pyplot as plt
# A library for data manipulation and analysis.
import pandas as pd

# Output of plotting commands is displayed inline within the Jupyter notebook.
%matplotlib inline 

# Set a seed so that the results are consistent.
np.random.seed(3) 
```

<a name='1'></a>
## 1 - Simple Linear Regression

<a name='1.1'></a>
### 1.1 - Simple Linear Regression Model

You can describe a simple linear regression model as

%%
你可以这样描述一个简单的线性回归模型：
%%

$$\hat{y} = wx + b,\tag{1}$$

where $\hat{y}$ is a prediction of dependent variable $y$ based on independent variable $x$ using a line equation with the slope $w$ and intercept $b$. 

%%
其中， $\hat{y}$ 是利用斜率 $w$, 截距 $b$ 组成的线性方程，同时 $\hat{y}$ 也是基于自变量 $x$ 对因变量 $y$ 的预测值。
%%

Given a set of training data points $(x_1, y_1)$, ..., $(x_m, y_m)$, you will find the "best" fitting line - such parameters $w$ and $b$ that the differences between original values $y_i$ and predicted values $\hat{y}_i = wx_i + b$ are minimum.

%%
给定一套训练数据点 $(x_1, y_1)$, ..., $(x_m, y_m)$，你可以找到这个最佳的拟合线——寻找参数 $w$ 和 $b$，使原始值 $y_i$ 与预测值 $\hat{y_i} = wx_i + b$ 之间的差异最小。
%%
<a name='1.2'></a>
### 1.2 - Neural Network Model with a Single Perceptron and One Input Node

The simplest neural network model that describes the above problem can be realized by using one **perceptron**. The **input** and **output** layers will have one **node** each ($x$ for input and $\hat{y} = z$ for output):

描述上述问题的最简神经网络模型可通过*单个***感知器**实现。
它的输入层和输出层会各有一个**节点**（$x$ 为输入，而 $\hat{y} = z$ 为输出）：

![nn_model_linear_regression_simple.png|600](https://obsidian-image.wwtt.xyz/2025/12/nn_model_linear_regression_simple.png)


**Weight** ($w$) and **bias** ($b$) are the parameters that will get updated when you **train** the model. They are initialized to some random values or set to 0 and updated as the training progresses.

%%
参数权重（$w$）和偏置（$b$）训练模型的时候会不断的更新。
它们被初始化为一些随机值或设为 0，并随着训练的进行而更新。
%%

For each training example $x^{(i)}$, the prediction $\hat{y}^{(i)}$ can be calculated as:

%%
对于每一个训练样本 $x^{(i)}$，它的预测结果 $\hat{y}^{(i)}$ 可以这么计算：
%%

$$\begin{align}
z^{(i)} &=  w x^{(i)} + b,\\
\hat{y}^{(i)} &= z^{(i)},
\tag{2}\end{align}$$

where $i = 1, \dots, m$.

%%
其中 $i = 1, \dots, m$.
%%

You can organise all training examples as a vector $X$ of size ($1 \times m$) and perform scalar multiplication of $X$ ($1 \times m$) by a scalar $w$, adding $b$, which will be broadcasted to a vector of size ($1 \times m$):

%%
你可以将所有的训练样本组织成一个尺寸为 ($1 \times m$) 的向量 $X$，然后对向量 $X$ 和标量 $w$ 做标量乘法，之后加上 $b$，后者将被广播成为一个维度为 ($1 \times m$) 的向量。
%%

$$
\begin{align}
Z &=  w X + b,\\
\hat{Y} &= Z,
\tag{3}\end{align}
$$
This set of calculations is called **forward propagation**.

%%
这组计算被称为**前向传播**。
%%

For each training example you can measure the difference between original values $y^{(i)}$ and predicted values $\hat{y}^{(i)}$ with the **loss function** $L\left(w, b\right)  = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$. Division by $2$ is taken just for scaling purposes, you will see the reason below, calculating partial derivatives. To compare the resulting vector of the predictions $\hat{Y}$ ($1 \times m$) with the vector $Y$ of original values $y^{(i)}$, you can take an average of the loss function values for each of the training examples:

%%
每个训练样本你可以通过**损失函数** $L\left(w, b\right)  = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$ 测量原始值 $y^{(i)}$ 和预测值 $\hat{y}^{(i)}$ 的差距。
除以 2 的目的仅用于缩放，你可以在下面计算偏导数的时候看到这么做的原因。
通过预测的结果的向量 $\hat{Y}$ ($1 \times m$) 和原始值 $y^{(i)}$ 构成的向量 $Y$ 做对比，就可以计算每个训练样本损失函数值的平均值：
%%

$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.\tag{4}$$

This function is called the sum of squares **cost function**. The aim is to optimize the cost function during the training, which will minimize the differences between original values $y^{(i)}$ and predicted values $\hat{y}^{(i)}$.

%%
此函数被称为平方和**成本函数**。
它的目标是在训练的时候优化成本函数，这将会最小化原始值 $y^{(i)}$ 和预测值 $\hat{y}^{(i)}$ 之间的差距。
%%

When your weights were just initialized with some random values, and no training was done yet, you can't expect good results. You need to calculate the adjustments for the weight and bias, minimizing the cost function. This process is called **backward propagation**. 

%%
当你的权重初始化为随机值，且训练没有完成的时候，你不能期望会有好的结果。
你需要通过计算调整权重和偏置，最小化成本函数。
这个步骤被称为**反向传播**。
%%

According to the gradient descent algorithm, you can calculate partial derivatives as:

%%
根据梯度下降的算法，你可以这样计算偏导数：
%%

$$\begin{align}
\frac{\partial \mathcal{L} }{ \partial w } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)},\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right).
\tag{5}\end{align}
$$
You can see how the additional division by $2$ in the equation $(4)$ helped to simplify the results of the partial derivatives. Then update the parameters iteratively using the expressions

%%
你可以看到方程（4）中额外除以 2 是如何帮助简化偏导数结果的。
然后在迭代中使用下面的表达式更新参数。
%%
$$\begin{align}
w &= w - \alpha \frac{\partial \mathcal{L} }{ \partial w },\\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b },
\tag{6}\end{align}$$

where $\alpha$ is the learning rate. Then repeat the process until the cost function stops decreasing.

%%
其中 $\alpha$ 是学习率。
然后重复该过程，直至代价函数停止下降。
%%

The general **methodology** to build a neural network is to:
1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
2. Initialize the model's parameters
3. Loop:
    - Implement forward propagation (calculate the perceptron output),
    - Implement backward propagation (to get the required corrections for the parameters),
    - Update parameters.
4. Make predictions.

%%
构建神经网络的通用方法是：
1. 定义神经网络的结构（输入单元，隐藏单元等等）。
2. 初始化模型的参数。
3. 循环：
	- 实现前向传播（计算感知器的输出）。
	- 实现反向传播（获取所需的参数修正）。
	- 更新参数。
4. 预测。
%%

You often build helper functions to compute steps 1-3 and then merge them into one function `nn_model()`. Once you've built `nn_model()` and learnt the right parameters, you can make predictions on new data.

%%
一般来说我们会先构建一些辅助函数来完成 1~3 步的计算，然后将它们合并到一个函数 `nn_model()` 中。
一旦你构建好了 `nn_model()`，并且模型学习到了正确的参数，你就可以预测新的数据。
%%
<a name='1.3'></a>
### 1.3 - Dataset

Load the [Kaggle dataset](https://www.kaggle.com/code/devzohaib/simple-linear-regression/notebook), saved in a file `data/tvmarketing.csv`. It has two fields: TV marketing expenses (`TV`) and sales amount (`Sales`).

```python
path = "data/tvmarketing.csv"

adv = pd.read_csv(path)
```

Print some part of the dataset.


```python
adv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230.1</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.5</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.2</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151.5</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>180.8</td>
      <td>12.9</td>
    </tr>
  </tbody>
</table>
</div>



And plot it:


```python
adv.plot(x='TV', y='Sales', kind='scatter', c='black')
```



> [!result]
>     <Axes: xlabel='TV', ylabel='Sales'>
![C2_W3_Lab_1_Regression_with_Perceptron_14_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Lab_1_Regression_with_Perceptron_14_1.png)


The fields `TV` and `Sales` have different units. Remember that in the week 2 assignment to make gradient descent algorithm efficient, you needed to normalize each of them: subtract the mean value of the array from each of the elements in the array and divide them by the standard deviation.

%%
字段 `TV` 和 `Sales` 的量纲不同。
记得第二周作业，要使梯度下降高效的运行，你需要标准化它们：每个数组元素减去它们的均值然后除以它们的标准差。
%%

Column-wise normalization of the dataset can be done for all of the fields at once and is implemented in the following code:

%%
按列对数据集进行标准化可以一次性完成所有字段的操作，具体实现如下：
%%

```python
adv_norm = (adv - adv.mean())/np.std(adv)
```

> [!result]
>     /home/w0330t/Projects/ machine_learning_and_data_science/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:4062: FutureWarning: The behavior of DataFrame.std with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
>     return std(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)


Plotting the data, you can see that it looks similar after normalization, but the values on the axes have changed:

%%
绘制这个数据，你可以看到它看上去和标准化之前非常相似，但坐标轴上的数值已经发生了变化：
%%

```python
adv_norm.plot(x='TV', y='Sales', kind='scatter', c='black')
```

> [!result]
>     <Axes: xlabel='TV', ylabel='Sales'>
![C2_W3_Lab_1_Regression_with_Perceptron_18_1.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Lab_1_Regression_with_Perceptron_18_1.png)


Save the fields into variables `X_norm` and `Y_norm` and reshape them to row vectors:

%%
将字段保存到变量 `X_norm` 和 `Y_norm` 中，然后将它们重塑为行向量：
%%

```python
X_norm = adv_norm['TV']
Y_norm = adv_norm['Sales']

X_norm = np.array(X_norm).reshape((1, len(X_norm)))
Y_norm = np.array(Y_norm).reshape((1, len(Y_norm)))

print ('The shape of X_norm: ' + str(X_norm.shape))
print ('The shape of Y_norm: ' + str(Y_norm.shape))
print ('I have m = %d training examples!' % (X_norm.shape[1]))
```

> [!result]
>     The shape of X_norm: (1, 200)
>     The shape of Y_norm: (1, 200)
>     I have m = 200 training examples!

<a name='2'></a>
## 2 - Implementation of the Neural Network Model for Linear Regression

Setup the neural network in a way which will allow to extend this simple case of a model with a single perceptron and one input node to more complicated structures later.

%%
构建神经网络的时候，应当使其能够从一个感知器和一个输入节点的简单模型扩展到未来更复杂的结构。
%%
<a name='2.1'></a>
### 2.1 - Defining the Neural Network Structure

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

(n_x, n_y) = layer_sizes(X_norm, Y_norm)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))
```

> [!result]
>     The size of the input layer is: n_x = 1
>     The size of the output layer is: n_y = 1

<a name='2.2'></a>
### 2.2 - Initialize the Model's Parameters

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

> [!result]
>     W = [[0.01788628]]
>     b = [[0.]]

<a name='2.3'></a>
### 2.3 - The Loop

Implement `forward_propagation()` following the equation $(3)$ in the section [1.2](#1.2):

$$\begin{align}
Z &=  w X + b\\
\hat{Y} &= Z
\end{align}$$


```python
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    Y_hat -- The output
    """
    W = parameters["W"]
    b = parameters["b"]

    # Forward Propagation to calculate Z.
    Z = np.matmul(W, X) + b
    Y_hat = Z

    return Y_hat

Y_hat = forward_propagation(X_norm, parameters)

print("Some elements of output vector Y_hat:", Y_hat[0, 0:5])
```

> [!result]
>     Some elements of output vector Y_hat: [ 0.01734705 -0.02141661 -0.02711838  0.00093098  0.00705046]


Your weights were just initialized with some random values, so the model has not been trained yet. 

%%
你的权重已经初始化为一些随机值了，所以模型尚未开始训练。
%%

Define a cost function $(4)$ which will be used to train the model:

%%
定义成本函数 $(4)$，这将用于训练这个模型：
%%
$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$$


```python
def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y_hat.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

print("cost = " + str(compute_cost(Y_hat, Y_norm)))
```

> [!result]
>     cost = 0.48616887080159704


Calculate partial derivatives as shown in $(5)$:

%%
如同 $(5)$ 所示， 计算偏导数。
%%

$$
\begin{align}
\frac{\partial \mathcal{L} }{ \partial w } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)}\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)
\end{align}
$$
```python
def backward_propagation(Y_hat, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    Y_hat -- the output of the neural network of shape (n_y, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity. 
    dZ = Y_hat - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(Y_hat, X_norm, Y_norm)

print("dW = " + str(grads["dW"]))
print("db = " + str(grads["db"]))
```

> [!result]
>     dW = [[-0.76433814]]
>     db = [[1.687539e-16]]

Update parameters as shown in $(6)$:
%%
如同 $(6)$ 所示，更新参数。
%%
$$\begin{align}
w &= w - \alpha \frac{\partial \mathcal{L} }{ \partial w },\\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b }.
\end{align}$$


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
>     W updated = [[0.93509205]]
>     b updated = [[-2.0250468e-16]]

<a name='2.4'></a>
### 2.4 - Integrate parts 2.1, 2.2 and 2.3 in nn_model() and make predictions

Build your neural network model in `nn_model()`.

%%
在函数 `nn_model()` 构建你的神经网络模型。
%%

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

        # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat".
        Y_hat = forward_propagation(X, parameters)
        # print(Y_hat)

        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat, Y)

        # Backpropagation. Inputs: "Y_hat, X, Y". Outputs: "grads".
        grads = backward_propagation(Y_hat, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```


```python
parameters_simple = nn_model(X_norm, Y_norm, num_iterations=30, learning_rate=1.2, print_cost=True)
print("W = " + str(parameters_simple["W"]))
print("b = " + str(parameters_simple["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]
```

> [!result]
>     Cost after iteration 0: 0.496595
>     Cost after iteration 1: 0.206164
>     Cost after iteration 2: 0.194547
>     Cost after iteration 3: 0.194082
>     Cost after iteration 4: 0.194063
>     Cost after iteration 5: 0.194063
>     Cost after iteration 6: 0.194062
>     Cost after iteration 7: 0.194062
>     Cost after iteration 8: 0.194062
>     Cost after iteration 9: 0.194062
>     Cost after iteration 10: 0.194062
>     Cost after iteration 11: 0.194062
>     Cost after iteration 12: 0.194062
>     Cost after iteration 13: 0.194062
>     Cost after iteration 14: 0.194062
>     Cost after iteration 15: 0.194062
>     Cost after iteration 16: 0.194062
>     Cost after iteration 17: 0.194062
>     Cost after iteration 18: 0.194062
>     Cost after iteration 19: 0.194062
>     Cost after iteration 20: 0.194062
>     Cost after iteration 21: 0.194062
>     Cost after iteration 22: 0.194062
>     Cost after iteration 23: 0.194062
>     Cost after iteration 24: 0.194062
>     Cost after iteration 25: 0.194062
>     Cost after iteration 26: 0.194062
>     Cost after iteration 27: 0.194062
>     Cost after iteration 28: 0.194062
>     Cost after iteration 29: 0.194062
>     W = [[0.78222442]]
>     b = [[-3.19744231e-16]]


You can see that after a few iterations the cost function does not change anymore (the model converges).

%%
你可以看到在几次迭代后成本函数就不再变化了（模型收敛了）。
%%

*Note*: This is a very simple model. In reality the models do not converge that quickly.

%%
这是一个非常简单的模型。
现实中的模型不可能收敛得这么快。
%%

The final model parameters can be used for making predictions, but don't forget about normalization and denormalization.

%%
最终模型的参数可以用于预测，但是别忘记标准化和反标准化的步骤
%%

```python
def predict(X, Y, parameters, X_pred):

    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]

    # Use the same mean and standard deviation of the original training array X.
    if isinstance(X, pd.Series):
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_pred_norm = ((X_pred - X_mean)/X_std).reshape((1, len(X_pred)))
    else:
        X_mean = np.array(X.mean()).reshape((len(X.axes[1]),1))
        X_std = np.array(np.std(X)).reshape((len(X.axes[1]),1))
        X_pred_norm = ((X_pred - X_mean)/X_std)
    # Make predictions.
    Y_pred_norm = np.matmul(W, X_pred_norm) + b
    # Use the same mean and standard deviation of the original training array Y.
    Y_pred = Y_pred_norm * np.std(Y) + np.mean(Y)

    return Y_pred[0]

X_pred = np.array([50, 120, 280])
Y_pred = predict(adv["TV"], adv["Sales"], parameters_simple, X_pred)
print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales:\n{Y_pred}")
```

> [!result]
>     TV marketing expenses:
>     [ 50 120 280]
>     Predictions of sales:
>     [ 9.40942557 12.7369904  20.34285287]


Let's plot the linear regression line and some predictions. The regression line is red and the predicted points are blue.

%%
绘制线性回归线，然后进行预测。
回归的线是红色，预测的点是蓝色。
%%
```python
fig, ax = plt.subplots()
plt.scatter(adv["TV"], adv["Sales"], color="black")

plt.xlabel("$x$")
plt.ylabel("$y$")
    
X_line = np.arange(np.min(adv["TV"]),np.max(adv["TV"])*1.1, 0.1)
Y_line = predict(adv["TV"], adv["Sales"], parameters_simple, X_line)
ax.plot(X_line, Y_line, "r")
ax.plot(X_pred, Y_pred, "bo")
plt.plot()
plt.show()
```

> [!result]
> ![C2_W3_Lab_1_Regression_with_Perceptron_44_0.png](https://obsidian-image.wwtt.xyz/2025/12/C2_W3_Lab_1_Regression_with_Perceptron_44_0.png)


Now let's increase the number of the input nodes to build a multiple linear regression model.
%%
现在让我们增加数据的输入节点，用于构建多元线性回归模型。
%%

<a name='3'></a>
## 3 - Multiple Linear Regression

<a name='3.1'></a>
### 3.1 - Multipe Linear Regression Model

You can write a multiple linear regression model with two independent variables $x_1$, $x_2$ as

%%
你可以写出含有两个自变量 $x_1$, $x_2$ 的多元线性回归模型：
%%

$$\hat{y} = w_1x_1 + w_2x_2 + b = Wx + b,\tag{7}$$

where $Wx$ is the dot product of the input vector $x = \begin{bmatrix} x_1 & x_2\end{bmatrix}$ and the parameters vector $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$, scalar parameter $b$ is the intercept. The goal of the training process is to find the "best" parameters $w_1$, $w_2$ and $b$ such that the differences between original values $y_i$ and predicted values $\hat{y}_i$ are minimum for the given training examples.

%%
其中 $Wx$ 是输入向量 $x = \begin{bmatrix} x_1 & x_2\end{bmatrix}$ 和参数向量 $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 的点积，标量参数 $b$ 是截距。
目标是训练过程中找到最好的参数 $w_1$, $w_2$ and $b$，也就是在使得在给定训练样本上，原始值 $y_i$ 和预测值 $\hat{y}_i$ 之间的差异最小化。
%%
<a name='3.2'></a>
### 3.2 - Neural Network Model with a Single Perceptron and Two Input Nodes

To describe the multiple regression problem, you can still use a model with one perceptron, but this time you need two input nodes, as shown in the following scheme:

%%
在描述这个多元回归问题的时候，你仍然可以使用一个单感知器模型，但是这次你需要两个输入节点，如下图所示：
%%


![nn_model_linear_regression_multiple.png|600](https://obsidian-image.wwtt.xyz/2025/12/nn_model_linear_regression_multiple.png)


The perceptron output calculation for every training example $x^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)}\end{bmatrix}$ can be written with dot product:
%%
对于每个训练样本 $x^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)}\end{bmatrix}$，感知器的输出计算可以通过点积来表示：
%%
$$z^{(i)} = w_1x_1^{(i)} + w_2x_2^{(i)} + b = Wx^{(i)} + b,\tag{8}$$

where weights are in the vector $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ and bias $b$ is a scalar. The output layer will have the same single node $\hat{y} = z$.
%%
其中 $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 为权重向量，而 $b$ 是偏置是一个标量。
输出层同样是单节点 $\hat{y} = z$。
%%

Organise all training examples in a matrix $X$ of a shape ($2 \times m$), putting $x_1^{(i)}$ and $x_2^{(i)}$ into columns. Then matrix multiplication of $W$ ($1 \times 2$) and $X$ ($2 \times m$) will give a ($1 \times m$) vector

%%
将所有训练样本组成形状为($2 \times m$)矩阵 $X$，其中 $x_1^{(i)}$ 和 $x_2^{(i)}$ 分别为这个矩阵的两列。
那么，权重 $W$ ($1 \times 2$) 与特征 $X$ ($2 \times m$) 做矩阵乘法，将得到一个 ($1 \times m$) 向量。
%%
$$WX = 
\begin{bmatrix} w_1 & w_2\end{bmatrix} 
\begin{bmatrix} 
x_1^{(1)} & x_1^{(2)} & \dots & x_1^{(m)} \\ 
x_2^{(1)} & x_2^{(2)} & \dots & x_2^{(m)} \\ \end{bmatrix}
=\begin{bmatrix} 
w_1x_1^{(1)} + w_2x_2^{(1)} & 
w_1x_1^{(2)} + w_2x_2^{(2)} & \dots & 
w_1x_1^{(m)} + w_2x_2^{(m)}\end{bmatrix}.$$

And the model can be written as

$$\begin{align}
Z &=  W X + b,\\
\hat{Y} &= Z,
\tag{9}\end{align}
$$
where $b$ is broadcasted to the vector of size ($1 \times m$). These are the calculations to perform in the forward propagation step. Cost function will remain the same (see equation $(4)$ in the section [1.2](#1.2)):

%%
其中 b 会被广播后得到形状为 $1 \times m$ 的向量。
然后执行前向传播的计算步骤。
成本函数将保持不变。
%%
$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

To implement the gradient descent algorithm, you can calculate cost function partial derivatives as:

%%
梯度下降的算法实现，你可以通过下面的式子计算成本函数的偏导数：
%%
$$\begin{align}
\frac{\partial \mathcal{L} }{ \partial w_1 } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x_1^{(i)},\\
\frac{\partial \mathcal{L} }{ \partial w_2 } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x_2^{(i)},\tag{10}\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right).
\end{align}$$

After performing the forward propagation as shown in $(9)$, the variable $\hat{Y}$ will contain the predictions in the array of size ($1 \times m$). The original values $y^{(i)}$ will be kept in the array $Y$ of the same size. Thus, $\left(\hat{Y} - Y\right)$ will be a ($1 \times m$) array containing differences $\left(\hat{y}^{(i)} - y^{(i)}\right)$. Matrix $X$ of size ($2 \times m$) has all $x_1^{(i)}$ values in the first row and $x_2^{(i)}$ in the second row. Thus, the sums in the first two equations of $(10)$ can be calculated as matrix multiplication of $\left(\hat{Y} - Y\right)$ of a shape ($1 \times m$) and $X^T$ of a shape ($m \times 2$), resulting in the ($1 \times 2$) array:

%%
如式子 $(9)$ 一般完成前向传播之后，将生成包含预测的结果的变量 $\hat{Y}$，它的形状是 $1 \times m$。
当然原始数据 $y^{(i)}$ 的形状也是相同的。
因此， $\left(\hat{Y} - Y\right)$ 将是一个包含原始值和预测值的差的 $\left(\hat{y}^{(i)} - y^{(i)}\right)$ 的 $1 \times m$ 数组。
所有特征都在形状为 $2 \times m$ 的矩阵 $X$ 中，其中 $x_1^{(i)}$ 在第一列，$x_2^{(i)}$ 在第二列。
由此，式子 $(10)$ 中前面两个方程的求和项，可通过对形状为 $1 \times m$ 的 $(\hat{Y} - Y)$ 与形状为 $m \times 2$ 的 $X^T$ 进行矩阵乘法运算来计算，其结果为一个 $1 \times 2$ 数组：
%%

$$\frac{\partial \mathcal{L} }{ \partial W } = 
\begin{bmatrix} \frac{\partial \mathcal{L} }{ \partial w_1 } & 
\frac{\partial \mathcal{L} }{ \partial w_2 }\end{bmatrix} = \frac{1}{m}\left(\hat{Y} - Y\right)X^T.\tag{11}$$

Similarly for $\frac{\partial \mathcal{L} }{ \partial b }$:

%%
$\frac{\partial \mathcal{L} }{ \partial b }$ 也是类似：
%%
$$\frac{\partial \mathcal{L} }{ \partial b } = \frac{1}{m}\left(\hat{Y} - Y\right)\mathbf{1}.\tag{12}$$

where $\mathbf{1}$ is just a ($m \times 1$) vector of ones.

%%其中 $1$ 是一个形状为 $m \times 1$ 的全一向量。%%

See how linear algebra and calculus work together to make calculations so nice and tidy! You can now update the parameters using matrix form of $W$:

%%
看看线性代数和微积分在一起计算的时候是多么的美好和整洁！
你现在可以使用矩阵更新参数 $W$ 了：
%%
$$
\begin{align}
W &= W - \alpha \frac{\partial \mathcal{L} }{ \partial W },\\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b },
\tag{13}\end{align}
$$
where $\alpha$ is a learning rate. Repeat the process in a loop until the cost function stops decreasing.

%%$\alpha$ 是学习率，这个循环将一直重复直到成本函数停止下降。%%
<a name='3.3'></a>
### 3.3 - Dataset

Let's build a linear regression model for a Kaggle dataset [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), saved in a file `data/house_prices_train.csv`. You will use two fields - ground living area (`GrLivArea`, square feet) and rates of the overall quality of material and finish (`OverallQual`, 1-10) to predict sales price (`SalePrice`, dollars).

%%
让我们用 Kaggle 的数据集[房屋价格](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 构建一个线性回归模型，它保存在 `data/house_prices_train.csv`。
你将使用居住面积（`GrLivArea`，单位是平方英尺）和材料和整体工艺质量水平（`OverallQual`, 1-10）两个字段来预测售价（`SalePrice`, 单位美元）。
%%

To open the dataset you can use `pandas` function `read_csv`:


```python
df = pd.read_csv('data/house_prices_train.csv')
```

Select the required fields and save them in the variables `X_multi`, `Y_multi`:


```python
X_multi = df[['GrLivArea', 'OverallQual']]
Y_multi = df['SalePrice']
```

Preview the data:


```python
display(X_multi)
display(Y_multi)
```

> [!result]
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GrLivArea</th>
      <th>OverallQual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1710</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1786</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1717</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2198</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1647</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2073</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2340</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1078</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1256</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 2 columns</p>
</div>
> 
> 
> 
    0       208500
    1       181500
    2       223500
    3       140000
    4       250000
             ...  
    1455    175000
    1456    210000
    1457    266500
    1458    142125
    1459    147500
    Name: SalePrice, Length: 1460, dtype: int64


Normalize the data:


```python
X_multi_norm = (X_multi - X_multi.mean(axis=0))/np.std(X_multi)
Y_multi_norm = (Y_multi - Y_multi.mean(axis=0))/np.std(Y_multi)
```

> [!result]
>     /home/w0330t/Projects/ machine_learning_and_data_science/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:4062: FutureWarning: The behavior of DataFrame.std with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
>       return std(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)


Convert results to the `NumPy` arrays, transpose `X_multi_norm` to get an array of a shape ($2 \times m$) and reshape `Y_multi_norm` to bring it to the shape ($1 \times m$):

将标准化后的数据转为 `NumPy` 数组，对 `X_multi_norm` 进行转置，获得一个形状为 $2 \times m$ 数组，然后将 `Y_multi_norm` 重塑为形状 $1 \times m$：

```python
X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))

print ('The shape of X: ' + str(X_multi_norm.shape))
print ('The shape of Y: ' + str(Y_multi_norm.shape))
print ('I have m = %d training examples!' % (X_multi_norm.shape[1]))
```

> [!result]
>     The shape of X: (2, 1460)
>     The shape of Y: (1, 1460)
>     I have m = 1460 training examples!


### 3.4 - Performance of the Neural Network Model for Multiple Linear Regression

Now... you do not need to change anything in your neural network implementation! Go through the code in section [2](#2) and see that if you pass new datasets `X_multi_norm` and `Y_multi_norm`, the input layer size $n_x$ will get equal to $2$ and the rest of the implementation will remain exactly the same, even the backward propagation!
%%
你不需要更改任何关于你的神经网络的实现！
查看[[#2 - Implementation of the Neural Network Model for Linear Regression|第二章节]]的代码，可以看到如果你传入新的数据集 `X_multi_norm` 和 `Y_multi_norm`，输入层的形状 $n_x$ 为 2，而其他的实现保持不变，包括反向传播！
%%

Train the model for $100$ iterations:


```python
parameters_multi = nn_model(X_multi_norm, Y_multi_norm, num_iterations=100 ,print_cost=True)

print("W = " + str(parameters_multi["W"]))
print("b = " + str(parameters_multi["b"]))

W_multi = parameters_multi["W"]
b_multi = parameters_multi["b"]
```

> [!result]
>     Cost after iteration 0: 0.514220
>     Cost after iteration 1: 0.448627
>     Cost after iteration 2: 0.396224
>     Cost after iteration 3: 0.353227
>     Cost after iteration 4: 0.317639
>     Cost after iteration 5: 0.288102
>     Cost after iteration 6: 0.263566
>     Cost after iteration 7: 0.243179
>     Cost after iteration 8: 0.226237
>     Cost after iteration 9: 0.212158
>     Cost after iteration 10: 0.200457
>     Cost after iteration 11: 0.190734
>     Cost after iteration 12: 0.182654
>     Cost after iteration 13: 0.175939
>     Cost after iteration 14: 0.170359
>     Cost after iteration 15: 0.165721
>     Cost after iteration 16: 0.161867
>     Cost after iteration 17: 0.158665
>     Cost after iteration 18: 0.156003
>     Cost after iteration 19: 0.153792
>     Cost after iteration 20: 0.151953
>     Cost after iteration 21: 0.150426
>     Cost after iteration 22: 0.149157
>     Cost after iteration 23: 0.148102
>     Cost after iteration 24: 0.147225
>     Cost after iteration 25: 0.146496
>     Cost after iteration 26: 0.145891
>     Cost after iteration 27: 0.145388
>     Cost after iteration 28: 0.144970
>     Cost after iteration 29: 0.144622
>     Cost after iteration 30: 0.144334
>     Cost after iteration 31: 0.144094
>     Cost after iteration 32: 0.143894
>     Cost after iteration 33: 0.143728
>     Cost after iteration 34: 0.143591
>     Cost after iteration 35: 0.143476
>     Cost after iteration 36: 0.143381
>     Cost after iteration 37: 0.143302
>     Cost after iteration 38: 0.143236
>     Cost after iteration 39: 0.143182
>     Cost after iteration 40: 0.143136
>     Cost after iteration 41: 0.143099
>     Cost after iteration 42: 0.143067
>     Cost after iteration 43: 0.143041
>     Cost after iteration 44: 0.143020
>     Cost after iteration 45: 0.143002
>     Cost after iteration 46: 0.142987
>     Cost after iteration 47: 0.142974
>     Cost after iteration 48: 0.142964
>     Cost after iteration 49: 0.142956
>     Cost after iteration 50: 0.142948
>     Cost after iteration 51: 0.142943
>     Cost after iteration 52: 0.142938
>     Cost after iteration 53: 0.142934
>     Cost after iteration 54: 0.142930
>     Cost after iteration 55: 0.142927
>     Cost after iteration 56: 0.142925
>     Cost after iteration 57: 0.142923
>     Cost after iteration 58: 0.142921
>     Cost after iteration 59: 0.142920
>     Cost after iteration 60: 0.142919
>     Cost after iteration 61: 0.142918
>     Cost after iteration 62: 0.142917
>     Cost after iteration 63: 0.142917
>     Cost after iteration 64: 0.142916
>     Cost after iteration 65: 0.142916
>     Cost after iteration 66: 0.142915
>     Cost after iteration 67: 0.142915
>     Cost after iteration 68: 0.142915
>     Cost after iteration 69: 0.142914
>     Cost after iteration 70: 0.142914
>     Cost after iteration 71: 0.142914
>     Cost after iteration 72: 0.142914
>     Cost after iteration 73: 0.142914
>     Cost after iteration 74: 0.142914
>     Cost after iteration 75: 0.142914
>     Cost after iteration 76: 0.142914
>     Cost after iteration 77: 0.142914
>     Cost after iteration 78: 0.142914
>     Cost after iteration 79: 0.142914
>     Cost after iteration 80: 0.142914
>     Cost after iteration 81: 0.142914
>     Cost after iteration 82: 0.142913
>     Cost after iteration 83: 0.142913
>     Cost after iteration 84: 0.142913
>     Cost after iteration 85: 0.142913
>     Cost after iteration 86: 0.142913
>     Cost after iteration 87: 0.142913
>     Cost after iteration 88: 0.142913
>     Cost after iteration 89: 0.142913
>     Cost after iteration 90: 0.142913
>     Cost after iteration 91: 0.142913
>     Cost after iteration 92: 0.142913
>     Cost after iteration 93: 0.142913
>     Cost after iteration 94: 0.142913
>     Cost after iteration 95: 0.142913
>     Cost after iteration 96: 0.142913
>     Cost after iteration 97: 0.142913
>     Cost after iteration 98: 0.142913
>     Cost after iteration 99: 0.142913
>     W = [[0.3694604  0.57181575]]
>     b = [[1.21181604e-16]]


Now you are ready to make predictions:


```python
X_pred_multi = np.array([[1710, 7], [1200, 6], [2200, 8]]).T
Y_pred_multi = predict(X_multi, Y_multi, parameters_multi, X_pred_multi)

print(f"Ground living area, square feet:\n{X_pred_multi[0]}")
print(f"Rates of the overall quality of material and finish, 1-10:\n{X_pred_multi[1]}")
print(f"Predictions of sales price, $:\n{np.round(Y_pred_multi)}")
```

> [!result]
>     Ground living area, square feet:
>     [1710 1200 2200]
>     Rates of the overall quality of material and finish, 1-10:
>     [7 6 8]
>     Predictions of sales price, $:
>     [221371. 160039. 281587.]


Congrats on finishing this lab!
