---
aliases: [Eigenvalues and Eigenvectors]
tags: []
created: 2025-11-10, 18:42:15
modified: 2025-11-11, 19:41:37
---

#  Eigenvalues and Eigenvectors

Welcome to the last assignment of this course and congratulations for making it this far. In this final assignment you will use your knowledge of linear algebra and your skills using Python and NumPy to address some real-world scenarios where linear algebra is actually used to solve and simplify problems.

**After this assignment you will be able to:**
- apply linear transformations, eigenvalues and eigenvectors in a webpage navigation model
- apply PCA on a dataset to reduce its dimensions

%%
- 在网页导航模型中应用线性转换，特征值和特征向量
- 使用 PCA 对数据集进行降维操作。
%%
## Important Note

Please **do not** delete any exercise cells or add your solutions in different cells. **Keep your solution in the original cell provided**; failing to do so will disrupt the autograder.

Additionally, **do not import any new libraries**, and **avoid importing libraries within any cell designated for grading**, as this will also interfere with the autograder's functionality.

# Table of Contents
- [ 1 - Application of Eigenvalues and Eigenvectors: Navigating Webpages](#1)
  - [ Exercise 1](#ex01)
  - [ Exercise 2](#ex02)
- [ 2 - Application of Eigenvalues and Eigenvectors: Principal Component Analysis](#2)
  - [2.1 Load the data](#2.1)
  - [2.2 Get the covariance matrix](#2.2)
    - [ Exercise 3](#ex03)
    - [ Exercise 4](#ex04)
  - [ 2.3 - Compute the eigenvalues and eigenvectors](#2.3)
  - [ 2.4 Transform the centered data with PCA](#2.4)
    - [ Exercise 5](#ex05)
  - [ 2.5 Analyzing the dimensionality reduction in 2 dimensions](#2.5)
  - [ 2.6 Reconstructing the images from the eigenvectors](#2.6)
  - [ 2.7 Explained variance](#2.7)

## Packages

Run the following cell to load the packages you'll need.


```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
```

Load the utils module and the unit tests defined for this notebook.


```python
import utils
import w4_unittest
```

<a name='1'></a>

---

## 1 - Application of Eigenvalues and Eigenvectors: Navigating Webpages

As you learned in the lectures, eigenvalues and eigenvectors play a very important role in what's called (discrete) dynamical systems. As you might recall, a **discrete dynamical system** describes a system where, as time goes by, the state changes according to some process. When defining this dynamical systems you could represent all the possible states, such as sunny, rainy or cloudy, in a vector called the **state vector**. 

%%
正如你在讲座中学到的那样，特征值和特征向量在被称为(离散)动力系统中扮演着重要的角色。
如果你记得，一个离散动力系统描述的是一个随着时间的推移，其状态根据特定过程而变化的系统。
当定义了这个系统就可以代表你所有的可能的状态，比如晴天雨天或者多云，都在这个状态向量的向量来表示。
%%

Each discrete dynamical system can be represented by a transition matrix $P$, which indicates, given a particular state, what are the chances or probabilities of moving to each of the other states. This means the element $(2,1)$ of the matrix represents the probability of transitioning from state $1$ to state $2$.

%%
每个离散动力系统都能使用一个转移矩阵 $P$ 来表示，这表明，给出一个特定状态，转移到其他状态的可能性和概率。
这意味着这个元素(2，1)表示从状态 1 到状态 2 转移的概率。
%%
 
Starting with an initial state $X_0$, the transition to the next state $X_1$ is a linear transformation defined by the transition matrix $P$: $X_1=PX_0$. That leads to $X_2=PX_1=P^2X_0$, $X_3=P^3X_0$, and so on. This implies that $X_t=PX_{t-1}$ for $t=0,1,2,3,\ldots$. In other words, we can keep multiplying by `P` to move from one state to the next.

%%
最初的状态是 $X_0$，转移的下一个状态 $X_{1}$ 是由一个线性转换定义的，它的转换矩阵写作：$P: X_{1}=PX_{0}$ 。
一次为线索可以得到 $X_2=PX_1=P^2X_0$, $X_3=P^3X_0$ 等等。
这意味着当 $t=0,1,2,3,\ldots$ 时 $X_t=PX_{t-1}$ 。我们可以继续乘以 `P` 将状态转移到下一个状态。
%%

One application of discrete dynamical systems is to model browsing web pages. Web pages often contain links to other pages, so the dynamical system would model how a user goes from one page to another by hopping from link to link. For simplicity, assume that the browser is only following links to a new page rather than navigating to an unlinked one. 

%%
关于离散动力系统的一个应用是模拟网页浏览的行为。
网页通常有许多链接到其他页面的链接，所以这个离散动力系统将模拟用户如何通过一个个链接进行页面跳转。
为了简化，假设浏览器仅通过点击链接跳转到新页面，而非直接导航至未链接的页面。
%%

In this case, the state vector $X_t$ will be the probabilities that the browser is on a particular page at time $t$. Navigation from one page to another advances the model from one state vector $X_{t-1}$ to another state vector $X_t$. A linear transformation, defined by a matrix $P$, will have entries $p_{ij}$ with the probabilities that the browser navigates to page $j$ from page $i$. For fixed column $j$, the entries represent a probability distribution describing location of the browser at the next step, given that you are at state $j$. Thus, the entries in each column must add to 1.

%%
在这个例子中，状态向量 $X_t$ 表示在时间 $t$ 的时候浏览器在某个特定页面的概率。
导航从一个页面到另一个页面的进展这个模型从一个状态向量 $X_{t-1}$ 到另外一个状态向量 $X_{t}$ 。
一个由矩阵 $P$ 定义的线性变换，其矩阵元素 $p_{ij}$ 即为浏览器从页面 $i$ 跳转到页面 $j$ 的概率。
因此，每列中的条目之和必须为 1。
%%

<a name='ex01'></a>
### Exercise 1

For the sake of the example, consider there are only a small number of pages $n=5$. This means that the transition matrix $P$ will be a $5 \times 5$ matrix. In this particular case, all elements on the main diagonal should be equal to $0$, since we are making the reasonable assumption that there is no existing link to the current page. Also, as metioned before, all the entries in each column must add to one. Here is an example of such a matrix for $n=5$:

%%
考虑这里只有几个页面，$n=5$。
这意味着这个转换矩阵 $P$ 是一个 $5 \times5$ 的矩阵。
在这个示例中，对角线的必须等于 0，因为我们合理的假设当前页面没有指向自身的链接。
当然，之前也提到了，每列中的条目和必须为 1。
下面就是这个矩阵的示例。
%%

$$P=
\begin{bmatrix}
0    & 0.75 & 0.35 & 0.25 & 0.85 \\
0.15 & 0    & 0.35 & 0.25 & 0.05 \\
0.15 & 0.15 & 0    & 0.25 & 0.05 \\
0.15 & 0.05 & 0.05 & 0    & 0.05 \\
0.55 & 0.05 & 0.25 & 0.25 & 0
\end{bmatrix}\tag{5}
$$

Define vector $X_0$, so the browser starts navigation at page $4$ ($X_0$ is a vector with a single entry equal to one, and all other entries equal to zero). Apply the transformation once: $X_1=PX_0$ to find a vector of the probabilities that the browser is at each of four pages.

%%
定义向量 $X_0$，然后让浏览器从第四个页面开始导航（$X_0$ 一个其中一项为 1，其他项目为 0 ）
应用一次线性变换 $X_1=PX_0$ ，可以得到一个表示浏览器在 4 页中每页概率的向量。
%%


```python
P = np.array([ 
    
    [0, 0.75, 0.35, 0.25, 0.85], 
    [0.15, 0, 0.35, 0.25, 0.05], 
    [0.15, 0.15, 0, 0.25, 0.05], 
    [0.15, 0.05, 0.05, 0, 0.05], 
    [0.55, 0.05, 0.25, 0.25, 0]  
]) 

X0 = np.array([[0],[0],[0],[1],[0]])

### START CODE HERE ###

# Multiply matrix P and X_0 (matrix multiplication).
X1 = P @ X0

### END CODE HERE ###

print(f'Sum of columns of P: {sum(P)}')
print(f'X1:\n{X1}')
```

    Sum of columns of P: [1. 1. 1. 1. 1.]
    X1:
    [[0.25]
     [0.25]
     [0.25]
     [0.  ]
     [0.25]]


##### __Expected Output__

```Python
Sum of columns of P: [1. 1. 1. 1. 1.]
X1:
[[0.25]
 [0.25]
 [0.25]
 [0.  ]
 [0.25]]
```

Applying the transformation $m$ times you can find a vector $X_m$ with the probabilities of the browser being at each of the pages after $m$ steps of navigation.

%%
应用这个线性变换 $m$ 次之后，你能找到一个浏览器访问页面 m 次之后导航到某个页面的概率向量 $X_{m}$ 
%%


```python
X = np.array([[0],[0],[0],[1],[0]])
m = 20

for t in range(m):
    X = P @ X
    
print(X)
```

    [[0.39392366]
     [0.13392366]
     [0.11407667]
     [0.0850993 ]
     [0.27297672]]


It is useful to predict the probabilities in $X_m$ when $m$ is large, and thus determine what pages a browser is more likely to visit after a long period of browsing the web. In other words, we want to know which pages ultimately get the most traffic. One way to do that is just apply the transformation many times, and with this small $5 \times 5$ example you can do that just fine. In real life problems, however, you'll have enormous matrices and doing so will be computationally expensive. Here is where eigenvalues and eigenvectors can help here significantly reducing the amount of calculations. Let's see how!

%%
当 $m$ 值很大的时候，$X_m$ 的概率预测有着重要的意义，这有助于确定浏览器在长时间浏览网页后更有可能访问哪些页面。
换言之，我们可以知道哪些页面最终可以获得最大的流量。
其中一个方法是运行这个线性变换很多次，对于这个小小的 $5 \times 5$ 的示例是完全可行的。
但是在真实生活的例子中，你需要计算的矩阵非常的巨大导致这样的方法成本非常高昂。
在这里特征值和特征向量可以显著减少计算量。
%%

Begin by finding eigenvalues and eigenvectors for the previously defined matrix $P$:

%%
寻找之前矩阵 $P$ 的特征值和特征向量。
%%


```python
eigenvals, eigenvecs = np.linalg.eig(P)
print(f'Eigenvalues of P:\n{eigenvals}\n\nEigenvectors of P\n{eigenvecs}')
```

    Eigenvalues of P:
    [ 1.         -0.70367062  0.00539505 -0.08267227 -0.21905217]
    
    Eigenvectors of P
    [[-0.76088562 -0.81362074  0.10935376  0.14270615 -0.39408574]
     [-0.25879453  0.050269   -0.6653158   0.67528802 -0.66465044]
     [-0.2204546   0.07869601 -0.29090665  0.17007443  0.35048734]
     [-0.1644783   0.12446953  0.19740707 -0.43678067  0.23311487]
     [-0.52766004  0.56018621  0.64946163 -0.55128793  0.47513398]]


As you can see, there is one eigenvalue with value $1$, and the other four have an aboslute values smaller than 1. It turns out this is a property of transition matrices. In fact, they have so many properties that these types of matrices fall into a category of matrices called **Markov matrix**. 

%%
正如你所见，这里有一个特征值为 1 的特征向量，同时其他 4 个的绝对值都小于 1。
这是转移矩阵的一个性质。
事实上，它们有许多特性，所以这类矩阵被归类为马尔可夫矩阵。
%%

In general, a square matrix whose entries are all nonnegative, and the sum of the elements for each column is equal to $1$ is called a **Markov matrix**. Markov matrices have a handy property - they always have an eigenvalue equal to 1. As you learned in the lectures, in the case of transition matrices, the eigenvector associated with the eigenvalue $1$ will determine the state of the model in the long run , after evolving for a long period of time. 

%%
一般来说，一个方阵，它的元素没有负值，且每个元素列相加等于 1 ，这个矩阵被称为马尔可夫矩阵。
马尔可夫矩阵有一个方便的特性——它们总有一个特征值为 1。
正如你在讲座中学习到的那样，在当前这个例子的转移矩阵中，经过长时间的演变后，这个特征向量关联的特征值 1 将决定模型的长期状态。
%%

You can easily verify that the matrix $P$ you defined earlier is in fact a Markov matrix. 

%%
你可以简单的验证这个之前定义的矩阵 $P$ 是否是马尔可夫矩阵。
%%

So, if $m$ is large enough, the equation $X_m=PX_{m-1}$ can be rewritten as $X_m=PX_{m-1}=1\times X_m$. This means that predicting probabilities at time $m$, when $m$ is large you can simply just look for an eigenvector corresponding to the eigenvalue $1$. 

%%
当 $m$ 已经足够大的时候，这个方程 $X_m=PX_{m-1}$ 可以被重新写作 $X_m=PX_{m-1}=1\times X_m$。
这意味着，当 m 值很大时，要预测 m 时刻的概率，你只需寻找与特征值 1 相对应的特征向量即可。
%%

So, let's extract the eigenvector associated to the eigenvalue $1$. 

%%
所以，让我们提取特征值为 1 相关的特征向量吧。
%%

```python
X_inf = eigenvecs[:,0]

print(f"Eigenvector corresponding to the eigenvalue 1:\n{X_inf[:,np.newaxis]}")
```

    Eigenvector corresponding to the eigenvalue 1:
    [[-0.76088562]
     [-0.25879453]
     [-0.2204546 ]
     [-0.1644783 ]
     [-0.52766004]]


<a name='ex02'></a>
### Exercise 2

Just to verify the results, perform matrix multiplication $PX$ (multiply matrix `P` and vector `X_inf`) to check that the result will be equal to the vector $X$ (`X_inf`).

%%
验证这个结果，执行矩阵乘法 $PX$ (矩阵 `P` 和向量 `X_inf` 相乘) 后检查结果是否等于向量 $X$ (`X_inf`)
%%

```python
# This is organised as a function only for grading purposes.
def check_eigenvector(P, X_inf):
    ### START CODE HERE ###
    X_check = P @ X_inf
    ### END CODE HERE ###
    return X_check

X_check = check_eigenvector(P, X_inf)
print("Original eigenvector corresponding to the eigenvalue 1:\n" + str(X_inf))
print("Result of multiplication:" + str(X_check))

# Function np.isclose compares two NumPy arrays element by element, allowing for error tolerance (rtol parameter).
print("Check that PX=X element by element:" + str(np.isclose(X_inf, X_check, rtol=1e-10)))
```

    Original eigenvector corresponding to the eigenvalue 1:
    [-0.76088562 -0.25879453 -0.2204546  -0.1644783  -0.52766004]
    Result of multiplication:[-0.76088562 -0.25879453 -0.2204546  -0.1644783  -0.52766004]
    Check that PX=X element by element:[ True  True  True  True  True]


This result gives the direction of the eigenvector, but as you can see the entries can't be interpreted as probabilities since you have negative values, and they don't add to 1. That's no problem. Remember that by convention `np.eig` returns eigenvectors with norm 1, but actually any vector on the same line is also an eigenvector to the eigenvalue 1, so you can simply scale the vector so that all entries are positive and add to one.This will give you the long-run probabilities of landing on a given web page.

%%
这个结果给出了特征向量，但是这些条目不能解释为概率，因为它们包含负值，同时它们相加不等于 1。
这不是什么大问题，按照惯例 `np.eig` 返回的是范数为 1 的特征向量，那么与该特征向量同线的任何向量，也都是特征值为  1 的特征向量，所以你可以简单的缩放这个向量让它变为正值并且让其元素相加为 1。
这样就可以得到访问某个特定网页的长期概率。
%%

```python
X_inf = X_inf/sum(X_inf)
print(f"Long-run probabilities of being at each webpage:\n{X_inf[:,np.newaxis]}")
```

    Long-run probabilities of being at each webpage:
    [[0.39377747]
     [0.13393269]
     [0.11409081]
     [0.08512166]
     [0.27307736]]


This means that after navigating the web for a long time, the probability that the browser is at page 1 is 0.394, of being on page 2 is 0.134, on page 3 0.114, on page 4 0.085, and finally page 5 has a probability of 0.273.

%%
这说明长时间浏览后，浏览页面 1 的可能性是 0.394，页面 2 是 0.134，页面 3 是 0.114，页面 4 是 0.085，页面 5 的概率则是 0.273。
%%

Looking at this result you can conclude that page 1 is the most likely for the browser to be at, while page 4 is the least probable one.

%%
根据此结果，可以推断浏览器最可能处于页面 1，而页面 4 的可能性最小。
%%

If you compare the result of `X_inf` with the one you got after evolving the systems 20 times, they are the same up to the third decimal!

%%
如果你将 X_inf 的结果与系统演化 20 次后所得的结果进行比较，你会发现两者精确到小数点后第三位都完全一致！
%%

Here is a fun fact: this type of a model was the foundation of the PageRank algorithm, which is the basis of Google's very successful search engine.

%%
这里有一个有趣的事实：这种模型是 PageRank 算法的基础，而 PageRank 算法正是谷歌公司大获成功的搜索引擎的核心。
%%


---

<a name='2'></a>
## 2 - Application of Eigenvalues and Eigenvectors: Principal Component Analysis

As you learned in the lectures, one of the useful applications of eigenvalues and eigenvectors is the dimensionality reduction algorithm called Principal Component Analyisis, or PCA for short.

%%
正如您在课程中所学，特征值和特征向量的一项重要实用应用，便是主成分分析（PCA）这一降维算法。
%%

In this second section of the assignment you will be applying PCA on an image dataset to perform image compression. 

%%
在本文的第二节你可以应用 PCA 对图片数据集进行图像压缩。
%%

You will be using a portion of the [Cat and dog face](https://www.kaggle.com/datasets/alessiosanna/cat-dog-64x64-pixel/data) dataset from Kaggle. In particular, you will be using the cat images.

%%
您将使用 Kaggle 猫狗面部数据集的一部分。具体而言，您会用到其中的猫图像。
%%

Remember that to apply PCA on any dataset you will begin by defining the covariance matrix. After that you will compute the eigenvalues and eigenvectors of this covariance matrix. Each of these eigenvectors will be a **principal component**. To perform the dimensionality reduction, you will take the $k$ principal components associated to the $k$ biggest eigenvalues, and transform the original data by projecting it onto the direction of these principal components (eigenvectors).

%%
值得注意的是，在对任何数据集应用主成分分析 (PCA) 时，首先需要定义其协方差矩阵。
之后你将计算这个协方差的特征值和特征向量。
每一个特征向量都对应一个主成分。
为了进行降维，你需要获取主成分 k 关联的最大的特征值 k，然后将原始数据投影到这些主成分（特征向量）的方向上进行转换。
%%
<a name='2.1'></a>
### 2.1 - Load the data
Begin by loading the images and transforming them to black and white using `load_images` function from utils. 

%%
使用 `load_images` 读取图片数据并将它们转换为黑白图片。
%%

```python
imgs = utils.load_images('./data/')
```

`imgs` should be a list, where each element of the list is an array (matrix). Let's check it out

%%
`imgs` 是一个列表，列表的每个元素都是一个数组（矩阵），让我们检查一下
%%


```python
height, width = imgs[0].shape

print(f'\nYour dataset has {len(imgs)} images of size {height}x{width} pixels\n')
```


    Your dataset has 55 images of size 64x64 pixels



Go ahead and plot one image to see what they look like. You can use the colormap 'gray' to plot in black and white. Feel free to look into as many pictures as you want.

%%
先绘制一张图片看看它们张什么样子。
你可以使用颜色图'灰'来绘制黑白图像。
想看多少图片都行。
%%


```python
plt.imshow(imgs[8], cmap='gray')
```

`<matplotlib.image.AxesImage at 0x7f4d1e8b2510>`

![C1W4_Assignment_28_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_28_1.png)



When working with images, you can consider each pixel as a variable. Having each image in matrix form is good for visualizing the image, but not so much for operating on each variable. 

%%
在处理图片的时候，你可以思考一下每个像素点的变量。
将每张图像以矩阵形式呈现，固然有助于图像的可视化，但若要对每个变量进行操作，这种形式则显得不甚理想。
%%

In order to apply PCA for dimensionality reduction you will need to flatten each image into a single row vector. You can do this using the `reshape` function from NumPy. 

%%
若要进行 PCA 降维，您需要将每张图像展平为单个行向量。
你可以使用 Numpy 的 `reshape` 函数来实现。
%%

The resulting array will have 55 rows, one for each image, and 64x64=4096 columns.

%%
这个结果中的数组有 55 行，每行就是一个图片，每个图片有 64x64=4096 列。
%%

```python
imgs_flatten = np.array([im.reshape(-1) for im in imgs])

print(f'imgs_flatten shape: {imgs_flatten.shape}')
```

    imgs_flatten shape: (55, 4096)

<a name='2.2'></a>
### 2.2 - Get the covariance matrix

Now that you have the images in the correct shape you are ready to apply PCA on the flattened dataset. 

%%
既然图像已调整为正确的形状，您便可以对扁平化数据集执行 PCA。
%%

If you consider each pixel (column) as a variable, and each image (rows) as an obervation you will have 55 observations of 4096 variables, $X_1, X_2, \ldots, X_{4096}$ so that

%%
如果你将每个像素（列）视为一个变量，并将每张图像（行）视为一个观测值，那么你将得到 55 个观测值，每个观测值包含 4096 个变量，即 X₁, X₂, ..., X₄₀₉₆，所以可以得到：
%%
$$\mathrm{imgs\_flatten} = \begin{bmatrix} x_{1,1} & x_{1,2} & \ldots & x_{1,4096}\\
                                           x_{2,1} & x_{2,2} & \ldots & x_{2,4096} \\
                                           \vdots & \vdots & \ddots & \vdots \\
                                           x_{55,1} & x_{55,2} & \ldots & x_{55,4096}\end{bmatrix}$$


As you might remember from the lectures, to compute PCA you first need to find the covariance matrix

%%
也许你记得讲座中说过，需要计算 PCA 需要先找到它的协方差矩阵。
%%

$$\Sigma = \begin{bmatrix}Var(X_1) & Cov(X_1, X_2) & \ldots & Cov(X_1, X_{4096}) \\
                          Cov(X_1, X_2) & Var(X_2) & \ldots & Cov(X_2, X_{4096})\\
                          \vdots & \vdots & \ddots & \vdots \\
                          Cov(X_1,X_{4096}) & Cov(X_2, X_{4096}) &\ldots & Var(X_{4096})\end{bmatrix}$$

<a name='ex03'></a>
#### Exercise 3

In order to get the covariance matrix you first need to center the data by subtracting the mean for each variable (column). 

%%
如果要获取这个协方差矩阵你首先对数据进行中心化，也就是减去每列的平均值。
%%

As you've seen in the lectures, the centered data matrix looks something like this:

%%
正如你在讲座中看到的那样，数据中心化后长这样：
%%

$$X = \begin{bmatrix} (x_{1,1}- \mu_1) & (x_{1,2}- \mu_2) & \ldots & (x_{1,4096}- \mu_{4096})\\
                                           (x_{2,1}- \mu_1) & (x_{2,2}- \mu_2) & \ldots & (x_{2,4096}- \mu_{4096}) \\
                                           \vdots & \vdots & \ddots & \vdots \\
                                           (x_{55,1}- \mu_1) & (x_{55,2}- \mu_2) & \ldots & (x_{55,4096}- \mu_{4096})\end{bmatrix}$$

From the lectures you know that, for example, the mean of the first variable (pixel) can be found as the mean of all the observations: $\mu_1 = \frac{1}{55} \sum_{i=1}^{55} x_{i,1}$.

%%
从讲座中你了解到，例如，第一个变量（像素）的均值可以通过所有观测值的均值计算得到
$$
\mu_1 = \frac{1}{55} \sum_{i=1}^{55} x_{i,1}
$$
%%

For the following exercise you will implement a function that takes an array of shape $\mathrm{Num. observations}\times\mathrm{Num. variables}$, and returns the centered data. 

%%
为了中心化你需要三个 Numy 函数，如果你需要阅读官方文档可以点击它们的函数名。
%%

To perform the centering you will need three numpy functions. Click on their names if you want to read the official documentation for each in more detail:
- [`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html): use this function to compute the mean of each variable, just remember to pass the correct `axis` argument.
- [`np.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy-repeat): This will allow for you to repeat the values of each $\mu_i$ . 
- [`np.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy-reshape): Use this function to reshape the repeated values into a matrix of shape the same size as your input data. To get the correct matrix after the reshape, remember to use the parameter `order='F'`.


%%
- [`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html): 计算每个变量的均值，注意需要传入 `axis` 正确的参数。
- [`np.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy-repeat): 它会让你重复每个 $\mu_i$ 的值. 
- [`np.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy-reshape): 利用此函数，将重复值重塑为与您的输入数据形状相同的矩阵。为了确保重塑后的矩阵正确无误，请务必使用参数 `order='F'`。
%%

```python
# Graded cell
def center_data(Y):
    """
    Center your original data
    Args:
         Y (ndarray): input data. Shape (n_observations x n_pixels)
    Outputs:
        X (ndarray): centered data
    """
    ### START CODE HERE ###
    mean_vector = np.mean(Y, axis=0)
    mean_matrix = np.repeat(mean_vector, Y.shape[0])
    # use np.reshape to reshape into a matrix with the same size as Y. Remember to use order='F'
    mean_matrix = np.reshape(mean_matrix, Y.shape, order='F')

    X = Y - mean_matrix
    ### END CODE HERE ###
    return X
```

> [!note]
> 完全不用这么麻烦，Y-mean_vector 即可搞定，少两行代码


Go ahead and apply the `center_data` function to your data in `imgs_flatten`. 

%%
将你的数据 `imgs_flatten` 传入 `center_data` 函数执行。
%%

You can also print the image again and check that the face of the cat still looks the same. This is because the color scale is not fixed, but rather relative to the values of the pixels. 

%%
你同样可以再次打印这个图片，可以看到这个猫脸和上面的是一致的。
这是因为相对于像素的数值而言，色阶并不是固定的。
%%

```python
X = center_data(imgs_flatten)
plt.imshow(X[8].reshape(64,64), cmap='gray')
```

    <matplotlib.image.AxesImage at 0x7f4d130e5250>


![C1W4_Assignment_34_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_34_1.png)



#### Exercise 4

Now that you have your centered data, $X$, you can go ahead and find the covariance matrix 

%%
现在你有了中心化的数据 $X$，你可以去寻找它的协方差矩阵了。
%%

You might remember from the lectures that once you have your centered data, the covariance matrix can be found by appliying the dot product between $X^T$ and $X$, and divide by the number of observations minus 1.

%%
你可能还记得，在课程中我们曾提到过，一旦你获得了中心化数据，协方差矩阵可以通过对 $X$ 的转置（$X^T$）与 $X$ 进行点积运算，再除以观测数量减一来计算得到。
%%

To perform the dot product you can simply use the function [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy-dot).

%%
执行点积的计算非常简单，你可以使用函数 [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy-dot)
%%

```python
def get_cov_matrix(X):
    """ Calculate covariance matrix from centered data X
    Args:
        X (np.ndarray): centered data matrix
    Outputs:
        cov_matrix (np.ndarray): covariance matrix
    """

    ### START CODE HERE ###
    cov_matrix = np.dot(X.T, X)
    cov_matrix = cov_matrix / (X.shape[0] - 1)
    ### END CODE HERE ###

    return cov_matrix
```

> [!note]
> 直接@也很香

```python
cov_matrix = get_cov_matrix(X)
```

Check the dimensions of the covariance matrix, it should be a square matrix with 4096 rows and columns. 

%%
检查这个降维的协方差，它应该是一个行列均为 4096 的方阵。
%%

```python
print(f'Covariance matrix shape: {cov_matrix.shape}')
```

    Covariance matrix shape: (4096, 4096)

<a name='2.3'></a>
### 2.3 - Compute the eigenvalues and eigenvectors
Now you are all set to compute the eigenvalues and eigenvectors of the covariance matrix.
Due to performance constaints, you will not be using `np.linalg.eig`, but rather the very similar function [`scipy.sparse.linalg.eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html), which exploits the fact that $\mathrm{cov\_matrix}^T=\mathrm{cov\_matrix}$. Also, this function allows you to compute fewer number of eigenvalue-eigenvector pairs. 

%%
现在你已经得到了计算特征值和特征矩阵的协方差矩阵。
由于性能限制，你不能使用 `np.linalg.eig`，而是使用非常相似的 [`scipy.sparse.linalg.eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)，它利用了协方差的转置等于其本身的特性 $\mathrm{cov\_matrix}^T=\mathrm{cov\_matrix}$。
此外，此函数还允许您仅计算少量的特征值-特征向量对。
%%

It is outside of the scope of this course, but it can be shown that at most 55 eigenvalues of `cov_matrix` will be different from zero, which is the smallest dimension of the data matrix `X`. Thus, for computational efficiency, you will only be computing the first biggest 55 eigenvalues $\lambda_1, \ldots, \lambda_{55}$ and their corresponding eigenvectors $v_1, \ldots, v_{55}$. Feel free to try changing the `k` parameter in `scipy.sparse.linalg.eigsh` to something slightly bigger, to verify that all the new eigenvalues are zero. Try to keep it below 80, otherwise it will take too long to compute. 

%%
虽然有点超纲，协方差矩阵最多只有 55 个非零特征值这件事是可以证明的。然后这恰好是数据矩阵 X 的最小维度。因此，为了计算效率，你只需要计算最大的 55 个特征向量 $\lambda_1, \ldots, \lambda_{55}$ 和它们对应的特征向量 $v_1, \ldots, v_{55}$。
你可以随意尝试将函数 `scipy.sparse.linalg.eigsh` 的 `k` 参数的值轻微的增大，验证新的特征值是否为 0。
尽量保持在 80 内，否则计算时间会过长。
%%

The outputs of this scipy function are exactly the same as the ones from `np.linalg.eig`, except eigenvalues are ordered in decreasing order, so if you want to check out the largest eigenvalue you need to look into the last position of the vector. 

%%
除了特征值按降序排列之外，这个 scipy 的函数的输出和 `np.linalg.eig` 一模一样，因此，若想查看最大特征值，你需关注向量中的最后一个。
%%

```python
# scipy.random.seed(7)
eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(cov_matrix, k=55)
print(f'Ten largest eigenvalues: \n{eigenvals[-10:]}')
```

    Ten largest eigenvalues: 
    [ 293228.19669624  383726.58867453  399184.59618649  479311.0892984
      839689.08870292  878711.73402995 1011093.05519075 1536617.26289103
     2483710.87630303 4199357.2773408 ]


The random seed is fixed in the code above to help ensure the same eigenvectors are calculated each time. This is because for each eigenvector, there are actually two possible outcomes with norm 1. They fall on the same line but point in opposite directions. An example of this would be the vectors 

%%
这个随机种子会确保代码每次运行都会输出相同的特征向量和特征值。
这是因为每个特征向量实际上存在两个番薯为 1 的可能性的结果。
它们位于同一条直线上但方向相反。
比如说下面这个向量：
%%

$$\begin{bmatrix}0.25 \\0.25 \\ -0.25 \\ 0.25 \end{bmatrix} \text{and } \begin{bmatrix}-0.25 \\ -0.25 \\ 0.25 \\ -0.25 \end{bmatrix}.$$

Both possibilities are correct, but by fixing the seed you guarantee you will always get the same result. 

%%
两个可能性都是正确的，但是通过固定种子，你总可以得到相同的结果。
%%

In order to get a consistent result with `np.linalg.eig`, you will invert the order of `eigenvals` and `eigenvecs`, so they are both ordered from largest to smallest eigenvalue.

%%
若想得到和 `np.linalg.eig` 一致的结果，你需要颠倒 `eigenvals` 和 `eigenvecs`，然后它们都是从大到小的特征值了。
%%

```python
eigenvals = eigenvals[::-1]
eigenvecs = eigenvecs[:,::-1]

print(f'Ten largest eigenvalues: \n{eigenvals[:10]}')
```

    Ten largest eigenvalues: 
    [4199357.2773408  2483710.87630303 1536617.26289103 1011093.05519075
      878711.73402995  839689.08870292  479311.0892984   399184.59618649
      383726.58867453  293228.19669624]


Each of the eigenvectors you found will represent one principal component. The eigenvector associated with the largest eigenvalue will be the first principal component, the eigenvector associated with the second largest eigenvalue will be the second principal component, and so on. 

%%
你找到的每个特征向量都代表了一个主成分。
特征向量相对应的最大的特征值则成为了第一个主成分，特征向量对应的第二大的特征值则成为了第二个主成分，以此类推。
%%

It is pretty interesting to see that each principal component usually extracts some relevant features, or patterns from each image. In the next cell you will be visualizing the first sixteen components

%%
有趣的是，每个主成分通常能从每张图像中提取出一些相关的特征或模式。
在下面你将看到前十六个组件的可视化。
%%

```python
fig, ax = plt.subplots(4,4, figsize=(20,20))
for n in range(4):
    for k in range(4):
        ax[n,k].imshow(eigenvecs[:,n*4+k].reshape(height,width), cmap='gray')
        ax[n,k].set_title(f'component number {n*4+k+1}')
```

![C1W4_Assignment_45_0.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_45_0.png)



What can you say about each of the principal components? 

%%
您能对各个主成分进行阐述吗？
%%
<a name='2.4'></a>
### 2.4 Transform the centered data with PCA

Now that you have the first 55 eigenvalue-eivenvector pairs, you can transform your data to reduce the dimensions. Remember that your data originally consisted of 4096 variables. Suppose you want to reduce that to just 2 dimensions, then all you need to do to perform the reduction with PCA is take the dot product between your centered data and the matrix $\boldsymbol{V}=\begin{bmatrix} v_1 & v_2 \end{bmatrix}$, whose columns are the first 2 eigenvectors, or principal components, associated to the 2 largest eigenvalues.

%%
现在你有最初 55 个特征值-特征向量对，您可以转换数据以减少它的维度了。
请记住你的原式数据又 4096 个变量。
假设您希望将其维度缩减至仅两维，那么在利用主成分分析（PCA）进行降维时，您只需将中心化数据与矩阵 $\boldsymbol{V}=\begin{bmatrix} v_1 & v_2 \end{bmatrix}$ 进行点积运算即可，矩阵 $V$ 的列是对应于 2 个最大特征值的前 2 个特征向量，亦即主成分。
%%

<a name='ex03'></a>
#### Exercise 5

In the next cell you will define a function that, given the data matrix, the eigenvector matrix (always sorted according to decreasing eignevalues), and the number of principal components to use, performs PCA.

%%
在下一个单元格中，您将定义一个执行主成分分析（PCA）的函数。该函数接受数据矩阵、特征向量矩阵（始终按特征值降序排序）以及要使用的主成分数量作为参数。
%%

```python
# GRADED cell
def perform_PCA(X, eigenvecs, k):
    """
    Perform dimensionality reduction with PCA
    Inputs:
        X (ndarray): original data matrix. Has dimensions (n_observations)x(n_variables)
        eigenvecs (ndarray): matrix of eigenvectors. Each column is one eigenvector. The k-th eigenvector 
                            is associated to the k-th eigenvalue
        k (int): number of principal components to use
    Returns:
        Xred
    """
    
    ### START CODE HERE ###
    V = eigenvecs[:,0:k]
    Xred = X @ V
    ### END CODE HERE ###
    return Xred
```

Try out this function, reducing your data to just two components

%%
尝试执行这个函数，将你的数据降到两个成分。
%%

```python
Xred2 = perform_PCA(X, eigenvecs,2)
print(f'Xred2 shape: {Xred2.shape}')
```

    Xred2 shape: (55, 2)
<a name='2.5'></a>
### 2.5 Analyzing the dimensionality reduction in 2 dimensions

One cool thing about reducing your data to just two components is that you can clearly visualize each cat image on the plane. Remember that each axis on this new plane represents a linear combination of the original variables, given by the direction of the two eigenvectors.

%%
有一件很酷的事情在于，将数据降至仅两个成分，可以让你在平面上清晰地可视化每一张猫猫的图片。
请记住，在这个新平面上的每个轴都代表原始变量的线性组合，给出的方向都来自于这两个特征向量。
%%

Use the function `plot_reduced_data` in `utils` to visualize the transformed data. Each blue dot represents an image, and the number represents the index of the image. This is so you can later recover which image is which, and gain some intuition.

%%
使用这个 `utils` 包里的 `plot_reduced_data` 函数来可视化这个转换后的数据。
每个蓝色的点代表了一个图像，点上的数字代表了图像的索引。
以便你之后能够分辨出每张图片分别是什么，并建立直观的理解。
%%

```python
utils.plot_reduced_data(Xred2)
```

![C1W4_Assignment_52_0.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_52_0.png)


If two points end up being close to each other in this representation, it is expected that the original pictures should be similar as well. 
Let's see if this is true. Consider for example the images 19, 21 and 41, which appear close to each other on the top center of the plot. Plot the corresponding cat images vertfy that they correspond to similar cats. 

%%
如果在这个表示（图）中，如果两个点最终彼此靠近，那么相应的原始图像也应具有相似性。
让我们看一看。
考虑样本图片 19，21 和 41，它们在图像的**横轴**中间靠右位置彼此靠近。
绘制相应的猫咪图片，验证它们是否对应相似的猫咪。
%%


```python
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(imgs[19], cmap='gray')
ax[0].set_title('Image 19')
ax[1].imshow(imgs[21], cmap='gray')
ax[1].set_title('Image 21')
ax[2].imshow(imgs[41], cmap='gray')
ax[2].set_title('Image 41')
plt.suptitle('Similar cats')
```

    Text(0.5, 0.98, 'Similar cats')

![C1W4_Assignment_54_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_54_1.png)


As you can see, all three cats have white snouts and black fur around the eyes, making them pretty similar.

%%
可以看到，这三只猫的口鼻都是白色的，眼周覆盖着黑色的毛发，这让它们看起来非常相似。
%%

Now, let's choose three images that seem far appart from each other, for example image 18, on the middle right, 41 on the top center and 51 on the lower left, and also plot the images

%%
现在我们选择三张相隔甚远的图片，比如 18 号，它在正中间偏右，41 号在中间偏上，51 号则偏左，然后我们同样绘制这个图像。
%%


```python
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(imgs[18], cmap='gray')
ax[0].set_title('Image 18')
ax[1].imshow(imgs[41], cmap='gray')
ax[1].set_title('Image 41')
ax[2].imshow(imgs[51], cmap='gray')
ax[2].set_title('Image 51')
plt.suptitle('Different cats')
```

    Text(0.5, 0.98, 'Different cats')

![C1W4_Assignment_56_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_56_1.png)




In this case, all three cats look really different, one being completely black, another completely white, and the the third one a mix of both colors.

%%
在这种情况下，三只猫看起来截然不同，一只纯黑，一只纯白，还有一只则是黑白相间。
%%

Feel free to choose different pairs of points and check how similar (or different) the pictures are. 

%%
请随意选择不同的点对，并查看这些图片的相似度（或差异）如何。
%%
<a name='2.6'></a>
### 2.6 Reconstructing the images from the eigenvectors

When you compress the images using PCA, you are losing some information because you are using fewer variables to represent each observation. 

%%
如果你用 PCA 压缩了图像，你会丢失一些信息，因为您是以更少的变量来表征每个观测样本。
%%

A natural question arises: how many components do you need to get a good reconstruction of the image? Of course, what determines a "good" reconstruction might depend on the application.

%%
自然而然出现了另外一个问题：你需要多少成分来重建这个图像？当然，"良好"的重建标准会根据应用场景的不同而改变。
%%

A cool thing is that with a simple dot product you can transform the data after applying PCA back to the original space. This means that you can reconstruct the original image from the transformed space and check how distorted it looks based on the number of components you kept.

%%
一个很酷的事情是，通过简单的点积操作，你可以在应用主成分分析（PCA）后将数据转换回原始空间。
这意味着你可以从变换后的空间重建原始图像，并根据保留的组件数量检查图像的失真程度。
%%

Suppose you obtained the matrix $X_{red}$ by keeping just two eigenvectors, then $X_{red} = \mathrm{X}\underbrace{\left[v_1\  v_2\right]}_{\boldsymbol{V_2}}$.
%%
假设你仅仅只保留了两个特征向量，从而获得的这个矩阵 $X_{red}$，那么 $X_{red} = \mathrm{X}\underbrace{\left[v_1\  v_2\right]}_{\boldsymbol{V_2}}$.
%%

To transform the images back to the original variables space all you need to do is take the dot product between $X_{red}$ and $\boldsymbol{V_2}^T$. If you were to keep more components, say $k$, then simply replace $\boldsymbol{V_2}$ by $\boldsymbol{V_k} = \left[v_1\ v_2\ \ldots\ v_k\right]$. Notice that you can't make any combination you like, if you reduced the original data to just $k$ components, then the recovery must consider only the first $k$ eigenvectors, otherwise you will not be able to perform the matrix multiplication.

%%
将图像转换回原始变量空间你需要对 $X_{red}$ 和 $V_{2}^T$ 做点积运算。
如果你保留了更多的成分，比如 $k$ 个，那么也是简单的将 $V_2$ 替换为 $\boldsymbol{V_k} = \left[v_1\ v_2\ \ldots\ v_k\right]$。
需要注意的是，你不能进行任意组合；如果你已将原始数据降维到只有 k 个成分，那么恢复过程必须仅考虑前 k 个特征向量，否则将无法执行矩阵乘法。
%%

In the next cell you will define a function that given the transformed data $X_{red}$ and the matrix of eigenvectors returns the recovered image. 

%%
下一个单元你将定义一个函数，该函数接收一个变换后的数据 $X_{red}$ 和一个用于恢复图片的特征矩阵。
%%

```python
def reconstruct_image(Xred, eigenvecs):
    X_reconstructed = Xred.dot(eigenvecs[:,:Xred.shape[1]].T)

    return X_reconstructed
```

Let's see what the reconstructed image looks like for different number of principal components

%%
让我们看看在不同主成分数量下，重构图像会是什么样子。
%%


```python
Xred1 = perform_PCA(X, eigenvecs,1) # reduce dimensions to 1 component
Xred5 = perform_PCA(X, eigenvecs, 5) # reduce dimensions to 5 components
Xred10 = perform_PCA(X, eigenvecs, 10) # reduce dimensions to 10 components
Xred20 = perform_PCA(X, eigenvecs, 20) # reduce dimensions to 20 components
Xred30 = perform_PCA(X, eigenvecs, 30) # reduce dimensions to 30 components
Xrec1 = reconstruct_image(Xred1, eigenvecs) # reconstruct image from 1 component
Xrec5 = reconstruct_image(Xred5, eigenvecs) # reconstruct image from 5 components
Xrec10 = reconstruct_image(Xred10, eigenvecs) # reconstruct image from 10 components
Xrec20 = reconstruct_image(Xred20, eigenvecs) # reconstruct image from 20 components
Xrec30 = reconstruct_image(Xred30, eigenvecs) # reconstruct image from 30 components

fig, ax = plt.subplots(2,3, figsize=(22,15))
ax[0,0].imshow(imgs[21], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(Xrec1[21].reshape(height,width), cmap='gray')
ax[0,1].set_title('reconstructed from 1 components', size=20)
ax[0,2].imshow(Xrec5[21].reshape(height,width), cmap='gray')
ax[0,2].set_title('reconstructed from 5 components', size=20)
ax[1,0].imshow(Xrec10[21].reshape(height,width), cmap='gray')
ax[1,0].set_title('reconstructed from 10 components', size=20)
ax[1,1].imshow(Xrec20[21].reshape(height,width), cmap='gray')
ax[1,1].set_title('reconstructed from 20 components', size=20)
ax[1,2].imshow(Xrec30[21].reshape(height,width), cmap='gray')
ax[1,2].set_title('reconstructed from 30 components', size=20)

```

    Text(0.5, 1.0, 'reconstructed from 30 components')


![C1W4_Assignment_60_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_60_1.png)


As you can see, as the number of components increases, the reconstructed image looks more and more as the original one. Even with as little as 1 component you can are least identify where the relevant features such as eyes and nose are located. 

%%
正如您所见，随着成分数量的增加，重建图像与原始图像越来越相似。
即便仅用一个成分，您也能大致辨别出眼睛、鼻子等关键特征所在的位置。
%%

What happens when you consider all of the 55 eigenvectors associated to non-zero eigenvalues? Go ahead and experiment with different number of principal components and see what happens.

%%
当你考虑到所有与非零特征值相关的 55 个特征向量时，会发生什么？不妨动手实验一下，选择不同数量的主成分，看看会得到什么结果。
%%
<a name='2.7'></a>
### 2.7 Explained variance

When deciding how many components to use for the dimensionality reduction, one good criteria to consider is the explained variance. 

%%
确定降维时使用多少个成分时，一个很好的考量标准是解释方差。
%%

The explained variance is measure of how much variation in a dataset can be attributed to each of the principal components (eigenvectors). In other words, it tells us how much of the total variance is “explained” by each component. 

%%
解释方差衡量的是数据集中有多少变异可以归因于每个主成分（特征向量）。
换句话说，它告诉我们每个成分能够“解释”总方差的比例。
%%

In PCA, the first principal component, i.e. the eigenvector associated to the largest eigenvalue, is the one with greatest explained variance. As you might remember from the lectures, the goal of PCA is to reduce the dimensionality by projecting data in the directions with biggest variability.

%%
在 PCA 中，第一个主成分，也就是关联最大的特征值的特征向量，是解释方差最大的那个。
也许你还记得讲座中提到过过，PCA 的目标是通过将数据投影到变化最大的方向上，以降低数据的维度。
%%

In practical terms, the explained variance of a principal component is the ratio between its associated eigenvalue and the sum of all the eigenvalues. So, for our example, if you want the explained variance of the first principal component you will need to do $\frac{\lambda_1}{\sum_{i=1}^{55} \lambda_i}$

%%
实际上，一个主成分的**解释方差**是其对应的特征值与所有特征值之和的比率。
所以，就我们的例子而言，如果你想要求得第一个主成分的解释方差你需要这样计算：
$$
\frac{\lambda_1}{\sum_{i=1}^{55} \lambda_i}
$$
%%

Next, let's plot the explained variance of each of the 55 principal components, or eigenvectors. Don't worry about the fact that you only computed 55 eigenvalue-eigenvector pairs, recall that all the remaining eigenvalues of the covariance matrix are zero, and thus won't add enything to the explained variance.

%%
让我们绘制关于这 55 个主成分的解释方差，或者叫它特征向量。
不必担心，我们只计算 55 对特征值-特征向量，请记住，协方差矩阵所有剩余的特征值都为零，因此它们不会对解释方差贡献任何内容。
%%

```python
explained_variance = eigenvals/sum(eigenvals)
plt.plot(np.arange(1,56), explained_variance)
```

    [<matplotlib.lines.Line2D at 0x7f4d1072f5f0>]

![C1W4_Assignment_63_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_63_1.png)


As you can see, the explained variance falls pretty fast, and is very small after the 20th component.

%%
正如你所见到的，解释方程下降得非常快，然后在 20 个成分之后就很小了。
%%

A good way to decide on the number of components is to keep the ones that explain a very high percentage of the variance, for example 95%. 

%%
确定主成分数量的一个好方法是，保留哪些能解释非常搞比利方差的部分，比如 95%。
%%

For an easier visualization you can plot the cumulative explained variance. You can do this with the `np.cumsum` function. Let's see what this looks like

%%
为便于直观理解，您可以绘制累积解释方差图。
你可以使用 `np.cumsum` 函数，让我们看看结果。
%%


```python
explained_cum_variance = np.cumsum(explained_variance)
plt.plot(np.arange(1,56), explained_cum_variance)
plt.axhline(y=0.95, color='r')
```

    <matplotlib.lines.Line2D at 0x7f4d0dbd5610>

![C1W4_Assignment_65_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_65_1.png)



In red you can see the 95% line. This means that if you want to be able to explain 95% of the variance of your data you need to keep 35 principal components. 

%%
图中红线表示 95%的水平。这意味着，若想解释数据 95%的方差，您需要保留 35 个主成分。
%%

Let's see how some of the original images look after the reconstruction when using 35 principal components 

%%
让我们看看一部分保留 35 个主成分对图片进行重建后，和原始图像比较会有什么效果。
%%

```python
Xred35 = perform_PCA(X, eigenvecs, 35) # reduce dimensions to 35 components
Xrec35 = reconstruct_image(Xred35, eigenvecs) # reconstruct image from 35 components

fig, ax = plt.subplots(4,2, figsize=(15,28))
ax[0,0].imshow(imgs[0], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(Xrec35[0].reshape(height, width), cmap='gray')
ax[0,1].set_title('Reconstructed', size=20)

ax[1,0].imshow(imgs[15], cmap='gray')
ax[1,0].set_title('original', size=20)
ax[1,1].imshow(Xrec35[15].reshape(height, width), cmap='gray')
ax[1,1].set_title('Reconstructed', size=20)

ax[2,0].imshow(imgs[32], cmap='gray')
ax[2,0].set_title('original', size=20)
ax[2,1].imshow(Xrec35[32].reshape(height, width), cmap='gray')
ax[2,1].set_title('Reconstructed', size=20)

ax[3,0].imshow(imgs[54], cmap='gray')
ax[3,0].set_title('original', size=20)
ax[3,1].imshow(Xrec35[54].reshape(height, width), cmap='gray')
ax[3,1].set_title('Reconstructed', size=20)

```

    Text(0.5, 1.0, 'Reconstructed')

![C1W4_Assignment_67_1.png](https://obsidian-image.wwtt.xyz/2025/11/C1W4_Assignment_67_1.png)



Most of these reconstructions look pretty good, and you were able to save a lot of memory by reducing the data from 4096 variables to just 35!

%%
大多数图像重建之后看上去不错，通过将数据从 4096 个变量减少到仅仅 35 个，你成功节省了大量内存！
%%

Now that you understand how the explained variance works you can play around with different amount of explained variance and see how this affects the reconstructed images. You can also explore how the reconstruction for different images looks. 

%%
既然您已经理解了解释方差的工作原理，您就可以尝试调整不同程度的解释方差，并观察这如何影响重构图像。
您还可以探索不同图像的重构效果如何。
%%

As you can see, PCA is a really useful tool for dimensionality reduction. In this assignment you saw how it works on images, but you can apply the same principle to any tabular dataset. 

%%
显而易见，主成分分析（PCA）是一种极为高效且实用的降维工具。
在本次实践中，你已了解其在图像处理中的应用，但这一核心原理同样适用于任何表格形式的数据集。
%%

Congratulations! You have finished the assignment in this week.
