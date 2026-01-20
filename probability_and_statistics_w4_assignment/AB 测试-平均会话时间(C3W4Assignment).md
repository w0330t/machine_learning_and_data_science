---
aliases: [AB Testing - Average Session Duration]
tags: []
created: 2026-01-18, 20:13:01
modified: 2026-01-20, 19:09:18
---

# AB Testing - Average Session Duration

Welcome! In this assignment you will be presented with two cases that require an AB test to choose an action to improve an existing product. You will perform AB test for a continuous and a proportion metric. For this you will define functions that estimate the relevant information out of the samples, compute the relevant statistic given each case and take a decision on whether to (or not) reject the null hypothesis.

%% 欢迎！本次任务将向你展示两个案例，需要通过 AB 测试来选择行动方案，以优化现有产品。你将执行 AB 测试执行一个连续指标和一个比例指标。为此，您需要定义一些函数，这些函数可以从样本中估计相关信息，计算每种情况下的相关统计数据，并决定是否拒绝原假设。 %%

Let's get started!

# Outline

- [ 1 - Introduction](#1)
- [ 2 - Exploring and handling the data](#2)
- [ 3 - Revisiting the theory](#3)
- [ 4 - Step by step computation](#4)
  - [ Exercise 1](#ex01)
  - [ Exercise 2](#ex02)
  - [ Exercise 3](#ex03)
  - [ Exercise 4](#ex04)
  - [ Exercise 5](#ex05)



```python
import math
import numpy as np
import pandas as pd
from scipy import stats
```

## 1 - Introduction

Suppose you have a website that provides machine learning content in a blog-like format. Recently you saw an article claiming that similar websites could improve their engagement by simply using a specific color palette for the background. Since this change seems pretty easy to implement you decide to run an AB test to see if this change does in fact drive your users to stay more time in your website.

%% 假设你有一个类似博客的网站，提供机器学习的内容。最近你看到一篇文章，声称类似的网站可以通过简单地使用特定的背景色调来提高用户的订阅量。由于这个改动的实现看上去非常简单，你决定运行 AB 测试看看这个改动是否能让你的用户花更多的时间在你的网站上。 %%

The metric you decide to evaluate is the `average session duration`, which measures how much time on average your users are spending on your website. This metric currently has a value of 30.87 minutes.

%% 你决定评估的指标是“平均会话时长”（`average session duration`）。即测量你的用户平均花了多少时间在你的网站上。目前这个指标的值为 30.87 分钟。 %%

Without further considerations you decide to run the test for 20 days by randomly splitting your users into two segments:

 %% 无需多言，你决定运行这个为期 20 天的测试，你将用户随机分成了两组： %% 
 
- `control`: These users will keep seeing your original website.
- `variation`: These users will see your website with the new background colors.
%% 
- `control`：这些用户继续浏览你原有的网站。
- `variation`：这些用户浏览你的网站将看见新的背景颜色。
 %%

## 2 - Exploring and handling the data

Run the next cell to load the data from the test:

%% 运行下面的单元格读取测试数据 %%

```python
# Load the data from the test using pd.read_csv
data = pd.read_csv("background_color_experiment.csv")

# Print the first 10 rows
data.head(10)
```

> [!result]
> |     | user_id    | user_type | session_duration |
> | --- | ---------- | --------- | ---------------- |
> | 0   | BM3C0BJ7CS | variation | 15.528769        |
> | 1   | MJWN6XNH6L | variation | 32.287590        |
> | 2   | 46ZPHHABLS | variation | 43.718217        |
> | 3   | OHA298DHUG | variation | 49.519702        |
> | 4   | AKJ77X6F4A | control   | 61.709028        |
> | 5   | BFNWMGU6DX | variation | 71.779283        |
> | 6   | UFO2V8ZKFB | variation | 23.291835        |
> | 7   | 4CEIM3VRS9 | control   | 25.219461        |
> | 8   | 90AGF68FF8 | control   | 26.240482        |
> | 9   | R3DQFO6068 | variation | 20.780244        |

```python
print(f"The dataset size is: {len(data)}")
```

> [!result]
> 	The dataset size is: 4186

The data shows for every user the average session duration and the version of the website they interacted with. To separate both segments for easier computations you can slice the Pandas dataframe by running the following cell. You may want to revisit our ungraded labs on Pandas if you are still not familiar with it. However, no need to worry because you don't need to be a Pandas expert to complete this assignment.

%% 这个数据显示了每个用户的评论回话时常，还有他们交互的版本。使用 Pandas 的切片功能可以简单的将数据分开。如果你仍然不熟悉它可以看看之前的相关实验。无论如何，不用担心，你不需要精通 Pandas 就可以完成这次作业。 %%


```python
# Separate the data from the two groups (sd stands for session duration)
control_sd_data = data[data["user_type"]=="control"]["session_duration"]
variation_sd_data = data[data["user_type"]=="variation"]["session_duration"]

print(f"{len(control_sd_data)} users saw the original website with an average duration of {control_sd_data.mean():.2f} minutes\n")
print(f"{len(variation_sd_data)} users saw the new website with an average duration of {variation_sd_data.mean():.2f} minutes")
```

> [!result]
> 	2069 users saw the original website with an average duration of 32.92 minutes
> 
> 	2117 users saw the new website with an average duration of 33.83 minutes

Notice that the split is not perfectly balanced. This is common in AB testing as there is randomness associated with the way the users are assigned to each group. 

%% 注意，切分后的两组数据量并不是相等的。这在 AB 测试中很常见，因为用户被分配到每个组的方式具有随机性。 %%

At first glance it looks like the change to the background did in fact drive users to stay longer on your website. However you know better than driving conclusions at face value out of this data so you decide to perform a hypothesis test to know if there is a significant difference between the **means** of these two segments. 

%% 乍一看，改变背景色后用户停留得更久。然而你不能仅凭数据表面就下结论，所以你决定进行假设检验，以确定这两个细分市场的均值之间是否存在显著差异。 %%

## 3 - Revisiting the theory

Let's revisit the theory you saw in the lectures and apply it to this problem. If you are confident with the theory and you feel that you don't need a revision, you may skip this section direct to 1.4!

%% 让我们复习一下课堂上学习的理论知识以应用在这个问题上。如果你对理论知识有自信，如果你对理论有信心，觉得不需要复习，可以直接跳过这部分，直接进入 1.4 部分！ %%

Remember that your job is to measure if changing the website's background color leads to an increase of the time visitors spend on it. Rewriting this as hypothesis test, the **null hypothesis** is that the change did not affect the time a visitor spend. Let's name the variables:

%% 请记住你的工作是评估网站改变背景色是否会增加用户在网站中停留的时间。那么假设检验中，零假设就是对用户的停留时间没有影响。让我们定义变量。 %%

- $\mu_c$ is the average time a user **in the control group** spend in the website. Recall that the **control group** is the group accessing the website without the change in the background color.
- $\mu_v$ is the average time a user **in the variation groups** spend in the website. Recall that the **variation group** is the groups accessing the website **with the updated background color**.

%%
- $\mu_c$ 是控制组在网站花费的平均时间。请记住，控制组指的是那些访问背景颜色未改变的网站的群体。
- $\mu_v$ 是变化组在网站花费的平均时间。请记住，变化组是指那些访问更新网站背景色的群体。
%%

Also, recall that your intention is to measure if the background color leads to an **increase** in the time a visitor spend in the website. So writing this experiment as a hypothesis test, the **null hypothesis** is then $H_0: \mu_c = \mu_v$ and the **alternative hypothesis** is $H_1: \mu_v > \mu_c$, or equivalently, $H_1: \mu_v - \mu_c > 0$. 

%% 另外请记住，你的目的是衡量在背景颜色改变后增加访问者在网站上花费的时间。所以假设检验的条件为：零假设是 $H_0: \mu_c = \mu_v$，备择假设则是 $H_1: \mu_v > \mu_c$，或者写作 $H_1: \mu_v - \mu_c > 0$ %%

Therefore, the hypothesis you will test is:

%% 因此，你将检验的假设是： %%

$$H_0: \mu_v = \mu_c \quad \text{vs.} \quad H_1: \mu_v - \mu_c > 0$$

As you saw in the lectures, this is a **right-tailed** test, as you are looking for an increase in the average time. As you saw above, you have more than 2000 users per group, this is a great amount of data so it is reasonable to rely in the Central Limit Theorem that the **average time** for each group follows a normal distribution. Remember that this result is for the group **average time** altogether and not that the time each user spend follows a normal distribution. You don't know the exact distribution for the amount of time a user spend in your website, however, the CLT assures that if we gather enough data, their average time will be very close to a normal distribution whose mean is the average time a user spend in the website. Let's then define two new quantities:

%% 正如你在讲桌中看到的那样，这是一个右尾测试，鉴于你希望提高平均时间。正如你上面看到的，每个组有超过 2000 的用户，这是一个庞大的数据，我们可以合理的依赖中心极限定理，每组的平均时间遵从正态分布。你不知道用户在你的网站上所花的时间真实的分布会怎么样，无论如何，只要数据足够，中心极限定理可以确保平均时间非常接近正态分布，该正态分布的均值是用户在你的网站上花费的平均时间。 %%

- $\overline{X}_c$ - the control group **sample mean**.
- $\overline{X}_v$ - the variation group **sample mean**.
- $n_c$ - the control group **size**.
- $n_v$ - the variation group **size**.

%% 
- $\overline{X}_c$ - 控制组的样本均值。
- $\overline{X}_v$ - 变化组的样本均值。
- $n_c$ - 控制组的样本数量。
- $n_v$ - 变化组的样本数量。
 %%

So, by the Central Limit Theorem, you may suppose that

%% 所以，根据中心极限定理，你可以假设为： %%
- $$\overline{X}_c \sim N\left(\mu_c, \left(\frac{\sigma_c}{\sqrt{n_c}}\right)^2\right)$$
- $$\overline{X}_v \sim N\left(\mu_v, \left(\frac{\sigma_v}{\sqrt{n_v}}\right)^2\right)$$

Note that with our assumptions of normality, $\overline{X}_v - \overline{X}_c$ also follows a normal distribution. So, if $H_0$ is true, then $\mu_c = \mu_v$ and $\mu_v - \mu_c = 0$, therefore:

%% 注意，根据我们的正态假设，$\overline{X}_v - \overline{X}_c$ 也应该服从一个正态分布。所以如果 $H_0$ 为真，同时 $\mu_c = \mu_v$，即 $\mu_v - \mu_c =0$，因此： %%

$$\overline{X}_c - \overline{X}_v \sim N\left(\mu_v - \mu_c, \left(\dfrac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\dfrac{\sigma_c}{\sqrt{n_c}}\right)^2\right) = N\left(0, \left(\dfrac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\dfrac{\sigma_c}{\sqrt{n_c}}\right)^2\right)$$

Or, equivalently:

%% 或者等同于： %%

$$\frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\frac{\sigma_c}{\sqrt{n_c}}\right)^2}} \sim N(0, 1)$$

However, remember that **you don't know the exact values for** $\sigma_v$ and $\sigma_c$, as they are the **population standard deviation** and you are working with a sample, so the best you can do is compute the **sample standard deviation**. So you must replace $\sigma_c$ and $\sigma_v$ by the sample standard deviation, respectively, $s_c$ and $s_v$. You also saw in the lectures that replacing the population standard deviation by the sample standard deviation changes the random variable from a Normal to a t-student:

%% 无论如何请记住，你不知道 $\sigma_v$ 和 $\sigma_c$ 精确的值，因为它们是总体标准差，而你处理的是样本。所以最好的办法是计算样本标准差。所以你必须使用样本标准差 $s_c$ 和 $s_v$ 分别替代 $\sigma_c$ 和 $\sigma_v$。你同样在课程中学习过，用样本标准差替换总体标准差，会使得随机变量从正态分布转变为 t 分布。 %%

$$t = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{s_v}{\sqrt{n_v}}\right)^2 + \left(\frac{s_c}{\sqrt{n_c}}\right)^2}} \sim t_d$$

Where $d$ is the **degrees of freedom** for this scenario. If we suppose that both groups have the same standard deviation, then $d = n_c + n_v - 2$, however there is no argument supporting this supposition, so the formula for the degrees of freedom gets a bit messier:

%% 其中 $d$ 是这个场景中的自由度，如果我们假设两组数据都有相同的标准差，那么自由度 $d = n_c + n_v - 2$，当然，没有任何论据支持这个假设，所以这个关于自由度的式子看上去有些混乱。 %%
$$d = \frac{\left[\frac{s_{v}^2}{n_v} + \frac{s_{c}^2}{n_c} \right]^2}{\frac{(s_{v}^2/n_v)^2}{n_v-1} + \frac{(s_{c}^2/n_c)^2}{n_c-1}}$$

Once you get the actual value for $t_d$ the, with a given significance level $\alpha$, you can decide if this value falls within the range of values that are likely to occur in the $t$ -student distribution (where 'likely' is related with your significance level). To perform this step you must find the value $p$ such that 

%% 一旦你得到了 $t_d$ 的实际值，还有给定的显著水平 $\alpha$，你可以判断该数值是否可能落在 t-分布的可能值范围内（“可能”与您的显著性水平相关）。为了执行这一步你必须找到 p 值： %%

$$p = P(t_d > t | H_0)$$

If this value is less than your significance level $\alpha$, then you **reject the null hypothesis**, because it means that you observed a value that is very unlikely to occur (unlikely here means that is less than the significance level you have set) if $H_0$ is true.

%% 如果小于你的显著性水平 $\alpha$，就可以拒绝零假设，因为这意味着，如果零假设成立，您观测到的这个数值是极不可能发生的（这里极不可能是指该事件发生的概率低于您设定的显著性水平）。 %%

Also, remember that $P(t_d \leq t)$ is the $\text{CDF}$ (cumulative distribution function) for the $t$ -student distribution with $d$ degrees of freedom in the point $x = t$, so to compute $P(t_d > t)$ you may compute:

%% 此外，请记住，$P(t_d \leq t)$ 是自由度为 $d$ 的 t 分布在点 $x = t$ 上的累积分布函数（CDF），因此要计算 $P(t_d > t)$，您可以通过以下方式计算： %%

$$P(t_d > t) = 1 - \text{CDF}_{t_d}(t)$$

Since $P(t_d \leq t) + P(t_d > t) = 1$

%% 因为 $P(t_d \leq t) + P(t_d > t) = 1$ 是成立的 %%


## 4 - Step by step computation


Wrapping up everything discussed above:

%% 对上面的所有进行总结： %%

The hypothesis test is given by:

%% 假定检验给出了： %%

$$H_0: \mu_v = \mu_c \quad \text{vs.} \quad H_1: \mu_v - \mu_c > 0$$

You will start computing:

- $n_c$ and $n_v$, the control and variation group sizes, respectively.
- $\overline{X}_c$ and $\overline{X}_v$, the average time spent by the users in the control and variation group, respectively. 
- $s_c$ and $s_v$, the **sample** standard deviation for the time spend by the users in the control and variation group, respectively.

%% 
你需要计算：

- $n_c$ 和 $n_v$，分别计算控制组和变化组的人数。
- $\overline{X}_c$ 和 $\overline{X}_v$，分别计算控制组和变化组用户的平均花费时间。
- $s_c$ 和 $s_v$，分别计算控制组和变化组用户花费时间的样本标准差。
%%

With these quantities in hand, the next steps are to compute:

- $d$, the degrees of freedom of the $t$ -student distribution, $t_d$.
- The $t$ -value, which it will be called $t$.
- The $p$ value for the distribution $t_d$ for the $t$ -value, i.e., the value  $p = P(t_d > t | H_0)$.

%% 
有了这些数据，下一步计算：

- $d$，t 分布的自由度 $t_d$
- $t$ 值，这里直接称为 $t$
- $p$ 值，t 分布的 t 值对应的 p 值，即  $p = P(t_d > t | H_0)$
 %%

Finally, for a given significance level $\alpha$, you will be able to decide if you reject or not $H_0$, depending on wether $p \leq \alpha$ or not.

%% 最终，给定显著性水平 $\alpha$，根据 $p$ 是否小于等于 $\alpha$，你可以决定是否拒绝零假设。 %%

Let's get your hands into work now! Run the cell below to retrieve the session times for the control and variation groups.

%% 让我们开始动手吧！运行下面的单元格取回控制组和变化组的会话时间。 %%

```python
# X_c stores the session tome for the control group and X_v, for the variation group. 
X_c = control_sd_data.to_numpy()
X_v = variation_sd_data.to_numpy()
```

```python
print(f"The first 10 entries for X_c are:\n{X_c[:20]}\n")
print(f"The first 10 entries for X_v are:\n{X_v[:20]}\n")
```

> [!result]
> 	The first 10 entries for X_c are:
> 	[ 61.70902753  25.21946052  26.2404824   58.7480264  137.03680289
> 	  19.92148102  18.8252202   75.25179496  38.27213776  29.17104128
> 	  15.69643672  37.83860271  30.06843075  21.00318148  86.19711927
> 	  46.96997965  46.47776713  14.83464105  17.70441365  26.44693676]
> 	
> 	The first 10 entries for X_v are:
> 	[15.52876878 32.28759003 43.7182168  49.51970242 71.77928343 23.29183517
> 	 20.78024375 36.44129464 48.75034676 16.5952978  44.49566616 26.67006134
> 	 34.43667579 20.72109411 19.60185277 41.74218978 19.74485294 32.62018094
> 	 44.99513901 70.8916231 ]
### Exercise 1

In this exercise, you will write a function to retrieve the basic statistics for `X_c` and `X_d`. In other words, this function will compute, for a given numpy array:

%% 在这个练习中，你将写出一个函数用于取得基础统计数据 `X_c` and `X_d`，换句话说，对于给定的 Numpy 数组，这个函数将计算： %%

- Its size (in your case, $n_c$ and $n_v$).
- Its mean (in your case, $\overline{X}_c$ and $\overline{X}_v$)
- Its sample standard deviation(in your case, $s_c$ and $s_v$)

%% 
- 它的尺寸（即 $n_c$ 和 $n_v$）
- 它的均值（即 $\overline{X}_c$ 和 $\overline{X}_v$）
- 它的样本标准差（即 $s_c$ and $s_v$）
%%

This function inputs a numpy array and outputs a tuple in the form `(n, x, s)` where `n` is the numpy array size, `x` is its mean and `s` is its **sample** standard deviation.

这个函数输入一个 Numpy 数组，输出一个元组，里面有 `(n, x, s)`，其中 `n` 是 Numpy 的尺寸，`x` 是均值，`s` 为样本标准差。

Hint: 
- Recall that the sample standard deviation is computed by replacing $N$ by $N-1$ in the variance formula. 
- You may compute an array size using the `len` function.
- Any array in numpy has a method called `.mean()` to compute its mean.
- Any array in numpy has a method called `.std()` to compute the standard deviation and a parameter called `ddof` where if you pass `ddof = 1`, it will use $N-1$ instead of $N$. 

提示：
- 回顾一下，样本标准差的计算是用 N-1 替代 N 的。
- 获取数组尺寸可以使用 `len` 函数。
- 在 NumPy 中，任何数组都有一个名为 `.mean()` 的方法计算均值。
- 在 NumPy中，任何数组都有一个名为.std()的方法用于计算标准差，以及一个名为ddof的参数，当传入ddof=1时，它将使用N-1而非N。


```python
def get_stats(X):
    """
    Calculate basic statistics of a given data set.

    Parameters:
    X (numpy.array): Input data.

    Returns:
    tuple: A tuple containing:
        - n (int): Number of elements in the data set.
        - x (float): Mean of the data set.
        - s (float): Sample standard deviation of the data set.
    """

    ### START CODE HERE ###

    # Get the group size
    n = len(X)
    # Get the group mean
    x = X.mean()
    # Get the group sample standard deviation (do not forget to pass the parameter ddof if using the method .std)
    s = X.std(ddof=1)

    ### END CODE HERE ###

    return (n,x,s)
```


```python
n_c, x_c, s_c = get_stats(X_c)
n_v, x_v, s_v = get_stats(X_v)
```


```python
print(f"For X_c:\n\tn_c = {n_c}, x_c = {x_c:.2f}, s_c = {s_c:.2f} ")
print(f"For X_v:\n\tn_v = {n_v}, x_v = {x_v:.2f}, s_v = {s_v:.2f} ")
```

> [!result]
> 	For X_c:
> 		n_c = 2069, x_c = 32.92, s_c = 17.54 
> 	For X_v:
> 		n_v = 2117, x_v = 33.83, s_v = 18.24


### Exercise 2

In this exercise you will implement a function to compute $d$, the degrees of freedom for the $t$ -student distribution. It is given by the following formula:

%% 在这个练习中你将实现计算 $d$ 的函数，即 t 分布的自由度。下面给出了公式： %%

$$d = \frac{\left[\frac{s_{c}^2}{n_c} + \frac{s_{v}^2}{n_v} \right]^2}{\frac{(s_{c}^2/n_c)^2}{n_c-1} + \frac{(s_{v}^2/n_v)^2}{n_v-1}}$$

Hint: You may use the syntax `x**2` to square a number in python, or you may use the function `np.square`. The latter may help to keep your code cleaner. Pay attention in the parenthesis as they will indicate the order that Python will perform the computation!

%% 提示：在 Python 中，你可以使用语法 `x**2` 来计算数字的平方，或者你也可以使用函数 `np.square`。这些符号可以帮助你维持代码整洁。请注意括号，因为它们将指示 Python 执行计算的顺序！ %%


```python
def degrees_of_freedom(n_v, s_v, n_c, s_c):
    """Computes the degrees of freedom for two samples.

    Args:
        control_metrics (estimation_metrics_cont): The metrics for the control sample.
        variation_metrics (estimation_metrics_cont): The metrics for the variation sample.

    Returns:
        numpy.float: The degrees of freedom.
    """

    ### START CODE HERE ###

    # To make the code clean, let's divide the numerator and the denominator.
    # Also, note that the value s_c^2/n_c and s_v^2/n_v appears both in the numerator and denominator, so let's also compute them separately

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = np.square(s_v)/n_v

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = np.square(s_c)/n_c


    # Compute the numerator in the formula given above
    numerator = np.square(s_v_n_v + s_c_n_c)

    # Compute the denominator in the formula given above. Attention that s_c_n_c and s_v_n_v appears squared here!
    # Also, remember to use parenthesis to indicate the operation order. Note that a/b+1 is different from a/(b+1).
    denominator = np.square(s_c_n_c)/(n_c-1) + np.square(s_v_n_v)/(n_v-1)

    ### END CODE HERE ###

    dof = numerator/denominator

    return dof
```

```python
d = degrees_of_freedom(n_v, s_v, n_c, s_c)
print(f"The degrees of freedom for the t-student in this scenario is: {d:.2f}")
```

> [!result]
> 	The degrees of freedom for the t-student in this scenario is: 4182.97


### Exercise 3

In this exercise, you will compute the $t$ -value, given by

%% 在这个练习中，你将计算 t 值，公式如下： %%

$$t = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{s_v}{\sqrt{n_v}}\right)^2 + \left(\frac{s_c}{\sqrt{n_c}}\right)^2}} = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\frac{s_v^2}{n_v} + \frac{s_c^2}{n_c}}}$$

Remember that you are storing $\overline{X}_c$ and $\overline{X}_v$ in the variables $x_c$ and $x_d$, respectively. 

%% 还记得你保存在变量 $x_c$ 和 $x_d$ 里的 $\overline{X}_c$ 和 $\overline{X}_v$ 吗？ %%

```python
def t_value(n_v, x_v, s_v, n_c, x_c, s_c):

    ### START CODE HERE ###

    # As you did before, let's split the numerator and denominator to make the code cleaner.
    # Also, let's compute again separately s_c^2/n_c and s_v^2/n_v.

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = np.square(s_v)/n_v

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = np.square(s_c)/n_c

    # Compute the numerator for the t-value as given in the formula above
    numerator = x_v - x_c

    # Compute the denominator for the t-value as given in the formula above. You may use np.sqrt to compute the square root.
    denominator = np.sqrt(s_v_n_v + s_c_n_c)

    ### END CODE HERE ###

    t = numerator/denominator

    return t
```


```python
t = t_value(n_v, x_v, s_v, n_c, x_c, s_c)
print(f"The t-value for this experiment is: {t:.2f}")
```

> [!result]
> 	The t-value for this experiment is: 1.64

### Exercise 4

In this exercise, you will compute the $p$ value for $t_d$, for a given significance level $\alpha$. Recall that this experiment is a right-tailed t-test, because you are investigating wether the background color change increases the time spent by users in your website or not. 

%% 在这个练习中，对于已经给定的显著水平 $\alpha$，你将根据 $t_d$ 来计算 $p$ 值。由于你调查的是背景颜色改变后用户在你网站中花费的时间是否增加，所以这个实验是右尾测试。 %%

In this experiment the $p$ -value for a significance level of $\alpha$ is given by

%% 在这个实验中，显著水平对应的 $p$ 值的公式为： %%

$$p = P(t_d > t) = 1 - \text{CDF}_{t_d}(t)$$


Hint: 
- You may use the scipy function `stats.t(df = d)` to get the $t$ -student distribution with `d` degrees of freedom. 
- To compute its CDF, you may use its method `.cdf`. 

%% 
提示：
- 您可以使用 scipy 函数 `stats.t(df = d)` 来获取具有 `d` 自由度的 t 分布。
- 你可以使用 `.cdf` 方法计算 CDF。
 %%
 
Example:

Suppose you want to compute the CDF for a $t$-student distribution with $d = 10$ degrees of freedom for a t-value of $1.21$.

%% 假设你想要计算自由度为 10 、 t 值为 1.21的 t 分布的 CDF。 %%

```python
t_10 = stats.t(df = 10)
cdf = t_10.cdf(1.21)
print(f"The CDF for the t-student distribution with 10 degrees of freedom and t-value = 1.21, or equivalently P(t_10 < 1.21) is equal to: {cdf:.2f}")
```

> [!result]
> 	The CDF for the t-student distribution with 10 degrees of freedom and t-value = 1.21, or equivalently P(t_10 < 1.21) is equal to: 0.87

This means that there is a probability of 87% that you will observe a value less than 1.21 when sampling from a $t$ -student distribution with 10 degrees of freedom.

%% 这意味着当你从自由度为 10 的 t 分布中抽样时，有 87%的概率会观测到小于 1.21 的值。 %%

Ok, now you are ready to write a function to compute the $p$ -value for the $t$ -student distribution, with $d$ degrees of freedom and a given $t$ -value.

%% 现在你已经准备好实现计算这个 $t$ 分布 $p$ 值的函数了，其中包含自由度 `d` 和给定的 `t` 值。 %%


```python
def p_value(d, t_value):

    ### START CODE HERE ###

    # Load the t-student distribution with $d$ degrees of freedom. Remember that the parameter in the stats.t is given by df.
    t_d = stats.t(df=d)

    # Compute the p-value, P(t_d > t). Remember to use the t_d.cdf with the proper adjustments as discussed above.
    p = 1 - t_d.cdf(t_value)

    ### END CODE HERE ###

    return p
```

```python
print(f"The p-value for t_15 with t-value = 1.10 is: {p_value(15, 1.10):.4f}")
print(f"The p-value for t_30 with t-value = 1.10 is: {p_value(30, 1.10):.4f}")
```

> [!result]
> 	The p-value for t_15 with t-value = 1.10 is: 0.1443
> 	The p-value for t_30 with t-value = 1.10 is: 0.1400

### Exercise 5

In this exercise you will wrap up all the functions you have built so far to decide if you accept $H_0$ or not, given a significance level of $\alpha$.

%% 在这个练习中，你将会把所有的函数进行整合，在给定显著性水平 $\alpha$ 的情况下，判断是否接受 $H_0$ 。%%

It will input both control and validation groups and it will output `Reject H_0$` or `Do not reject H_0` accordingly.

%% 它将输入控制和变化组，并相应的输出 `Reject H_0$` 或者 `Do not reject H_0`  %%

Remember that you **reject** $H_0$ if the p-value is **less than** $\alpha$. 

%% 记住，如果 p 值小于 $\alpha$，则拒绝 $H_0$ %%

```python
def make_decision(X_v, X_c, alpha = 0.05):

    ### START CODE HERE ###

    # Compute n_v, x_v and s_v
    n_v, x_v, s_v = get_stats(X_v)

    # Compute n_c, x_c and s_c
    n_c, x_c, s_c = get_stats(X_c)

    # Compute the degrees of freedom for the t-student distribution for this experiment.
    # Pay attention to the arguments order. You may look the function definition above to make sure you don't swap values.
    # Also, remember that x_c and x_v are not used in this computation
    d = degrees_of_freedom(n_v, s_v, n_c, s_c)

    # Compute the t-value
    t = t_value(n_v, x_v, s_v, n_c, x_c, s_c)

    # Compute the p-value for the t-student distribution with d degrees of freedom
    p = p_value(d, t)

    # This is the decision step. Compare p with alpha to decide about rejecting H_0 or not. 
    # Pay attention to the return value for each block to properly write the condition.

    if p < alpha:
        return 'Reject H_0'
    else:
        return 'Do not reject H_0'

    ### END CODE HERE ###
```

```python
alphas = [0.06, 0.05, 0.04, 0.01]
for alpha in alphas:
    print(f"For an alpha of {alpha} the decision is to: {make_decision(X_v, X_c, alpha = alpha)}")
```

> [!result]
> 	For an alpha of 0.06 the decision is to: Reject H_0
> 	For an alpha of 0.05 the decision is to: Do not reject H_0
> 	For an alpha of 0.04 the decision is to: Do not reject H_0
> 	For an alpha of 0.01 the decision is to: Do not reject H_0

**Congratulations on finishing this assignment!**

%% 恭喜你完成了这次作业！ %%

Now you have created all the required steps to perform an AB test for a simple scenario!

%% 现在您已经为简单场景创建了执行 A/B 测试所需的所有步骤！ %%

**This is the last assignment of the course and the specialization so give yourself a pat on the back for such a great accomplishment! Nice job!!!!**

%% 这是本课程乃至整个专业系列的最后一个作业，为自己取得如此了不起的成就鼓个掌吧！干得漂亮！ %%

