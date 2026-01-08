---
aliases: ["Lab: Central Limit Theorem"]
tags: []
created: 2026-01-06, 18:17:18
modified: 2026-01-08, 17:38:35
---

# Lab: Central Limit Theorem

Welcome! In this ungraded lab see applications of the Central Limit Theorem when working with different distributions of data. You will see how to see the theorem in action, as well as scenarios in which the theorem doesn't hold.
%%
欢迎，在这个练习中你将看到[[中心极限定理]]应用在不同分布的数据中。你将看到定理的实际作用，以及在某些情况下该定理不成立的情况。
%%
Let's get started!


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt:
from scipy import stats
from scipy.stats import norm

import utils
```

## Gaussian population

Begin with the most straightforward scenario: when your population follows a Gaussian distribution. You will generate the data for this population by using the [np.random.normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html) function. 

%%
让我们直接开始：使用 [[numpy.random.Generator.normal]] 函数来创建一个服从高斯分布的总体数据
%%

```python
mu = 10
sigma = 5

gaussian_population = np.random.normal(mu, sigma, 100_000)
```

The population has a mean of 10 and a standard deviation of 5 (since these are the true parameters you used to generate the data) and a total of 100'000 observations. You can visualize its histogram by running the next cell:
%%
总体的均值为 10，标准差为 5（这些是用于生成数据的真实参数），总共有 100000 个观察值。下面单元格创建了可以观察它的直方图。
%%

```python
sns.histplot(gaussian_population, stat="density")
plt.show()
```

> [!result]
![C3W3_UGL_Central_Limit_Theorem_5_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_5_0.png)

## Sampling from the population

Since this lab uses simulated data you could very easily use the whole population to draw conclusions from the data. For instance if you didn't know about the values of $\mu$ and $\sigma$ you could get very close estimates of the true values by computing the mean and standard deviation of the whole population like so:
%%
由于这个实验使用了数据，你可以非常简单的使用总体数据来得到结论。在实际应用中，如果你不知道 $\mu$ 和 $\sigma$ 的具体值，通过计算总体的均值和标准差，你可以得到非常接近真实值的估计：
%%

```python
gaussian_pop_mean = np.mean(gaussian_population)
gaussian_pop_std = np.std(gaussian_population)

print(f"Gaussian population has mean: {gaussian_pop_mean:.1f} and std: {gaussian_pop_std:.1f}")
```

> [!result]
> 	Gaussian population has mean: 10.0 and std: 5.0


However in real life this will most certainly not be possible and you will need to use samples that are nowhere near as big as the population to draw conclusions of the behaviour of the data.  After all, this is what statistics is all about.

%%
无论如何，在现实生活中这几乎是不可能的，你需要使用远远小于总体的样本来推断数据的行为。毕竟，这就是统计学的意义。
%%

Depending on the sampling techniques you could encounter different properties, this is where the Central Limit Theorem comes in handy. For many distributions (**but not all**) the following is true:

%%
根据你采样的方法，你可以遇到不同的特性的数据，此时中心极限定理就派上用场了。对于许多分布（不是全部），以下的说法成立：
%%

The sum or average of a large number of independent and identically distributed random variables tends to follow a normal distribution, regardless of the distribution of the individual variables themselves. This is important because the normal distribution is well-understood and allows for statistical inference and hypothesis testing.

%%
不论各变量自身的分布如何，大量独立且相同分布的随机变量的和或平均值，它们往往趋于服从正态分布。这非常重要，因为正态分布易于理解，这样便于进行统计推断和假设检验。
%%

With this in mind you need a way of averaging samples out of your population. For this the `sample_means` is defined:

%%
基于这个思想，你需要一种方法对总体数据进行平均抽样，为此定义了函数 `sample_means`：
%%

```python
def sample_means(data, sample_size):
    # Save all the means in a list
    means = []

    # For a big number of samples
    # This value does not impact the theorem but how nicely the histograms will look (more samples = better looking)
    for _ in range(10_000):
        # Get a sample of the data WITH replacement
        sample = np.random.choice(data, size=sample_size)

        # Save the mean of the sample
        means.append(np.mean(sample))

    # Return the means within a numpy array
    return np.array(means)
```

Let's break down the function above:

- You take random samples out of the population (the sampling is done with replacement, which means that once you select an element you put it back in the sampling space so you could choose a particular element more than once). This ensures that the independence condition is met.

- Compute the mean of each sample

- Save the means of each sample in a numpy array

%%
让我们来分析上面这个函数：
- 从总体中随机抽取数据，这确保了独立性。
- 计算样本均值。
- 将每次抽样的均值保存为 NumPy 数组。
%%

The theorem states that if a large enough `sample_size` is used (usually bigger than 30) then the distribution of the sample means should be Gaussian. See it in action by running the next cell:

%%
该定理指出，如果上面传入的参数 `sample_size` 足够大（一般要大于 30）。它返回的样本均值的分布应该是高斯分布。运行下面的单元格看一下。
%%

```python
# Compute the sample means
gaussian_sample_means = sample_means(gaussian_population, sample_size=5)

# Plot a histogram of the sample means
sns.histplot(gaussian_sample_means, stat="density")
plt.show()
```

> [!result]
![C3W3_UGL_Central_Limit_Theorem_11_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_11_0.png)


The distribution of the sample means looks pretty Gaussian. However this is not good enough to determine if the theorem holds, after all you used a very small `sample_size` in this example. There are various ways to check if the sample means do follow a Gaussian distribution.

%%
样本均值的分布非常“高斯”。但是这并不能很好的证明中心极限定理的成立，毕竟在这个例子中，你使用的是一个非常小的 `sample_size`。这里有多种方法可以检验样本均值是否服从高斯分布。
%%

The first one is to compute the theoretical $\mu$ and $\sigma$ of the sample means which will be denoted with the symbols $\mu_{\bar{X}}$ and $\sigma_{\bar{X}}$ respectively. These values can be computed as follows:

%%
第一种方法是计算这个样本均值 $\mu$ 和 $\sigma$ 的解析解，样本的 $\mu$ 和 $\sigma$ 分别使用符号 $\mu_{\bar{X}}$ 和 $\sigma_{\bar{X}}$ 表示。它们的值可以用下面的式子进行计算：
%%

- $\mu_{\bar{X}} = \mu$
- $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$

**Note: In this case $n$ is the size of the sample.**

%%
注意，这个例子中，$n$ 为样本的大小。
%%

And then use these values to plot a Gaussian curve with parameters $\mu_{\bar{X}}$ and $\sigma_{\bar{X}}$. If the theorem holds then the resulting distribution of the sample means should resemble this Gaussian curve. Run the next cell to include this into the plot:

%%
然后用去绘制高斯曲线，它的参数是 $\mu_{\bar{X}}$ 和 $\sigma_{\bar{X}}$ 。如果定理成立，样本均值的分布结果与绘制的高斯曲线会非常相似。运行下面的单元格进行绘图。
%%

```python
# Compute estimated mu
mu_sample_means = mu

# Compute estimated sigma
# 5 is being used because you used a sample size of 5
sigma_sample_means = sigma / np.sqrt(5)

# Define the x-range for the Gaussian curve (this is just for plotting purposes)
x_range = np.linspace(min(gaussian_sample_means), max(gaussian_sample_means), 100)

# Plot everything together
sns.histplot(gaussian_sample_means, stat="density")
plt.plot(
    x_range,
    norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
    color="black",
)
plt.show()
```


> [!result]    
![C3W3_UGL_Central_Limit_Theorem_13_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_13_0.png)

    
They look pretty similar. However you can go one step further and plot a smooth function that attempts to estimate the probability density function of the sample means through a method known as `kernel density estimation`. If this smooth function resembles the Gaussian function then you know that the distribution of the sample means is very similar to a Gaussian:

%%
它们看上去非常的相似，无论如何，你可以更进一步，通过 `kernel density estimation`（核密度估计）的方法绘制一条平滑的函数曲线，并尝试估计样本均值的概率密度函数。如果这个平滑函数和高斯函数类似，那么样本均值的分布也同样和高斯分布相似：
%%

```python
# Histogram of sample means (blue)
sns.histplot(gaussian_sample_means, stat="density", label="hist")

# Estimated PDF of sample means (red)
sns.kdeplot(
    data=gaussian_sample_means,
    color="crimson",
    label="kde",
    linestyle="dashed",
    fill=True,
)

# Gaussian curve with estimated mu and sigma (black)
plt.plot(
    x_range,
    norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
    color="black",
    label="gaussian",
)

plt.legend()
plt.show()
```


    
![C3W3_UGL_Central_Limit_Theorem_15_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_15_0.png)


Both curves look almost identical!

%%
两个曲线看上去几乎相同！
%%

Another way of checking for normality is to perform a QQ plot of the sample means. The points in this plot should resemble a straight line if the distribution of the data is Gaussian:

%%
另一种检查的方法就是用样本均值去绘制 QQ 图。如果数据分布是高斯分布，图中的点应当与直线高度重合。
%%

```python
# Create the QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(gaussian_sample_means, plot=ax, fit=True)
plt.show()
```

> [!result]
![C3W3_UGL_Central_Limit_Theorem_17_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_17_0.png)


The resulting QQ plot yields an almost perfect straight line which further confirms that the sample means do follow a Gaussian distribution.

%%
得到的 QQ 图几乎是一条完美的直线，这进一步确认样本均值遵循正态分布。
%%

Now, put everything together in an interactive widget to experiment with different values for $\mu$, $\sigma$ and `sample_size`. **To update the plots you will need to click the `Run Interact` button after changing the parameters**:

%%
现在，将所有元素整合到一个交互式小部件中，以便尝试不同的$μ$、$σ$和样本大小值。**更改参数后，您需要点击 `运行交互` 按钮来更新图表**：
%%

```python
utils.gaussian_clt()
```

> [!result]
![中心极限定理-总体为高斯分布.png|800](https://obsidian-image.wwtt.xyz/2026/01/中心极限定理-总体为高斯分布.png)


Even with very small values for `sample_size` the sample means follow a Gaussian distribution. This is actually one of the properties of the Gaussian distribution.

%%
即使 `sample_size` 非常小，样本均值也服从高斯分布。这实际上也是高斯分布的特性之一。
%%

Now test the theorem with other distributions!

%% 
接下来测试中心极限定理在其他分布上的应用！
 %%

## Binomial Population

Now try with a population distribution that is not Gaussian. One such distribution is the Binomial distribution which you already saw covered in the lectures. To generate data that follows this distribution you will need to define values for the parameters of `n` and `p`:

%% 现在尝试总体分布不为高斯分布的数据。比如你在讲座中看到的[[离散概率分布#二项分布]]。要创建二项分布的总体数据集，需要定义参数 `n` 和 `p`: %%

```python
n = 5
p = 0.8

binomial_population = np.random.binomial(n, p, 100_000)
```

The population has a total of 100'000 observations. You can visualize its histogram by running the next cell:

%% 
总体一共有十万个可观测数据。运行下面的代码绘制直方图。
 %%

```python
sns.histplot(binomial_population, stat="count")
plt.show()
```


> [!result]
![C3W3_UGL_Central_Limit_Theorem_24_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_24_0.png)


The mean and standard deviation is not as straightforward as in the Gaussian case (since these parameters were needed to generate the data in that case). However you can easily compute those values by drawing them directly from the population:


%% 这个高斯分布的均值和标准差看上去并不明显（由于需要创建这样数据需要这样的参数）。无论如何，你可以直接计算总体的结构 %%


```python
binomial_pop_mean = np.mean(binomial_population)
binomial_pop_std = np.std(binomial_population)

print(f"Gaussian population has mean: {binomial_pop_mean:.1f} and std: {binomial_pop_std:.1f}")
```

> [!result]
 > 	  Gaussian population has mean: 4.0 and std: 0.9


Once again, in real life you will not have access to the whole population so you need another method to compute this values. Actually the mean and standard deviation of binomal distributions are well defined and can be computed by using the following formulas:

%% 再说一次，在现实世界中你不可能拿到总体数据，所以你需要另外的方法计算这个值。二项分布的均值和标准差的计算公式如下： %%

- $\mu = np$
- $\sigma = \sqrt{np(1-p)}$


```python
binomial_pop_mean = n * p
binomial_pop_std = np.sqrt(n * p * (1 - p))

print(f"Gaussian population has mean: {binomial_pop_mean:.1f} and std: {binomial_pop_std:.1f}")
```

> [!result]
> 	Gaussian population has mean: 4.0 and std: 0.9


Now you have found these same values but without needing to sample the whole population. Nice!

%%现在，你没有用总体数据的采样就求得了相同的值。真棒！%%

Before seeing the theorem for this case, you should know that there is a rule of thumb to know if the theorem will hold or not for the Binomial distribution case. This condition is the following:

%%在查看中心极限定理应用于这个案例之前，你需要知道一条经验法则：该定理在二项分布下是否成立，取决于如下条件：%%

if $min(Np, N(1-p)) >= 5$ then CLT holds

where $N = n*sample\_size$

%% $Np$ 和 $N(1-p)$ 取小值，它如果大于等于 5，CLT 成立。
其中 $N$ 为试验次数 $n \times 样本大小$。%%

However, it is important to note that this rule is only a rough guideline, and other factors such as the presence of outliers and the purpose of the analysis should also be taken into consideration when choosing an appropriate statistical method.

%%无论如果，这个重要的注意事项只是一个粗糙的指导方案，在选择合适的统计方法时，还应考虑到异常值的存在及分析目的等其他因素。%%

Now check the theorem in action. Begin by using a small `sample_size`:

%% 现在来验证该定理的实际应用。首先使用一个小小的 `sample_size`：%%

```python
sample_size = 3
N = n * sample_size

condition_value = np.min([N * p, N * (1 - p)])
print(f"The condition value is: {int(condition_value*10)/10:.1f}. CLT should hold?: {True if condition_value >= 5 else False}")
```

> [!result]
> 	The condition value is: 2.9. CLT should hold?: False

Perform the sampling and compute the theoretical values for the mean and standard deviation of the sample means. Remember these latter two can be computed like so:

%%抽样并计算样本均值的理论均值和标准差。记住，这两项可以按照下面的方法计算：%%

- $\mu_{\bar{X}} = \mu$
- $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{sample\_size}}$


```python
# Compute sample means
binomial_sample_means = sample_means(binomial_population, sample_size=sample_size)

# Compute estimated mu
mu_sample_means = n * p

# Compute estimated sigma
sigma_sample_means = np.sqrt(n * p * (1 - p)) / np.sqrt(sample_size)
```

Visualize the KDE vs Gaussian curve plot and the QQ plot to see how well the theorem is holding:

%% 绘制 KDE 与高斯曲线对比图及 QQ 图，以观察该定理的吻合程度。%%

```python
# Create the plots
utils.plot_kde_and_qq(binomial_sample_means, mu_sample_means, sigma_sample_means)
```

> [!result]
![C3W3_UGL_Central_Limit_Theorem_34_0.png|800](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_34_0.png)

This doesn't look as good as with the Gaussian example. It looks that by using a small `sample_size` the sample means do not follow a Gaussian distribution.

%% 这看上去不太高斯。使用小小的 `sample_size` 导致样本均值不太服从高斯分布。 %%

Try again but now increasing the size of each sample:

%% 增加样本大小然后再尝试一次。 %%

```python
sample_size = 30
N = n * sample_size

condition_value = np.min([N * p, N * (1 - p)])
print(f"The condition value is: {int(condition_value*10)/10:.1f}. CLT should hold?: {True if condition_value >= 5 else False}")
```

> [!result]
> 	The condition value is: 29.9. CLT should hold?: True

According to the rule of thumb, the theorem should hold under these conditions. Run the next cell to check if this is true:

%% 根据这个经验法则，这种情况下定理成立。运行下面的代码验证一下： %%

```python
binomial_sample_means = sample_means(binomial_population, sample_size=sample_size)

# Compute estimated mu
mu_sample_means = n * p

# Compute estimated sigma
sigma_sample_means = np.sqrt(n * p * (1 - p)) / np.sqrt(sample_size)

# Create the plots
utils.plot_kde_and_qq(binomial_sample_means, mu_sample_means, sigma_sample_means)
```


> [!result]
![C3W3_UGL_Central_Limit_Theorem_38_0.png|800](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_38_0.png)


This time everything seems to indicate that the theorem is holding nicely!

%% 这次一切似乎都表明，定理成立，而且相当不错！ %%

As with the previous distribution, by running the next cell you will launch an interactive widget in which you can play around with different values of $n$, $p$ and $sample\_size$. 

%% 和之前的分布一样，运行下面的的代码可以启动一个小组件，你可以定义不同的 $n$, $p$ 和 $sample\_size$.  %%

See if you can find anything interesting, for instance does the theorem seem to hold better when $p$ is close to 0.5?

%% 看看你是否能找到任何有趣的分布，比如，当 p 接近 0.5 的时候，该定理更容易成立吗？ %%

```python
utils.binomial_clt()
```

> [!result]
![中心极限定理-总体为二项分布.png](https://obsidian-image.wwtt.xyz/2026/01/中心极限定理-总体为二项分布.png)


Keep on going with another distribution!

%% 继续看看其他的分布！ %%
## Poisson Population

Another popular distribution you might have heard of is the `poisson` distribution. It models the number of events occurring in a fixed interval of time or space, given the average rate of occurrence $\mu$ of those events.

%% 另一个分布你可能听说过，就是泊松分布。它（泊松分布）用于模拟**在已知平均发生率为** $μ$ 的情况下，特定时间段或空间范围内事件发生的次数。 %%

Since you are already familiar with the process of checking the theorem for a distribution you will skip all intermediate steps and jump straight to playing with the interactive widget.

%% 既然您已熟悉检查分布定理的流程，您将跳过所有中间步骤，直接开始操作交互式小工具。 %%

The only thing to consider here is that the mean and standard deviation of this distribution can be computed like this:

%% 唯一的事情是需要考虑这个分布的均值和标准差，它的计算如下： %%

- $\mu = \mu$
- $\sigma = \sqrt{\mu}$


```python
utils.poisson_clt()
```

> [!result]
![中心极限定理-总体为泊松分布.png](https://obsidian-image.wwtt.xyz/2026/01/中心极限定理-总体为泊松分布.png)

As expected, you should see that the bigger the `sample_size` the more closely the distribution of the sample means follows a Gaussian distribution.

%% 不出所料，你应该观察到，样本的容量越大，样本均值的分布越接近高斯分布。  %%

## Cauchy Distributions

The Cauchy distribution is not as well-known as the other ones seen throughout this lab. It has heavy tails, which means that the probability of observing extreme values is higher than in other distributions with similar spread. It also does not have a well-defined mean or variance, which makes it less suitable for many statistical applications.

%% 柯西分布，比起这个实验中的其他分布并不那么有名。它有一个厚尾，这意味着观察到极值的概率远远高于其他相似的分布。它同样不方便定义均值和方差，这使得它在多种统计中不太适用。 %%

As a result of the properties of this distribution, the central limit theorem does not hold. Run the next cell to generate a population of 1000 points that distribute Cauchy:

%% 鉴于该分布的性质，中心极限定理不能在这种情况下成立。运行下面的代码创建一个 1000 个数据点的柯西分布。 %%

```python
cauchy_population = np.random.standard_cauchy(1000)
```

Now take a look at the histogram of this population:

%% 现在来观察一下它的直方图： %%

```python
sns.histplot(cauchy_population, stat="density", label="hist")
plt.show()
```

> [!result]    
![C3W3_UGL_Central_Limit_Theorem_48_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_48_0.png)

It is very difficult to even see the histogram due to the extreme values it has. Now compute the sample means with a `sample_size` of 30, which is usually a safe bet for the theorem to hold under other distributions:

%% 由于极值的存在，直方图都很难看清。现在通过定义 `sample_size` 为 30 来计算样本均值，这通常是一个可靠的假设，使该定理在其他分布下成立： %%

```python
cauchy_sample_means = sample_means(cauchy_population, sample_size=30)
```

Since this distribution has an undefined mean and standard deviation and the histogram is very hard to interpret you will only create the QQ plot for the sample means:

%% 由于这个分布不能定义均值和方差，同时直方图也无法解释它，现在只有绘制关于这个样本均值的 QQ 图。 %%

```python
# Create the QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(cauchy_sample_means, plot=ax, fit=True)
plt.show()
```


    
![C3W3_UGL_Central_Limit_Theorem_52_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_52_0.png)

    


As you can see, this is very different from a straight line which let's you know that the sample means do not distribute normally. But what if you used a much bigger `sample_size`?

%% 正如您所见，这与直线截然不同，这表明样本均值并非正态分布。但是如果使用更大的 `sample_size` 会怎么样呢？ %%

```python
cauchy_sample_means = sample_means(cauchy_population, sample_size=100)

# Create the QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(cauchy_sample_means, plot=ax, fit=True)
plt.show()
```

![C3W3_UGL_Central_Limit_Theorem_54_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W3_UGL_Central_Limit_Theorem_54_0.png)


Even when using a `sample_size` of 100, which might be unrealistic in real-life scenarios you still don't achieve normality for the sample means. This is important because it is a fact that the central limit theorem does not hold for all distributions and that is a limitation to consider when applying it.

%% 即使使用 100 的样本量（在实际情境中可能不切实际），你仍然达不到样本均值的正态性。这一点很重要，因为事实是中心极限定理并不适用于所有分布，这在应用时是一个需要考虑的限制。 %%

**Congratulations on finishing this lab!**
