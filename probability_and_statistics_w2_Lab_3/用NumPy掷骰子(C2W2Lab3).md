---
aliases: ["Lab: Simulate Dice Throws with NumPy 🎲🤖"]
tags: []
created: 2026-01-02, 15:24:54
modified: 2026-01-03, 14:44:29
banner: https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/primary/numpylogo.svg
banner-height: 400
banner-display: cover
banner-fade: -6
---

# Lab: Simulate Dice Throws with NumPy 🎲🤖

Welcome! This lab shows how you can use Numpy to simulate rolling dice from rolling a single die up to summing the results from multiple rolls. You will also see how to handle situations in which one of the sides of the dice is loaded (it has a greater probability of landing on that side comparing to the rest).
%%
欢迎！这个实验展示了如何使用 Numpy 模拟掷骰子，从掷一次到累积多次。你还将学习如何处理骰子某一面被加重的情况（即该面落地的概率高于其他面）。
%%

Let's get started!


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## Represent a dice

The first thing you will need is to define how many sides your dice will have. You can even go a step further and represent a dice by using a NumPy array and assigning to each side a label which will be equal to the number of that side:
%%
第一件事情是需要定义一个有几个面的字典，你可以更进一步，用 NumPy 数组代表一个骰子，为每个面指定一个等于该面数量的标签：
%%
```python
# Define the desired number of sides (try changing this value!)
n_sides = 6

# Represent a dice by using a numpy array
dice = np.array([i for i in range(1, n_sides+1)])

dice
```

> [!result]
>     array([1, 2, 3, 4, 5, 6])

## Roll the dice

With your dice ready it is time to roll it. For now you will assume that the dice is fair, which means the probability of landing on each side is the same (it follows a uniform distribution). To achieve this behaviour you can use the function [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html), which given a NumPy array returns one of the entries in it randomnly:
%%
准备好骰子后就可以用了，现在先假设骰子是公平的，这意味着最后的结果的概率是相同的（它遵循[[连续概率分布#均匀分布（Uniform Distribution|均匀分布]]）要实现这个行为你可以使用 [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) 函数，给定一个 Numpy 数组随机返回一个条目。
%%

```python
# Run this cell multiple times (every time you should get a different result at random)
np.random.choice(dice)
```

> [!result]
>     5

This is great but if you wanted to roll the dice 20 times you will need to run the cell 20 times and record each result. Now you need a way to simulate several rolls at the same time. For this you can define the number of rolls you desire and use a list comprehension to roll the dice as many times as you like, you can also save every roll in a NumPy array:
%%
这挺好，但是如果你想投 20 次，则必须要运行 20 次并记录每次的结果。现在你需要一个方法来模拟它同时多次投掷。你可以定义一个投掷数，然后使用一个列表推导骰子的投掷，这样你就可以保存每次的投掷结果了。
%%
```python
# Roll the dice 20 times
n_rolls = 20

# Save the result of each roll
rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

rolls
```

> [!result]
>     array([1, 6, 3, 4, 2, 6, 4, 5, 1, 5, 1, 2, 3, 2, 3, 6, 2, 5, 5, 2])

Now you have a convenient way of keeping track of the result of each roll, nice!
%%
现在可以方便的将投掷的结果保存下来了。
%%
What is you would like to know the mean and variance of this process. For this you can use NumPy's functions [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) and [np.var](https://numpy.org/doc/stable/reference/generated/numpy.var.html):
%%
如果你想知道它们的均值和方差，可以使用 NumPy 的函数 [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) 和 [np.var](https://numpy.org/doc/stable/reference/generated/numpy.var.html):
%%

```python
# Compute mean of 20 rolls
m = np.mean(rolls)

# Compute variance of 20 rolls
v = np.var(rolls)

print(f"mean of rolls: {m:.2f}\nvariance of rolls: {v:.2f}")
```

> [!result]
> 	mean of rolls: 3.40
> 	variance of rolls: 2.94


You can even check the distribution of the rolls by plotting a histogram of the NumPy array that holds the result of each throw. For this you will use the plotting library Seaborn, concretely the [sns.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html) function:
%%
查看一下骰子投掷的分布，此时使用 NumPy 绘制透支结果的直方图。需要使用绘图的库 Seaborn，正是 [sns.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html) 函数。
%%

```python
# Display histogram of 20 rolls
n_rolls_hist = sns.histplot(rolls, discrete=True)
n_rolls_hist.set(title=f"Histogram of {n_rolls} rolls")
plt.show()
```

> [!result]
![C3W2_UGL_Dice_Simulations_11_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_11_0.png)

You probably didn't get a distribution that looks uniform (since the results are random). This happened because you are only simulating 20 rolls so far. Now try doing the same but for 20000 rolls:
%%
你可能无法得到均匀分布。这是因为你只模拟了 20 次的投掷。现在尝试 20000 次投掷：
%%
```python
n_rolls = 20_000

rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

print(f"mean of rolls: {np.mean(rolls):.2f}\nvariance of rolls: {np.var(rolls):.2f}")

n_rolls_hist = sns.histplot(rolls, discrete=True)
n_rolls_hist.set(title=f"Histogram of {n_rolls} rolls")
plt.show()
```

> [!result]
>     mean of rolls: 3.50
>     variance of rolls: 2.92
![C3W2_UGL_Dice_Simulations_13_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_13_1.png)

Does this plot and the metrics of mean and variance align with what you have learned about the uniform distribution during the course?
%%
这个绘图的指标，包括均值和方差，和你在课堂上学到的均匀分布的特性是否一致？
%%
Simulations are a great way of contrasting results against analytical solutions. For example, in this case the theoretical mean and variance are 3.5 and 2.916 respectively (you can check the formulas to get this results [here](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)). The important thing to keep in mind is that the more simulations you perform the closer your results will be to the analytical values so always choose an appropriate number of simulations! 
%%
模拟是验证分析结果有效性的绝佳方法。打个比方，在这个例子中理论平均数和方差分别为 3.5 和 2.916（你可以查看[这里](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)的公式代入并计算结果）。需要牢记的是，模拟的次数越多，结果就越接近解析解，所以请选择合适的模拟次数！
%%
NumPy is quite fast so performing 20 thousand runs is done fairly quick.
%%
2 万次的运行对于 NumPy 来说是非常快的。
%%
## Summing the result of rolling twice

Now you want to throw the dice twice and record the sum of the two rolls. For this you can do as before and save all results of the first roll in a NumPy array but this time you will have a second array that saves the results for the second rolls. 
%%
现在骰子掷两次，然后记录两次的和。你可以像之前一样，将第一次投掷的所有结果保存在一个 NumPy 数组中，但这次你还需要第二个数组来保存第二次投掷的结果。
%%
To get the sum you can simply sum the two arrays. This is possible because NumPy allows for vectorized operations such as this one. When you sum two NumPy arrays you will get a new array that includes the element-wise sum of the elements in the arrays you summed up.
%%
至于获取它们的和，你可以直接将两个数组相加。因为 NumPy 支持向量化的操作。当两个 NumPy 数组相加时，会返回一个新的数组，它包含了你求和的数组中元素的元素之和。
%%
Notice that now you can compute the the mean and variance for the first rolls, the second rolls and the sum of rolls. You can also compute the covariance between the first and second rolls:
%%
注意，你现在可以计算第一次，第二次以及两次相加的均值和方差。你同样可以计算第一次和第二次的协方差。
%%
```python
n_rolls = 20_000

# First roll (same as before)
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

# Second roll (code is the same but saved in a new numpy array)
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

# Sum both rolls (this is easy since numpy allows vectorization)
sum_of_rolls = first_rolls + second_rolls

# Print mean, variance and covariance
print(f"mean of first_rolls: {np.mean(first_rolls):.2f}\nvariance of first_rolls: {np.var(first_rolls):.2f}\n")
print(f"mean of second_rolls: {np.mean(second_rolls):.2f}\nvariance of second_rolls: {np.var(second_rolls):.2f}\n")
print(f"mean of sum_of_rolls: {np.mean(sum_of_rolls):.2f}\nvariance of sum_of_rolls: {np.var(sum_of_rolls):.2f}\n")
print(f"covariance between first and second roll:\n{np.cov(first_rolls, second_rolls)}")

# Plot histogram
sum_2_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
sum_2_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (sum of rolling twice)")
plt.show()
```

> [!result]
> 	mean of first_rolls: 3.49
> 	variance of first_rolls: 2.88
> 
> 	mean of second_rolls: 3.49
> 	variance of second_rolls: 2.93
> 
> 	mean of sum_of_rolls: 6.98
> 	variance of sum_of_rolls: 5.80
> 
> 	covariance between first and second roll:
> 	[[ 2.88089275 -0.00750594]
> 	 [-0.00750594  2.9341038 ]]
 ![C3W2_UGL_Dice_Simulations_16_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_16_1.png)


The resulting plot looks pretty Gaussian, as you might expect. Notice that the covariance between the first and second rolls is very close to zero since these two processes are independant of one another.
%%
这个绘图看上去非常的“高斯”，正如你所预料的。注意，第一次和第二次的协方差非常接近零。因为它们时相互独立的。
%%
Also notice that you can change the stat displayed in the histogram by changing the `stat` parameter of the `sns.histplot` function. In the previous exercises you were displaying the frequency but in this latter one you are plotting the probability, which makes more sense in this context. To check what other stats are available you can check the [docs](https://seaborn.pydata.org/generated/seaborn.histplot.html).
%%
还有一个注意注意的是，你可以更改直方图中显示的数据，修改 `sns.histplot` 函数的 `stat` 参数即可。在之前的练习中，你展示的是掷骰子结果的次数，而在此后的练习里，你绘制的是概率，在现在的情境下，这更加合理。要查看其他可用的统计数据，您可以查看文档。
%%
## Using loaded dice

So far you have only simulated dice that are fair (all of the sides on them have the same probability of showing up), but what about simulating loaded dice (one or more of the sides have a greater probability of showing up)?
%%
到目前位置你模拟的是公平的骰子（所有的面出现的概率相同），但是模拟灌铅的骰子呢（一面或者多面有更大概率出现）？
%%
It is actually pretty simple. [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) has support for these kind of scenarios by having a parameter `p` you can set. This parameter controls the probability of selecting each one of the entries in the array.
%%
其实非常简单。[np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) 有一个参数 `p` 可以设置，用于支持这种场景。这个参数控制了选择数组条目的概率。
%%
To see it in action, code a function that returns the probabilities of the dice landing on each side given that one of the sides must have twice as much probability as the rest of them:
%%
看看实际效果。构建一个函数，返回骰子落在每一边的概率，假设其中一边的概率必须是其他边的两倍：
%%
```python
def load_dice(n_sides, loaded_number):
    
    # All probabilities are initially the same
    probs = np.array([1/(n_sides+1) for _ in range(n_sides)])
    
    # Assign the loaded side a probability that is twice as the other ones
    probs[loaded_number-1] = 1 - sum(probs[:-1])
    
    # Check that all probabilities sum up to 1
    if not np.isclose(sum(probs), 1):
        print("All probabilities should add up to 1")
        return
    
    return probs 
```

Before using this function, check how the probabilities of a fair dice would look like:
%%
使用这个函数之前，首先是公平的骰子的概率是这样的：
%%
```python
# Compute same probabilities for every side
probs_fair_dice = np.array([1/n_sides]*n_sides)

# Plot probabilities
fair_dice_sides = sns.barplot(x=dice, y=probs_fair_dice)
fair_dice_sides.set(title=f"Histogram for fair dice with {n_sides} sides")
fair_dice_sides.set_ylim(0,0.5)
plt.show()
```

> [!result]
![C3W2_UGL_Dice_Simulations_21_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_21_0.png)


Now get the probabilities by using the `load_dice` function. Try changing the loaded side!
%%
现在用 `load_dice` 函数来获取概率。
%%
```python
# Get probabilities if dice is loaded towards side 2
probs_loaded_dice = load_dice(n_sides, loaded_number=2)

# Plot probabilities
loaded_dice_sides = sns.barplot(x=dice, y=probs_loaded_dice)
loaded_dice_sides.set(title=f"Histogram for loaded dice with {n_sides} sides")
loaded_dice_sides.set_ylim(0,0.5)
plt.show()
```

> [!result]    
![C3W2_UGL_Dice_Simulations_23_0.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_23_0.png)

Now, feed the `probs_loaded_dice` array into `np.random.choice` and see how this affect the metrics and plot:
%%
现在使用 `np.random.choice` 来填充 `probs_loaded_dice` 的数组，然后看看指标和绘图是怎么被影响的。
%%
```python
n_rolls = 20_000

# Notice that the p parameter is being set
first_rolls = np.array([np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)])

second_rolls = np.array([np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)])

sum_of_rolls = first_rolls + second_rolls

print(f"mean of first_rolls: {np.mean(first_rolls):.2f}\nvariance of first_rolls: {np.var(first_rolls):.2f}\n")
print(f"mean of second_rolls: {np.mean(second_rolls):.2f}\nvariance of second_rolls: {np.var(second_rolls):.2f}\n")
print(f"mean of sum_of_rolls: {np.mean(sum_of_rolls):.2f}\nvariance of sum_of_rolls: {np.var(sum_of_rolls):.2f}\n")
print(f"covariance between first and second roll:\n{np.cov(first_rolls, second_rolls)}")

# Plot histogram
loaded_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
loaded_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (sum of rolling twice a loaded dice)")
loaded_rolls_hist.set_xticks(range(min(sum_of_rolls),max(sum_of_rolls)+1))
plt.show()
```

> [!result]
>     mean of first_rolls: 3.29
>     variance of first_rolls: 2.79
> 
>     mean of second_rolls: 3.28
>     variance of second_rolls: 2.80
>     
>     mean of sum_of_rolls: 6.57
>     variance of sum_of_rolls: 5.58
 >    
>     covariance between first and second roll:
>     [[ 2.79335935 -0.0054822 ]
>      [-0.0054822   2.80130054]]
![C3W2_UGL_Dice_Simulations_25_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_25_1.png)


Now the histogram is skewed towards some values since some sums are now more likely than others. Try changing the loaded side and see how the histogram changes!
%%
现在，由于某些值的总和现在出现的可能性更高，直方图会偏向于某些值。尝试改变 loaded side，然后查看直方图如何变化。
%%
Notice that covariance is still very close to zero since there is not any dependance between rolls of the dice.
%%
注意，协方差依然非常接近零，因为两次投掷依然是独立事件。
%%
## Dependant Rolls

To finish this lab you will now simulate the scenario in which the second roll depends on the result of the first one. Say that you are playing a variant of the game you have played so far and you only roll the dice a second time if the result of the first roll is greater or equal to 4.
%%
在这个实验的最后，我们来模拟这个场景，第二次掷骰子取决于第一次的结果。假设你正在玩一个游戏的变体，如果第一次的结果不小于 4，你才能掷第二次。
%%
Before doing the simulations reflect on what might happen in this scenario. Some behavior you will probably see:
%%
在开始模拟前，先确认一下这个场景，你可能会观察到：
%%
- 1 is now a possible result since if you get a 1 in the first roll you don't roll again
- 1, 2 and 3 now have a greater chance of showing up
- 4 is now not a possible result since you need to roll again if you get a 4 in the first roll

%%
- 如果在第一次掷骰中得到 1，就不能再掷第二次了，所以 1 成为了结果之一。
- 1、2 和 3 现在有更大的概率出现。
- 由于第一次投掷结果为 4 的时候会再投掷一次，那么结果 4 将不再出现。
%%

To achieve this behaviour you can use the [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function, which given a condition can be used to zero-out the elements that don't meet its criteria:
%%
为了实现这个行为，你需要使用 [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html) 函数，它在给定条件下，可用于将不符合其标准的元素清零：
%%

```python
n_rolls = 20_000

first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

# Preserve the result of the second throw only if the first roll was greater or equal to 4
second_rolls = np.where(first_rolls>=4, second_rolls, 0)

sum_of_rolls = first_rolls + second_rolls

print(f"mean of first_rolls: {np.mean(first_rolls):.2f}\nvariance of first_rolls: {np.var(first_rolls):.2f}\n")
print(f"mean of second_rolls: {np.mean(second_rolls):.2f}\nvariance of second_rolls: {np.var(second_rolls):.2f}\n")
print(f"mean of sum_of_rolls: {np.mean(sum_of_rolls):.2f}\nvariance of sum_of_rolls: {np.var(sum_of_rolls):.2f}\n")
print(f"covariance between first and second roll:\n{np.cov(first_rolls, second_rolls)}")

# Plot histogram
dependant_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
dependant_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (dependant sum of rolling twice)")
dependant_rolls_hist.set_xticks(range(min(sum_of_rolls),max(sum_of_rolls)+1))
plt.show()
```

> [!result]
>     mean of first_rolls: 3.49
>     variance of first_rolls: 2.93
    > 
>     mean of second_rolls: 1.74
>     variance of second_rolls: 4.51
    > 
>     mean of sum_of_rolls: 5.23
>     variance of sum_of_rolls: 12.72
    > 
>     covariance between first and second roll:
>     [[2.93191435 2.63630534]
>      [2.63630534 4.51260641]]
![C3W2_UGL_Dice_Simulations_28_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_28_1.png)

Looks like all of the predictions of this new scenario indeed happened. Notice that the covariance now is nowhere near zero since there is a dependency between the first and the second roll!
%%
结果正如我们所料。注意，由于两次投掷是相关的，协方差的结果和零没关系了。
%%
**Now you have finished this ungraded lab, nice job!**
