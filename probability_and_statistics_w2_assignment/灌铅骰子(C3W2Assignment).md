---
aliases: ["Loaded dice "]
tags: []
created: 2026-01-03, 15:49:46
modified: 2026-01-05, 12:05:41
---

# Loaded dice 

Welcome to the second assignment in the course Probability and Statistics for Machine Learning and Data Science! In this quiz-like assignment you will test your intuition about the concepts covered in the lectures by taking the example with the dice to the next level. 
%%
欢迎来到《机器学习和数据科学——概率与统计》课程的第二次练习！在这个类测验式的作业中，你将通过把骰子示例提升到更高层次，来检验你对课程中所涵盖概念的直觉。
%%
**This assignment can be completed with just pencil and paper, or even your intuition about probability, but in many questions using the skills you're developing as a programmer may help**. 
%%
要完成这个测试，你可以只用铅笔和纸，甚至是你对概率的直觉，但在许多问题中，运用你作为程序员的技能可能会有所帮助。
%%
## 1 - Introduction

You will be presented with 11 questions regarding a several dice games. Sometimes the dice is loaded, sometimes it is not. You will have clear instructions for each exercise.
%%
你将面对关于骰子游戏的 11 个问题。有一些是灌铅骰子，有一些则不是，每个练习都有明确的指示。
%%
### 1.1 How to go through the assignment

In each exercise you there will be a question about throwing some dice that may or may not be loaded. You will have to answer questions about the results of each scenario, such as calculating the expected value of the dice throw or selecting the graph that best represents the distribution of outcomes. 
%%
在每个练习中，问题都围绕着掷一些骰子进行，它们可能灌铅了可能没灌铅。你需要根据每个情景回答问题，比如计算掷骰子的期望值，或者选择最能代表结果分布的图形。
%%
In any case, **you will be able to solve the exercise with one of the following methods:**
%%
无论如何，你都可以通过下面两种方式的任意一种解决问题：
%%
- **By hand:** You may make your calculations by hand, using the theory you have developed in the lectures.
- **Using Python:** You may use the empty block of code provided to make computations and simulations, to obtain the result.

%%
- 动手：运用在讲座中阐述的理论进行手动计算。
- 使用 Python：你同样可以使用空代码块计算和模拟，从而获得结果。
%%

After each exercise you will save your solution by running a special code cell and adding your answer. The cells contain a single line of code in the format `utils.exercise_1()` which will launch the interface in which you can save your answer. **You will save your responses to each exercise as you go, but you won't submit all your responses for grading until you submit this assignment at the end.**
%%
在每个练习完成后，你需要保存结果并运行一个特殊的代码块来填写你的答案。这个代码单元包含了一行代码，其格式为 `utils.exercise_1()` ，它将启动一个交互界面，你可以将答案保存在里面。但直到最后提交此作业时，才会将所有回答一并提交以供评分。
%%
Let's go over an example! Before, let's import the necessary libraries.
%%
开始示例之前，需要导入必要的库
%%
## 2 - Importing the libraries


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
```

## 3 - A worked example on how to complete this assignment.

Now let's go over one example question, so you understand how to go through the assignment.
%%
我们先来看一个示例，这样你就明白如何完成这项任务了。
%%
### 3.1 Example question

Question: Given a 6-sided fair dice, you throw it two times and save the result. What is the probability that the sum of your two throws is greater than 5? (Give your result with 1 decimal place).
%%
问题：一个 6 面的公平骰子，投掷两次并记录结果。它们的和超过 5 的概率是多少？（结果保留一位小数）
%%
After the question, you will see the following block of code.
%%
问题的后面你可以看到一个代码块
%%
#### 解析解

```python
for i in range(1, 7):
    print('')
    for j in range(1, 7):
        print(str(i+j) + ',', end='')
```

> [!result]
> 	2,3,4,5,6,7,
> 	3,4,5,6,7,8,
> 	4,5,6,7,8,9,
> 	5,6,7,8,9,10,
> 	6,7,8,9,10,11,
> 	7,8,9,10,11,12,

上面求出的是两个骰子所有的和，一共 $6\times6=36$ 种结果。找到一共有 4 个 “5”，而 5 的左侧的值都比 5 小，即 36 个结果种小于 6 的值为 10 个。那么大于 5 的结果就有 $36-10=26$ 个，概率的解析解即为：
$$
\frac{26}{36}=0.7\dot{2}
$$
#### 蒙特卡洛模拟

```python
# 定义骰子
n_sides = 6
dice = np.array([i for i in range(1, n_sides+1)])

# 定义投掷的次数并开始投
n_rolls = 20_0000
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

# 两次投掷的和
sum_of_rolls = first_rolls + second_rolls

# 输出最终大于 5 的概率
(sum_of_rolls > 5).sum()/n_rolls
```

> [!result]
> 	0.723295

---

You may use it as you wish to solve the exercise. Or you can just ignore it and use pencil and pen to solve. It is up to you! **You will only save your final answer**. 
%%
你可以使用它来解决这个练习，或者你可以忽略他，用笔来解决。无论你如何解决，只需要保存你最后的答案即可。
%%
### 3.2 Solving using simulations in Python

Let's solve this question in both ways. First, using Python. You may check the ungraded lab Dice Simulations that appears right before this assignment to help you simulate dice throws. Remember that, to get a good approximation, you need to simulate it a lot of times! You will see why this is true in the following weeks, but this is quite intuitive.
%%
让我们用这两种方法解答这个问题。首先使用 Python。你也许在本次练习前看了模拟骰子的实验，为了获得一个良好的近似值，你需要模拟很多次！在之后的课程中你会知道这样做的原因，这非常直观。
%%
```python
# You can use this cell for your calculations (not graded)

# This list represents each dice side
dice = [1,2,3,4,5,6]

# The idea is to randomly choose one element from this list three times and sum them. 
# Each time we choose, it is as if we had thrown a dice and the side is the chosen number.
# This list will store the sum for each iteration. The idea is to repeat this experiment several times.
sum_results = []

number_iterations = 1000

# Setting a random seed just for reproducibility
np.random.seed(42)
# It will play this game number_iteration times
for i in range(number_iterations):
    # Throw the first dice
    throw_1 = np.random.choice(dice)
    # Throw the second dice
    throw_2 = np.random.choice(dice)
    # Sum the result
    sum_throw = throw_1 + throw_2
    # Append to the sum_result list
    sum_results.append(sum_throw)

# After recording all the sums, the actual probability will be very close to the proportion among every sum greater than 10 in the sum_results list.
greater_5_count = 0

for x in sum_results:
    if x > 5:
        greater_5_count += 1

probability = greater_5_count/len(sum_results)    
print(f"The probability by this simulation is: {probability}")
```

> [!result]
> 	The probability by this simulation is: 0.719

So the result you would get, rounding in to decimal place, would be 0.7! Let's solve it "by hand".
%%
所以你得到了保留一位小数的结果是 0.7！接下来我们手动求解。
%%
### 3.3 Solving using the theory

When throwing two dice, there are $36$ possible outcomes:
%%
当投掷 2 颗骰子，有 36 种可能的结果。
%%
$$(1,1), (1,2), \ldots, (6,6)$$

You must count how many of them lead to a sum greater than 5. They are:
%%
你必须统计大于 5 的情况有多少种。它们分别是：
%%
* If the first throw is $1$, there are $2$ possibilities for the second throw: 5 or 6.
* If the first throw is $2$, there are $3$ possibilities for the second throw: 4, 5 or 6.
* If the first throw is $3$, there are $4$ possibilities for the second throw: 3, 4, 5 or 6.
* If the first throw is $4$, there are $5$ possibilities for the second throw: 2, 3, 4, 5 or 6.
* If the first throw is $5$, there are $6$ possibilities for the second throw: 1, 2, 3, 4, 5 or 6.
* If the first throw is $6$, there are $6$ possibilities for the second throw: 1, 2, 3, 4, 5 or 6.

%%
- 如果第一次是 1，那么第二次则是 2 种可能：5 或者 6。
- 如果第一次是 2，那么第二次则是 3 种可能：4，5，6。
- 如果第一次是 3，那么第二次则是 4 种可能：3，4，5，6。
- 如果第一次是 4，那么第二次则是 5 种可能：2，3，4，5，6。
- 如果第一次是 5，那么第二次则是 6 种可能：1，2，3，4，5，6。
- 如果第一次是 6，那么第二次同样是 6 种可能：1，2，3，4，5，6。
%%

So, in total there are $2 + 3 + 4 + 5 + 6 + 6 = 26$, possibilities that sum greater than 5.
%%
所以，两次投掷和大于 5 的概率为 $2 + 3 + 4 + 5 + 6 + 6 = 26$。
%%

The probability is then $\frac{26}{36} \approx 0.72$. Rounding it to 1 decimal place, the result is also 0.7!
%%
概率为 $\frac{26}{36} \approx 0.72$，保留一位小数，结果同样是 0.7！
%%
### 3.4 Saving your answer

Once you get your answer in hands, it is time to save it. Run the next code below to see what it will look like. You just add your answer as requested and click on "Save your answer!"
%%
一旦你得到了答案，就可以保存了。运行下面的代码。你只需要将你的答案填入然后点击 "Save your answer!"
%%

```python
utils.exercise_example()
```

And that's it! Once you save one question, you can go to the next one. If you want to change your solution, just run the code again and input the new solution, it will overwrite the previous one. At the end of the assignment, you will be able to check if you have forgotten to save any question. 
%%
然后就完成了！一旦你保存了一个问题，你可以去解决下一个。如果你想要修改你的答案，只需要再次运行这段代码并输入新的解，它将覆盖上一次的答案。在作业结束时，您可以检查是否忘记保存所有的问题。
%%
Once you finish the assignment, you may submit it as you usually would. Your most recently save answers to each exercise will then be graded.
%%
一旦你完成了这个作业，你可以直接提交它。你在每个练习种最近一次保存的答案将进行评分。
%%
## 4 - Some concept clarifications 🎲🎲🎲

During this assignment you will be presented with various scenarios that involve dice. Usually dice can have different numbers of sides and can be either fair or loaded.
%%
在这个作业中会有各种关于骰子的场景，一般来说骰子都有不同的面对应不同的数，同时它有可能是公平的也有可能灌铅了。
%%
- A fair dice has equal probability of landing on every side.
- A loaded dice does not have equal probability of landing on every side. Usually one (or more) sides have a greater probability of showing up than the rest.

%%
- 公平的骰子在投掷后所有的面的概率都是相同的。
- 灌铅骰子的每个面的概率不一定相同，一般来说一个（或者多个）面的概率比其他面的概率更大。
%%

Alright, that's all your need to know to complete this assignment. Time to start rolling some dice!
%%
好了，以上就是为了完成这个练习需要知道的。开始投掷骰子吧！
%%
## Exercise 1:

Given a 6-sided fair dice (all of the sides have equal probability of showing up), compute the mean and variance for the probability 
distribution that models said dice. The next figure shows you a visual represenatation of said distribution:
%%
给定一个公平的 6 面骰子（所有面都有相同的概率），计算描述该骰子的概率分布的均值和方差。下图是它的可视化分布：
%%

![fair_dice.png|400](https://obsidian-image.wwtt.xyz/2026/01/fair_dice.png)


**Submission considerations:**
- Submit your answers as floating point numbers with three digits after the decimal point
- Example: To submit the value of 1/4 enter 0.250

%%
提交注意：
- 以浮点数提交答案，保留三位小数。
- 示例：如果提交的值为 1/4，则输入 0.250
%%

Hints: 
- You can use [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) to simulate a fair dice.
- You can use [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) and [np.var](https://numpy.org/doc/stable/reference/generated/numpy.var.html) to compute the mean and variance of a numpy array.

%%
提示：
- 你可以使用 [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) 来模拟公平的骰子。
- 你可以使用 [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) 和 [np.var](https://numpy.org/doc/stable/reference/generated/numpy.var.html) 来计算 NumPy 数组的均值和方差。
%%

### 蒙特卡洛模拟

```python
# 定义骰子
n_sides = 6
dice = np.array([i for i in range(1, n_sides+1)])

# 定义投掷骰子的次数并开始投掷
n_rolls = 20_0000
rolls = np.random.choice(dice, size=n_rolls)


# 计算骰子的期望值（均值）
mean_roll = np.mean(rolls)
# 计算骰子的方差
var_roll = np.var(rolls)

# 打印结果
print(f"The expected value of rolling a {n_sides}-sided dice is {mean_roll}")
print(f"The variance of rolling a {n_sides}-sided dice is {var_roll}")
```

> [!result]
> 	The expected value of rolling a 6-sided dice is 3.50701
> 	The variance of rolling a 6-sided dice is 2.920270859899999

### 解析解

#### 期望值

首先已知每个概率均为 $\frac{1}{6}$，同时它们的值为 1 到 6，代入[[期望值#离散期望值]]的公式可得到
$$
\begin{align}
\mathbb{E}[X]=&\ \sum_{i}x_{i} \cdot P(X=x_{i}) \\
=& \ 1 \times \frac{1}{6} + 2 \times \frac{1}{6} + 3\times \frac{1}{6}+4\times \frac{1}{6}+5\times \frac{1}{6}+6\times \frac{1}{6} \\
=& \ \frac{7}{2}=3.5
\end{align}
$$
#### 方差

代入 [[方差#平方的期望 - 期望的平方|平方的期望 - 期望的平方]]，其中期望值已经求出为 3.5，那么：
$$
\begin{aligned}
Var(X) &= \mathbb{E}[X^2] - \mathbb{E}[X]^2 \\
&=\left( 1^2 \times \frac{1}{6} + 2^2 \times \frac{1}{6} + 3^2 \times \frac{1}{6} + 4^2 \times \frac{1}{6} + 5^2 \times \frac{1}{6} + 6^2 \times \frac{1}{6}\right) - (3.5)^2 \\
&=\frac{35}{12} = 2.91\dot{6}
\end{aligned}
$$


```python
# Run this cell to submit your answer
utils.exercise_1()
```

## Exercise 2:

Now suppose you are throwing the dice (same dice as in the previous exercise) two times and recording the sum of each throw. Which of the following `probability mass functions` will be the one you should get?

%%
现在假设你投掷两次骰子（和上一次练习一样的骰子），记录它们的值并计算它们的和。你得到了下面三个图中哪个[[离散概率分布#概念|概率质量函数]]？
%%

<table><tr>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/hist_sum_6_side.png" style="height: 300px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/hist_sum_5_side.png" style="height: 300px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/hist_sum_6_uf.png" style="height: 300px;"/> </td>
</tr></table>


Hints: 
- You can use numpy arrays to hold the results of many throws.
- You can sum to numpy arrays by using the `+` operator like this: `sum = first_throw + second_throw`
- To simulate multiple throws of a dice you can use list comprehension or a for loop

%%
提示
- 你可以使用 NumPy 数组保存多次投掷的结果。
- 使用 `+` 号就可以对 NumPy 数组求和，，比如像这样：`sum = first_throw + second_throw`。
- 你可以使用列表推导或者 for 循环来模拟多次投掷骰子。
%%
### 蒙特卡洛模拟

```python
# 定义骰子
n_sides = 6
dice = np.array([i for i in range(1, n_sides+1)])

# 定义投掷的次数并开始投
n_rolls = 20_0000
first_rolls = np.random.choice(dice, size=n_rolls)
second_rolls = np.random.choice(dice, size=n_rolls)

# 计算两次投掷的和
sum_of_rolls = first_rolls + second_rolls

# 绘制直方图
sum_2_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
sum_2_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (sum of rolling twice)")
plt.show()
```

> [!result]
 ![C3W2_UGL_Dice_Simulations_16_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/C3W2_UGL_Dice_Simulations_16_1.png)

样本不足会直接导致数据偏差，20000 次其实并不能完全模拟出比较精确的结果。
### 解析解

**古典概型**方法，直接列出即可：

- 两个骰子的和组成的方式为 36 种，即样本空间为 36。
- 2 只有 1 种组成方式：$11$，那么 2 的概率为 $\frac{1}{36}=0.02\dot{7}$。
- 3 只有 2 种组成方式：$12,21$，那么 3 的概率为 $\frac{2}{36}=0.0\dot{5}$。
- 4 有 3 种组成方式：$13,22,31$，那么 4 的概率为 $\frac{3}{36}=0.19\dot{4}$。
- 5 有 4 种组成方式：$14,23,32,41$，那么 5 的概率为 $\frac{4}{36}=0.\dot{1}$。
- 6 有 5 种组成方式：$15,24,33,42,51$ 那么 6 的概率为 $\frac{6}{36}=0.1\dot{6}$。
- 7 有 6 种组成方式：$16,25,34,43,52,61$ 那么 7 的概率为 $\frac{6}{36}=0.19\dot{4}$。
- 8 有 5 种组成方式：$26,35,44,53,62$ 那么 8 的概率和 6 相同，为 $\frac{6}{36}=0.1\dot{6}$。
- 9 有 4 种组成方式：$36,45,54,63$，那么 9 的概率和 5 相同，为 $\frac{4}{36}=0.\dot{1}$。
- 10 有 3 种组成方式：$46,55,64$，那么 10 的概率和 4 相同，为 $\frac{3}{36}=0.19\dot{4}$。
- 11 只有 2 种组成方式：$65,56$，那么 11 的概率和 3 相同，为 $\frac{2}{36}=0.0\dot{5}$。
- 12 只有 1 种组成方式：$66$，那么 12 的概率与 2 相同，为 $\frac{1}{36}=0.02\dot{7}$。


```python
# Run this cell to submit your answer
utils.exercise_2()
```

## Exercise 3:

Given a fair 4-sided dice, you throw it two times and record the sum. The figure on the left shows the probabilities of the dice landing on each side and the right figure the histogram of the sum. Fill out the probabilities of each sum (notice that the distribution of the sum is symetrical so you only need to input 4 values in total):
%%
给定一个公平的四面骰子，投掷两次记录它的和。左图为骰子每面出现的概率，右图为它的和的直方图。填写每个和的概率。（注意，分布是对称的，所以只需要 4 个值）。
%%

![4_side_hists.png|700](https://obsidian-image.wwtt.xyz/2026/01/4_side_hists.png)

**Submission considerations:**
- Submit your answers as floating point numbers with three digits after the decimal point
- Example: To submit the value of 1/4 enter 0.250
### 蒙特卡洛模拟

```python
# 定义骰子
n_sides = 4
dice = np.array([i for i in range(1, n_sides+1)])

# 定义投掷的次数并开始投
n_rolls = 20_0000
first_rolls = np.random.choice(dice, size=n_rolls)
second_rolls = np.random.choice(dice, size=n_rolls)

# 计算两次投掷的和
sum_of_rolls = first_rolls + second_rolls

# 提取所有出现的和并计算每个和的概率
for i in np.unique(sum_of_rolls):
    print(f"Probability of sum = {i}: {(sum_of_rolls == i).sum() / n_rolls}")
```

> [!result]
> 	Probability of sum is 2: 0.063115
> 	Probability of sum is 3: 0.12455
> 	Probability of sum is 4: 0.18742
> 	Probability of sum is 5: 0.25022
> 	Probability of sum is 6: 0.18687
> 	Probability of sum is 7: 0.125475
> 	Probability of sum is 8: 0.06235

### 解析解

和上面一道题几乎一模一样：

- 两个骰子的和组成的方式为 $4\times 4=16$ 种，即样本空间为 16。
- 2 有 1 种组成方式：$11$，那么 2 的概率为 $\frac{1}{16}=0.0625$。
- 3 有 2 种组成方式：$12,21$，那么 3 的概率为 $\frac{2}{16}=0.125$。
- 4 有 3 种组成方式：$13,22,31$，那么 4 的概率为 $\frac{3}{16}=0.1875$。
- 5 有 4 种组成方式：$14,23,32,41$，那么 5 的概率为 $\frac{4}{16}=0.25$。
- 6 有 3 种组成方式：$24,33,42$，那么 6 的概率与 4 相同，为 $\frac{3}{16}=0.1875$。
- 7 有 2 种组成方式：$34,43$，那么 7 的概率与 3 相同，为 $\frac{2}{16}=0.125$。
- 8 有 1 种组成方式：$44$，那么 8 的概率与 2 相同，为 $\frac{1}{16}=0.0625$。


```python
# Run this cell to submit your answer
utils.exercise_3()
```

## Exercise 4:

Using the same scenario as in the previous exercise. Compute the mean and variance of the sum of the two throws and the covariance between the first and the second throw:
%%
使用上个练习相同的情景。计算两次投掷之和的均值和方差，并计算两次投掷之间的协方差。
%%

![4_sided_hist_no_prob.png|400](https://obsidian-image.wwtt.xyz/2026/01/4_sided_hist_no_prob.png)



Hints:
- You can use [np.cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) to compute the covariance of two numpy arrays (this may not be needed for this particular exercise).

%%
提示：
- 可以使用 [np.cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) 来计算两个 NumPy 数组的协方差（这在本次练习中可能不需要这个。）
%%

### 蒙特卡洛模拟

```python
# 定义骰子
n_sides = 4
dice = np.array([i for i in range(1, n_sides+1)])

# 定义投掷的次数并开始投
n_rolls = 20_0000
first_rolls = np.random.choice(dice, size=n_rolls)
second_rolls = np.random.choice(dice, size=n_rolls)

# 计算两次投掷的和
sum_of_rolls = first_rolls + second_rolls

# 输出结果
print(f"mean of sum_of_rolls: {np.mean(sum_of_rolls)}")
print(f"variance of sum_of_rolls: {np.var(sum_of_rolls)}")
print(f"covariance between first and second roll: \n {np.cov(first_rolls, second_rolls)}")
```

> [!result]
> 	mean of sum_of_rolls: 5.00189
> 	variance of sum_of_rolls: 2.4964964279000004
> 	covariance between first and second roll: 
> 	 [[ 1.24966037 -0.00179371]
> 	  [-0.00179371  1.25043597]]



### 解析解

这道题和 [[#Exercise 1]] 差不多
#### 期望值

[[#Exercise 3]] 已经得到每项的概率，那么直接代入可得：
$$
\mathbb{E}[X] = 2 \times \frac{1}{16}+
3 \times \frac{2}{16}+
4 \times \frac{3}{16}+
5 \times \frac{4}{16}+
6 \times \frac{3}{16}+
7 \times \frac{2}{16}+
8 \times \frac{1}{16}
=\frac{40}{8}=5
$$
#### 方差
$$
\begin{align}
Var(X) &= 
\left( 
2^2 \times \frac{1}{16}+
3^2 \times \frac{2}{16}+
4^2 \times \frac{3}{16}+
5^2 \times \frac{4}{16}+
6^2 \times \frac{3}{16}+
7^2 \times \frac{2}{16}+
8^2 \times \frac{1}{16}
 \right) - 5^2 \\
&=\frac{55}{2} - 25=\frac{5}{2}=2.5
\end{align}
$$
#### 协方差

由于第一次和第二次相互独立，所以它们没有相关性，协方差结果为 0。


```python
# Run this cell to submit your answer
utils.exercise_4()
```

## Exercise 5:


Now suppose you are have a loaded 4-sided dice (it is loaded so that it lands twice as often on side 2 compared to the other sides): 
%%
现在假设你有一个 4 面的灌铅骰子（它被加重，使得它落在第 2 面上的次数是其他面的两倍）：
%%

![4_side_uf.png|400](https://obsidian-image.wwtt.xyz/2026/01/4_side_uf.png)


You are throwing it two times and recording the sum of each throw. Which of the following `probability mass functions` will be the one you should get?
%%
投掷两次记录它的和。得到下面哪个图符合概率质量函数？
%%
<table><tr>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/hist_sum_4_4l.png" style="height: 300px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/hist_sum_4_3l.png" style="height: 300px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/hist_sum_4_uf.png" style="height: 300px;"/> </td>
</tr></table>

Hints: 
- You can use the `p` parameter of [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) to simulate a loaded dice.
%%
提示：
- 你可以使用 [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) 的 `p` 参数模拟灌铅骰子。
%%

### 蒙特卡洛模拟

```python
# 参数定义
n_sides = 4
n_rolls = 200_0000
loaded_number = 2
base_prob = 1 / (n_sides + 1)

# 创建灌铅骰子的概率数组，先填充，再修改
probs_loaded = np.full(n_sides, base_prob)
probs_loaded[loaded_number - 1] *= 2

# 创建骰子
dice = np.array([i for i in range(1, n_sides+1)])

# 投掷骰子
first_rolls = np.random.choice(dice, size=n_rolls, p=probs_loaded)
second_rolls = np.random.choice(dice, size=n_rolls, p=probs_loaded)

sum_of_rolls = first_rolls + second_rolls

# 绘制直方图
sum_2_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
sum_2_rolls_hist.set(title=f"Histogram of {n_rolls} Loaded Dice Rolls (Sum of 2)")
plt.show()
```

> [!result]
![四面灌铅骰子投两次分布直方图1.png|600](https://obsidian-image.wwtt.xyz/2026/01/四面灌铅骰子投两次分布直方图1.png)

### 解析解

1、3、4 的概率分别为 0.2，2 的概率为 0.4，那么：
- 2 有 1 种组成方式：$11$，那么 2 的概率为 $0.2\times0.2=0.04$。
- 3 有 2 种组成方式：$12,21$，那么 3 的概率为 $0.2\times 0.4 + 0.4 \times 0.2 =0.16$。
- 4 有 3 种组成方式：$13,22,31$，那么 4 的概率为 $0.2 \times 0.2 +0.4 \times 0.4 +0.2 \times 0.2 =0.24$。
- 5 有 4 种组成方式：$14,23,32,41$，那么 5 的概率为 $2(0.2 \times 0.2)+ 2(0.2 \times 0.4)=0.24$。
- 6 有 3 种组成方式：$24,33,42$，那么 6 的概率为 $2(0.4 \times 0.2) +0.2 \times 0.2 =0.2$。
- 7 有 2 种组成方式：$34,43$，那么 7 的概率为 $2(0.2 \times 0.2)=0.08$。
- 8 有 1 种组成方式：$44$，那么 8 的概率与 2 相同，为  $0.2\times0.2=0.04$。

```python
# Run this cell to submit your answer
utils.exercise_5()
```

## Exercise 6:

You have a 6-sided dice that is loaded so that it lands twice as often on side 3 compared to the other sides:
%%
你有一个 6 面灌铅的骰子，掷出 3 的概率为其他面的两倍。
%%

![loaded_6_side.png|500](https://obsidian-image.wwtt.xyz/2026/01/loaded_6_side.png)


You record the sum of throwing it twice. What is the highest value (of the sum) that will yield a cumulative probability lower or equal to 0.5?

%%
当你记录两次投掷之和时，使得‘累积概率’小于或等于 0.5 的‘最大和’是多少？
%%

![loaded_6_cdf.png|500](https://obsidian-image.wwtt.xyz/2026/01/loaded_6_cdf.png)


Hints:
- The probability of side 3 is equal to $\frac{2}{7}$

%%
提示：
- 3 的概率等于 $\frac{2}{7}$
%%

### 蒙特卡洛模拟

```python
# 参数定义
n_sides = 6
n_rolls = 200_0000
loaded_number = 3
base_prob = 1 / (n_sides + 1)

# 创建灌铅骰子的概率数组，先填充，再修改
probs_loaded = np.full(n_sides, base_prob)
probs_loaded[loaded_number - 1] *= 2

# 创建骰子
dice = np.array([i for i in range(1, n_sides+1)])

# 投掷骰子
first_rolls = np.random.choice(dice, size=n_rolls, p=probs_loaded)
second_rolls = np.random.choice(dice, size=n_rolls, p=probs_loaded)

sum_of_rolls = first_rolls + second_rolls

# 统计每个“和”出现的次数
sums, counts = np.unique(sum_of_rolls, return_counts=True)

# 计算每个“和”的概率并求累积概率
cum_probs = np.cumsum(counts / n_rolls)

for i in range(len(sums)):
    print(f"和为 {sums[i]} 的概率为: {counts[i] / n_rolls}")
print("")

# 筛选出累积概率 <= 0.5 的所有“和”，并取其中的最大值
result = sums[cum_probs <= 0.5][-1]
print(f"CDF小于等于0.5的最高值为: {result}")
```

> [!result]
> 	和为 2 的概率为: 0.020274
> 	和为 3 的概率为: 0.040775
> 	和为 4 的概率为: 0.1022415
> 	和为 5 的概率为: 0.122736
> 	和为 6 的概率为: 0.1633815
> 	和为 7 的概率为: 0.1632205
> 	和为 8 的概率为: 0.142622
> 	和为 9 的概率为: 0.1220725
> 	和为 10 的概率为: 0.061258
> 	和为 11 的概率为: 0.040843
> 	和为 12 的概率为: 0.020576
> 
> 	CDF 小于等于 0.5 的最高值为: 6

### 解析解

事实上，当概率计算出来后，后续的计算方法和 Python 几乎一致。

1. 计算每种结果的概率，其中 3 的概率为 $\frac{2}{7}$，其他数字为 $\frac{1}{7}$，概率结果四舍五入到 3 位小数:

| 数值  | 构成                  | 概率                                                                                                              |
| --- | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| 2   | $11$                | $$\frac{1}{7} \times \frac{1}{7} \approx 0.020$$                                                                |
| 3   | $12,21$             | $$2\left( \frac{1}{7} \times \frac{1}{7} \right) \approx 0.041$$                                                |
| 4   | $13,22,31$          | $$\frac{1}{7} \times \frac{1}{7} +2\left( \frac{1}{7} \times \frac{2}{7} \right)\approx 0.102$$                 |
| 5   | $14,23,32,41$       | $$2\left( \frac{1}{7} \times \frac{1}{7} \right) +2\left( \frac{1}{7} \times \frac{2}{7} \right)\approx 0.122$$ |
| 6   | $15,24,33,42,51$    | $$4\left( \frac{1}{7} \times \frac{1}{7} \right) +\frac{2}{7} \times \frac{2}{7} \approx 0.163$$                |
| 7   | $16,25,34,43,52,61$ | $$4\left( \frac{1}{7} \times \frac{1}{7} \right) +2\left( \frac{1}{7} \times \frac{2}{7} \right)\approx 0.163$$ |
| 8   | $26,35,44,53,62$    | $$3\left( \frac{1}{7} \times \frac{1}{7} \right) +2\left( \frac{1}{7} \times \frac{2}{7} \right)\approx 0.143$$ |
| 9   | $36,45,54,63$       | $$2\left( \frac{1}{7} \times \frac{1}{7} \right) +2\left( \frac{1}{7} \times \frac{2}{7} \right)\approx 0.122$$ |
| 10  | $46,55,64$          | $$3\left( \frac{1}{7} \times \frac{1}{7} \right)\approx 0.061$$                                                 |
| 11  | $65,56$             | $$2\left( \frac{1}{7} \times \frac{1}{7} \right) \approx 0.041$$                                                |
| 12  | $66$                | $$\frac{1}{7} \times \frac{1}{7} \approx 0.020$$                                                                |
2. 计算 CDF 的累加：

| 数值  | 概率      | CDF     |
| --- | ------- | ------- |
| 2   | $0.020$ | $0.020$ |
| 3   | $0.041$ | $0.061$ |
| 4   | $0.102$ | $0.163$ |
| 5   | $0.122$ | $0.285$ |
| 6   | $0.163$ | $0.448$ |
| 7   | $0.163$ | $0.611$ |
| 8   | $0.143$ | $0.754$ |
| 9   | $0.122$ | $0.876$ |
| 10  | $0.061$ | $0.937$ |
| 11  | $0.041$ | $0.978$ |
| 12  | $0.020$ | $0.998$ |

求得 CDF 小于等于 0.5 的最高值为 6，最后 12 对应的结果近似等于 1。


```python
# Run this cell to submit your answer
utils.exercise_6()
```

## Exercise 7:

Given a 6-sided fair dice you try a new game. You only throw the dice a second time if the result of the first throw is **lower** or equal to 3. Which of the following `probability mass functions` will be the one you should get given this new constraint?

%%
给定一个公平的六面骰子。如果第一次投掷的结果小于等于 3，则可以再掷一次。在这个新的约束条件下哪个图为匹配的概率质量函数？
%%

<table><tr>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_green.png" style="height: 250px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_blue.png" style="height: 250px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_red.png" style="height: 250px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_brown.png" style="height: 250px;"/> </td>

</tr></table>

Hints:
- You can simulate the second throws as a numpy array and then make the values that met a certain criteria equal to 0 by using [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
%%
提示：
- 您可以将第二次投掷模拟为 numpy 数组，然后利用 np.where 函数将满足特定条件的值设为 0。
%%

### 蒙特卡洛模拟

```python
# 参数定义
n_sides = 6
n_rolls = 200_0000

# 创建骰子
dice = np.array([i for i in range(1, n_sides+1)])

# 投掷骰子
first_rolls = np.random.choice(dice, size=n_rolls)
second_rolls = np.random.choice(dice, size=n_rolls)

# np.where的第一参数为True返回第二参数，如果为False返回第三参数，最终结果为数组。
second_rolls = np.where(first_rolls <= 3, second_rolls, 0)

sum_of_rolls = first_rolls + second_rolls

# 绘制直方图
sum_2_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
sum_2_rolls_hist.set(title=f"Histogram of {n_rolls} Loaded Dice Rolls (Sum of 2)")
plt.show()
```

> [!result]
![灌铅骰子-Exercise7.png|600](https://obsidian-image.wwtt.xyz/2026/01/灌铅骰子-Exercise7.png)

### 解析解

已知每次投掷的概率为均匀的 $\frac{1}{6}$ 直接计算出和概率即可，需要注意 4、5、6 本身会组成概率的一部分。

| 数值  | 构成           | 概率                                                                    |
| --- | ------------ | --------------------------------------------------------------------- |
| 2   | $11$         | $$\frac{1}{6} \times \frac{1}{6} = 0.02\dot{7}$$                      |
| 3   | $12,21$      | $$2\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.0\dot{5}$$       |
| 4   | $13,22,31,4$ | $$3\left( \frac{1}{6} \times \frac{1}{6} \right)+\frac{1}{6} = 0.25$$ |
| 5   | $14,23,32,5$ | $$3\left( \frac{1}{6} \times \frac{1}{6} \right)+\frac{1}{6} = 0.25$$ |
| 6   | $15,24,33,6$ | $$3\left( \frac{1}{6} \times \frac{1}{6} \right)+\frac{1}{6} = 0.25$$ |
| 7   | $16,25,34$   | $$3\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.08\dot{3}$$      |
| 8   | $26,35$      | $$2\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.0\dot{5}$$       |
| 9   | $36$         | $$\frac{1}{6} \times \frac{1}{6} = 0.02\dot{7}$$                      |
最大值为4、5、6 ，且概率为 0.25，和第二幅图形一致。

```python
# Run this cell to submit your answer
utils.exercise_7()
```

## Exercise 8:

Given the same scenario as in the previous exercise but with the twist that you only throw the dice a second time if the result of the first throw is **greater** or equal to 3. Which of the following `probability mass functions` will be the one you should get given this new constraint?
%%
和前一个情景一致，但游戏规则有一些改变，只有第一次大于等于 3 才能投掷第二次。在这个新的约束条件下哪个图为匹配的概率质量函数？
%%

<table><tr>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_green2.png" style="height: 250px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_blue2.png" style="height: 250px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_red2.png" style="height: 250px;"/> </td>
<td> <img src=" https://obsidian-image.wwtt.xyz/2026/01/6_sided_cond_brown2.png" style="height: 250px;"/> </td>
</tr></table>

### 蒙特卡洛模拟

```python
# 参数定义
n_sides = 6
n_rolls = 200_0000

# 创建骰子
dice = np.array([i for i in range(1, n_sides+1)])

# 投掷骰子
first_rolls = np.random.choice(dice, size=n_rolls)
second_rolls = np.random.choice(dice, size=n_rolls)

# np.where的第一参数为True返回第二参数，如果为False返回第三参数，最终结果为数组。
second_rolls = np.where(first_rolls >= 3, second_rolls, 0)

sum_of_rolls = first_rolls + second_rolls

# 绘制直方图
sum_2_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
sum_2_rolls_hist.set(title=f"Histogram of Dice Rolls - Exercise 8")
plt.show()
```

> [!result]
![灌铅骰子-Exercise7.png|600](https://obsidian-image.wwtt.xyz/2026/01/灌铅骰子-Exercise8.png)

### 解析解

几乎和上面一题一模一样，每次投掷的概率还是均匀的 $\frac{1}{6}$ ，需要注意 1、2 本身会组成概率的一部分，同时由于投掷到 3 会再投掷一次的缘故，所以 3 不会存在。

| 数值  | 构成            | 概率                                                               |
| --- | ------------- | ---------------------------------------------------------------- |
| 1   | $1$           | $$\frac{1}{6}= 0.1\dot{6}$$                                      |
| 2   | $2$           | $$\frac{1}{6}= 0.1\dot{6}$$                                      |
| 4   | $31$          | $$\frac{1}{6} \times \frac{1}{6} = 0.02\dot{7}$$                 |
| 5   | $32,41$       | $$2\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.0\dot{5}$$  |
| 6   | $33,42,51$    | $$3\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.08\dot{3}$$ |
| 7   | $34,43,52,61$ | $$4\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.\dot{1}$$   |
| 8   | $35,44,53,62$ | $$4\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.\dot{1}$$   |
| 9   | $36,45,54,63$ | $$4\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.\dot{1}$$   |
| 10  | $46,55,64$    | $$3\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.08\dot{3}$$ |
| 11  | $65,56$       | $$2\left( \frac{1}{6} \times \frac{1}{6} \right) = 0.0\dot{5}$$  |
| 12  | $66$          | $$\frac{1}{6} \times \frac{1}{6} = 0.02\dot{7}$$                 |
结果和第四幅图片一致

```python
# Run this cell to submit your answer
utils.exercise_8()
```

## Exercise 9:

Given a n-sided fair dice. You throw it twice and record the sum. How does increasing the number of sides `n` of the dice impact the mean and variance of the sum and the covariance of the joint distribution?
%%
给定一个 n 面的公平骰子。投掷两次并记录和。随着 `n` 的增加，均值和方差有什么变化？对于两次投掷之间的协方差有什么变化。
%%
```python
def rolling(n_sides=6, n_rolls=200_0000):
    # # 参数定义
    # n_sides = 6
    # n_rolls = 200_0000

    # 创建骰子
    dice = np.array([i for i in range(1, n_sides+1)])

    # 投掷骰子
    rolls = np.random.choice(dice, size=n_rolls)

    return rolls

for i in range(2, 10):
    first_rolls = rolling(i)
    second_rolls = rolling(i)
    sum_of_rolls = first_rolls + second_rolls

    print(f"当骰子有{i}面时,")
    print(f"两次和的均值为: {np.mean(sum_of_rolls)}")
    print(f"两次和的方差为: {np.var(sum_of_rolls)}")
    print(f"两次之间的协方差为: : {np.cov(first_rolls, second_rolls)[0,1]}")
    print("")
```

### 蒙特卡洛模拟

> [!result]
> 	当骰子有 2 面时,
> 	两次和的均值为: 3.0006725
> 	两次和的方差为: 0.4999060477437501
> 	两次之间的协方差为: : -4.68560734280804e-05
> 
> 	当骰子有 3 面时,
> 	两次和的均值为: 4.000909
> 	两次和的方差为: 1.3330861737189992
> 	两次之间的协方差为: : 0.0003228152001575589
> 
> 	当骰子有 4 面时,
> 	两次和的均值为: 5.0015685
> 	两次和的方差为: 2.4981170398077492
> 	两次之间的协方差为: : -0.0009815305247652003
> 
> 	当骰子有 5 面时,
> 	两次和的均值为: 6.001444
> 	两次和的方差为: 4.002652914864
> 	两次之间的协方差为: : 0.001596979514739746
> 
> 	当骰子有 6 面时,
> 	两次和的均值为: 6.9971675
> 	两次和的方差为: 5.838191476943751
> 	两次之间的协方差为: : -0.00023804313152163192
> 
> 	当骰子有 7 面时,
> 	两次和的均值为: 7.9998275
> 	两次和的方差为: 7.992660470243755
> 	两次之间的协方差为: : -0.005648258140629031
> 
> 	当骰子有 8 面时,
> 	两次和的均值为: 9.001169
> 	两次和的方差为: 10.500955633438998
> 	两次之间的协方差为: : 0.0012394063397033143
> 
> 	当骰子有 9 面时,
> 	两次和的均值为: 10.0015445
> 	两次和的方差为: 13.34929211451975
> 	两次之间的协方差为: : 0.002982423881211643
> 

### 解析解

设两次投掷的概率为 $X_1$ 和 $X_2$，每面的概率则为 $\frac{1}{n}$。
#### 期望

根据离散期望的计算公式 $\mathbb{E}[X]=\sum_{i}x_{i} \cdot P(X=x_{i})$，即 $\frac{1+2+...+n}{n}$，当 $n+1$ 时，则式子变为 $\frac{1+2+...+n+(n+1)}{n+1}$，单个骰子的期望值递增。同时根据期望的线性性 $\mathbb{E}[X_{1}+X_{2}]=\mathbb{E}[X_{1}]+\mathbb{E}[X_{2}]$ ，两次投掷的期望值同样递增。
#### 方差

根据上面期望递增的结论，同时方差公式为 $Var(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$，且 $\mathbb{E}[X^2]>\mathbb{E}[X]^2$，可以得出两次投掷的方差同样递增。
### 协方差

由于第一次和第二次相互独立，所以它们没有相关性，协方差结果近似为 0 且保持不变。

```python
# Run this cell to submit your answer
utils.exercise_9()
```

## Exercise 10:

Given a 6-sided loaded dice. You throw it twice and record the sum. Which of the following statemets is true?

%%
给定一个六面的灌铅骰子。投掷两次并计算它们的和。以下描述哪个是正确的？
%%

```python
# You can use this cell for your calculations (not graded)


```

题目如下：
- the mean and variance is the same regardless of which side is loaded
- having the sides 3 or 4 loaded will yield a higher covariance than any other sides
- the mean will decrease as the value of the loaded side increases
- changing the loaded side from 1 to 6 will yield a higher mean but the same variance

%%
- 无论哪一面被加载，均值和方差都相同  
- 当加载第 3 或第 4 面时，协方差比其他任何面都高  
- 随着加载面数值的增大，均值会下降  
- 将加载面从 1 改为 6 会得到更高的均值，但方差保持不变
%%

直觉认为正确的是第四个。

```python
# Run this cell to submit your answer
utils.exercise_10()
```

## Exercise 11:

Given a n-sided dice (could be fair or not). You throw it twice and record the sum (there is no dependance between the throws). If you are only given the histogram of the sums can you use it to know which are the probabilities of the dice landing on each side?
%%
给定一个 n 面的骰子（不知道是否灌铅）。投掷两次记录它们的和（投掷之间没有依赖关系）。如果只有一个骰子和的直方图，是否可以通过这个图来确定每个面的概率？
%%
In other words, if you are provided with only the histogram of the sums like this one:
%%
换句话说，如果有下面一个直方图
%%

![hist_sum_6_side.png|500](https://obsidian-image.wwtt.xyz/2026/01/hist_sum_6_side.png)


Could you use it to know the probabilities of the dice landing on each side? Which will be equivalent to finding this histogram:

%%
能否判定每个骰子的概率，就像上图对应的下图一样。
%%

![hist_sum_6_side.png|500](https://obsidian-image.wwtt.xyz/2026/01/fair_dice.png)


```python
# You can use this cell for your calculations (not graded)


```

- yes, but only if one of the sides is loaded
- no, regardless if the die is fair or not
- yes, but only if the die is fair
- yes, regardless if the die is fair or not

%%
- 是的，但仅限于骰子某一面被加重的情况  
- 不，无论骰子是否公平  
- 是的，但仅限于骰子公平的情况  
- 是的，无论骰子是否公平
%%

直觉认为正确的是第三个。

错误，最终 Gemini 给出的结果如下：

> [!Gemini Result]
> 这道题的答案是 **“不能”（No）**。
> 
> 1.  **多对一的关系**：虽然骰子每面的概率决定了“和”的概率分布（数学上称为**卷积**），但不同的侧面概率配置可能会产生完全相同的“和”分布直方图。
> 2.  **不可逆性**：仅仅观察“和”的结果，你会丢失关于单次投掷的具体信息。由于存在多种可能性产生相同的加和效果，你无法仅凭直方图就唯一、确定地逆推回原始骰子每一面的概率。
> 
> 这道题旨在测试你对概率分布性质的直觉，即**结果的分布并不能总是唯一确定过程的参数**。


```python
# Run this cell to submit your answer
utils.exercise_11()
```

## Before Submitting Your Assignment

Run the next cell to check that you have answered all of the exercises


```python
utils.check_submissions()
```

**Congratulations on finishing this assignment!**

During this assignment you tested your knowledge on probability distributions, descriptive statistics and visual interpretation of these concepts. You had the choice to compute everything analytically or create simulations to assist you get the right answer. You probably also realized that some exercises could be answered without any computations just by looking at certain hidden queues that the visualizations revealed.

**Keep up the good work!**

