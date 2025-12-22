---
aliases: ["Ungraded Lab: Birthday Problems"]
tags: []
created: 2025-12-12, 19:16:39
modified: 2025-12-14, 19:41:13
---

# Ungraded Lab: Birthday Problems

Welcome! During this lab you will reinforce the notion of how counter-intuitive probabilities can be by taking a look at the famous birthday problem. In fact you will take a look at 4 variations of this problem. You can use one you have already seen the solution for, to try and come up with the solution for the next one, the results might surprise you!

%%
在本实验中，你将通过探讨著名的生日问题，加深对概率反直觉特性的理解。
事实上，你会看到这个问题的四种变体。
你可以使用一个你已经见过的方案，尝试给另一个问题找到解决办法，结果会让你吃惊的。
%%

Let's get started!


```python
import numpy as np
import matplotlib.pyplot as plt
import utils

%matplotlib widget
```

## Introduction to the problem

All of these problems share a similar setting. You have a classroom full of students (the number may vary) and want to know the probabilities of two students having the same birthday or of any student having a particular birthday, anything along those lines. As mentioned before, you will see 4 variations of the problem.
%%
这些问题都有一个相似的背景。
一个坐满学生的教室（人数可能不同），想知道有没有可能有两个学生有相同的生日，或者有没有学生有特别的生日，都是类似的。
如前所述，你将会看到四种这样的问题。
%%

You can think of these problems in two ways:
   - What is the minimum number of students `n` that need to be in the classroom to have a matching birthday with a given probability?
   - Given `n` what is the probability of having a match?
%%
你可以从两个角度来考虑这些问题：
	- 在教室里至少有多少个学生 `n` 才有可能有一对学生匹配生日。
	- 当有了 `n`，匹配的概率是多少？
%%

Both ways model the situation from different angles but they are essentially covering the same.
%%
两种方式从不同的角度对情况进行建模，但本质上它们所涵盖的内容是相同的。
%%
## Play the game of matching your birthday

To further motivate this situation a game is presented. You can use the following cell to run an interactive game, it is very simple to use: you need to select your birthday (the year does not matter) in the dropdown widget and then you can click the `Simulate!` button to randomly create students until one of them has the same birthday as you. The left plot shows you the history of the result for each simulation and the right plot shows you the same information in a histogram so you can see how this variable distributes.
%%
为了进一步，现在引入一个游戏。
使用下面的单元格运行这个游戏，非常简单：你需要在下拉组件中选择你的生日（不用在意年份），然后点击 `Simulate!` 按钮随机生成学生，直到其中一位的生日与你的生日相同相同。
左侧的表格绘制了每一次模拟的结果，而右侧图表则以直方图形式呈现相同信息，以便您观察该变量的分布情况。
%%

You can try this for as long as you want so you get a sense of the probability distribution for this process (it is recommended to try it for at least 30 runs):

%%
你可以随意尝试多次，以便了解这一过程的概率分布（建议至少进行 30 次运行）。
%%

```python
game = utils.your_bday()
```

下面是随机了 100 次的结果：
![生日问题-100次随机.png](https://obsidian-image.wwtt.xyz/2025/12/生日问题-100次随机.png)
## First Problem

The first problem tries to answer the question: given a pre-defined date, what is the value of `n` such that the probability of having a match is greater than or equal to 0.5?
%%
第一个问题是尝试回答这个问题：给定一个预设的日期，需要多少样本 `n` 才能让样本中的日期等于这个日期的概率大于等于 0.5？
%%

![生日问题-first.png|700](https://obsidian-image.wwtt.xyz/2025/12/生日问题-first.png)


Before taking a look at the analytical solution you will try to solve it by creating simulations with Python. For this purpose you can use the `simulate` helper function provided in the next cell. Run it to load this function which will be used shortly:
%%
在查看解析解之前，你需要尝试通过 Python 来模拟解决。
为了达成这个目的你可以使用这个下面单元格提供的 `simulate` 辅助函数。
运行它以加载此函数，稍后将使用。
%%

```python
def simulate(problem_func, n_students=365, n_simulations=1000):
    
    # Initialize the counter of matches at 0
    matches = 0
    
    # Run the simulation for the desired number of times
    for _ in range(n_simulations):
        
        # If there is a match in the classroom add 1 to the counter of matches
        if problem_func(n_students):
            matches += 1
    
    # Return the ratio of number of matches / number of simulations
    return matches/n_simulations
```

This function returns the simulated probability for a given problem when you pass to it the number of students and the number of simulations and it has these two properties:

   - The higher the number of students the higher the probability of a match.
   - The higher the number of simulations the more accurate the simulated probability will be (This fact will be discussed further in Week 3. This is known as the Central Limit Theorem). 

%%
此函数在传入学生人数和模拟次数后，会返回给定问题的模拟概率，并具备以下两个特性：
- 学生的人数越多，匹配的概率越大。
- 模拟的次数越多，模拟的概率越准确。（这件事会在第三周讨论，这被称为中心极限定理）
%%

This is pretty cool but how do you use this helper function? You need to pass another function to it that models the situation at hand. This other function should have two criteria so that `simulate` works as expected:

   - It should receive the number of students as input.
   - It should return True if there was a match or False otherwise.

%%
如何使用这个辅助函数？你需要传入另外一个函数，该函数用于对模拟当前的环境。另一个函数应该有两个符合 `simulate` 工作的标准条件
- 它接受学生的数量作为输入条件。
- 如果匹配则返回 True，否则返回 False。
%%
	
You can create a function that models problem number 1 by running the following cell:

```python
def problem_1(n_students):
    
    # Predefine a specific birthday
    predef_bday = np.random.randint(0, 365)
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Check if predefined bday is among students
    return predef_bday in gen_bdays
```

Now you can use these two functions in conjuction to get the probability of a match for a given classroom size. Notice that you can tweak the value of the `n` variable to simulate classrooms with different number of students. Also notice that this time the simulation is run 10,000 times instead of the default 1000. This gives you a more accurate simulated probability at the expense of taking longer to execute:

%%
现在，你可以结合使用这两个函数来计算特定教室规模的匹配概率。
注意你可以调整这个变量 `n` 的值来模拟不同数量的学生在这个教室里的情况。
同时也要注意本次模拟运行了 10000 次而不是默认的 1000 次。
这会给你更准确的模拟概率，但代价是会花费更长的时间：
%%

```python
n = 100 # try changing this value!
simulated_prob = simulate(problem_1, n_students=n, n_simulations=10_000)

print(f"The simulated probability of any student to have a bday equal to a predefined value is {simulated_prob} in a classroom with {n} students")
```

> [!result]
> The simulated probability of any student to have a bday equal to a predefined value is **0.2411** in a classroom with **100** students

This is very cool but it still has one major drawback: you would need to try a bunch of values for `n` before arriving at the solution. Instead of taking this approach you can generate a plot that shows the simulated probability as a function of the number of students in the classroom:

%%
这里还是有一个主要的缺点：你需要尝试对 `n` 做一系列的取值才能得到解。
不如生成一幅图表，展示模拟概率随教室中学生人数变化的情况：
%%

```python
# Generate the simulated probability for every classroom
simulated_probs_1 = [simulate(problem_1, n_students=n) for n in utils.big_classroom_sizes]

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_1, utils.big_classroom_sizes)
```

> [!result]
> ![生日问题-概率与学生数量的对比-1.png](https://obsidian-image.wwtt.xyz/2025/12/生日问题-概率与学生数量的对比-1.png)

Remember that this approach is a simulation and thus you are generating simulated (or approximated) probabilities. Because of this the curve is not completely smooth and you will get slightly different values every time you run the simulation.
%%
请记住，这个方法是模拟的，因此你生成的是模拟（或者近似）的概率。所以这条曲线并不是平滑的，并且每次运行的结果都略微的不同。
%%
## Analytical Solution

Now that you have built a stronger intuition, let's calculate explicitily the probability $P$ that at least one student in the room has the birthday the same as the pre-defined date. It is clear that $P = P(n)$, i.e., the value for $P$ depends on the number of students in the room and, as $n$ become large, $P(n)$ must become closer to $1$. With a formula for $P(n)$ we can then find the minimum $n$ such that $P(n) \geq 0.5$. Let's suppose that a year has $365$ days.

%%
现在你应该建立起了更强的直觉（有吗？），让我们来计算这个可能性 $P$ ，也就是在这个教室里至少有一名学生的生日和这个预设的日期相同的概率。显然，$P = P(n)$，也就是说，$P$ 的值取决于房间里的学生的数量，如果 $n$ 越大，那么 $P(n)$ 就越接近 1。有了 $P(n)$ 的公式，我们便能求得当 $P(n) \geq 0.5$ 的时候 $n$ 的最小值。
%%

Let's consider $D$ the pre-defined birthday and suppose a student is selected at random. 
%%
预设的生日日期为 $D$，假设选择的学生是随机的。
%%

Defining the event $S_i$ as the $i$ -th student has birthday in the day $D$. Then $P(S_i) = \frac{1}{365}$, because there are $365$ equally likely possibilities for their birthday. So, using the **complement rule**, the probability that this student's birthday isn't day $D$ is $P(S_i^c) = 1 - P(S_i) = 1 - \frac{1}{365}$. 
%%
定义事件 $S_i$，第 $i$ 个学生的生日就是日期 $D$。因为他们的生日有 365 种可能性，那么 $P(S_i)=\frac{1}{365}$。然后根据**补集规则**，这些学生的生日不可能为日期 $D$ 的概率则为：$P(S_i^c) = 1 - P(S_i) = 1 - \frac{1}{365}$。
%%

Note that this probability is the same for any student and we can fairily assume that each student's birthday is independent from each other. Consider the event $\mathcal{S}$ the desired event, i.e., at least one student has birthday in day $D$. Note that:
%%
注意，这个概率对于任何学生而言都是相同的，我们可以公平的假设每个学生的生日都是相互独立的。考虑到事件 $S$ 为期望发生的事件，也就是说最后一个学生的生日就是日期 $D$
%%

$\mathcal{S}^c$ is the probability that **no student has birthday in day $D$** and this is the same as:
%%
$\mathcal{S}^c$ 是**没有学生在日期 D 过生日**的概率，那么：
%%
- Student $1$ has birthday in a day different than D, AND
- Student $2$ has birthday in a day different than D, AND,
...
- Student $k$ has birthday in a day different than D.
%%
- 第一个学生的生日不是日期 $D$，**并且**
- 第二个学生的生日不是日期 $D$，**并且**
- ...
- 第 $k$ 个学生的生日不是日期 $D$。
%%

With our definitions, this is just $S_1^c \cap S_2^c \cap \ldots \cap S_k^c.$ Therefore

%%
根据我们的定义，它其实就是 $S_1^c \cap S_2^c \cap \ldots \cap S_k^c$。由此可以得到：
%%
$$
\begin{equation}
\begin{split}
P(\mathcal{S}) {} & = 1 - P(\mathcal{S}^c) \\
              & = 1 - P(S_1^c \cap S_2^c \cap \ldots \cap S_k^c) \\
              & = 1 - P(S_1^c)P(S_2^c) \cdots P(S_k^c) \text{ (independence)}\\
              & = 1 - (1 - \frac{1}{365})^n.
\end{split}
\end{equation}
$$

As you've expected, $P(\mathcal{S}) = 1 - (1 - \frac{1}{365})^n = P(n)$. Now, you are ready to answer the question: for wich value of $n$, $P(n) \geq \frac{1}{2}$?
%%
正如你想到的，$P(\mathcal{S}) = 1 - (1 - \frac{1}{365})^n = P(n)$。现在你准备好回答这个问题了：当 n 的值为多少的时候，$P(n) \geq \frac{1}{2}$？
%%

Well, $P(n) \geq \frac{1}{2}$ is equivalent to
%%
 好吧，$P(n) \geq \frac{1}{2}$ 等同于：
%%

$$
\begin{align}
 1 - \left(1 - \frac{1}{365}\right)^n &\geq \frac{1}{2} \textit{, passing 1 to the other side and inverting the inequality sign}\\
\left(1 - \frac{1}{365}\right)^n &\leq \frac{1}{2}\\
\left(\frac{364}{365}\right)^n &\leq 2^{-1} \\
\ln{\left(\frac{364}{365}\right)}^n &\leq \ln{2^{-1}} \\
n \ln{\left(\frac{364}{365}\right)} &\leq -\ln{2}\\
\end{align}
$$
Now, using a calculator we can easily find that $\ln{2} \approx 0.693$ and $\ln{\frac{364}{365}} \approx -0.003$, the last inequality becomes
%%
现在，通过计算我们可以简单的求得右边是 $\ln{2} \approx 0.693$，而左边是 $\ln{\frac{364}{365}} \approx -0.003$，那么这个不等式变为了：
%%
$$n \cdot -0.003 \leq -0.693,$$

which is equivalent to $n \geq \frac{0.693}{0.003} = 253$. 
%%
那么这相当于 $n \geq \frac{0.693}{0.003} = 253$
%%
## Second Problem

The second problem is very similar to the first one, with the difference that the predefined value is not previously defined but it is drawn from one of the students at random so it can be worded like this: given a classroom with `n` students, if you draw any student at random what is the value of `n` such that the probability of having a match with another student is greater than or equal to 0.5?

第二个问题和第一个问题非常相似，不同点的在于预定义值并没有事先设定好，而是从学生中随机抽取一位来作为预定值，因此可以这样表述：一个教室中有 `n` 个学生，随机抽取任意一个学生，此时刚好有另外一个学生的生日刚好和这个随机抽取的学生一样，如果这个概率大于等于 0.5，此时至少有多少个学生？

![生日问题-second.png|700](https://obsidian-image.wwtt.xyz/2025/12/生日问题-second.png)


You can reuse the `simulate` helper function defined earlier so the only thing left is to code the function that models this particular problem. **But before doing this try to come up with a hypothesis about the result. What do you think will happen? Will `n` be similar to the previous one or do you need a higher value? What about a smaller value?**
%%
你可以重用之前定义的这个 `simulate` 辅助函数，因此剩下的唯一的事情就是编写一个模拟这个问题的函数。但是在此之前尝试对结果提出一个假设。你认为会发生什么？`n` 的值会和前一个问题相似或者更大？还是说会更小？
%%
Run the next cells to find out!
%%
运行这个单元格找到它！
%%
```python
def problem_2(n_students):
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Pick one student at random
    rnd_index = np.random.randint(0, len(gen_bdays))
    
    # Get the bday from the selected student
    rnd_bday = gen_bdays[rnd_index]
    
    # Take the bday out of the pool of bdays (otherwise there is always a match)
    remaining_bdays = np.delete(gen_bdays, rnd_index, axis=0)
    
    # Check if another student shares the same bday
    return rnd_bday in remaining_bdays
```


```python
# Generate the simulated probability for every classroom
simulated_probs_2 = [simulate(problem_2, n_students=n) for n in utils.big_classroom_sizes]

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_2, utils.big_classroom_sizes)
```
> [!result]
> ![生日问题-概率与学生数量的对比-2.png](https://obsidian-image.wwtt.xyz/2025/12/生日问题-概率与学生数量的对比-2.png)

## Analytical Solution

Note that this problem is very similar to the first one. The difference is that, instead of selecting one day $D$ at random from the room, you select a student and fix their birthday $D$. You can reduce this problem to the previous one by removing this student from the room and considering now a room with $n-1$ students. The problem now is analogous to the previous one but you will end up with a  $n-1$ instead of $n$. Therefore, it is easy to see that, in this case,
%%
注意，这个问题和第一个问题非常的相似。唯一的不同是在教室里选择了一个日期 $D$，你选择一个学生，然后用他的生日作为 $D$。你可以将这个问题简化为上一个问题，从教室中将这个学生去掉，然后考虑现在的教室中的学生数量是 $n-1$。那么这个问题和上一个问题差不多了，但是最终结果变成了 $n-1$ 而不是 $n$。由此，从这个情况下，很容易得出：
%%
$$P(n) = 1 - \left(1 - \frac{1}{365} \right)^{n-1}.$$

With some calculations you get that $P(n) \geq \frac{1}{2}$ if and only if $n \geq {\frac{0.693}{0.003}} + 1 = 254$.
%%
通过计算，当 $n \geq {\frac{0.693}{0.003}} + 1 = 254$ 的时候， $P(n) \geq \frac{1}{2}$。
%%
## Third Problem

The third one is the most famous of all the birthday problems and it was covered in the lectures in a sligthly different way.
%%
第三个问题是所有生日问题中最著名的，讲座中对它的讲解方法略微有些差异。
%%

This time you don't want to find a match with a predefined value but rather to find a match between any two birthdays, it can be worded like this: given a classroom with students, what is the value of `n` such that the probability of having a match is greater than or equal to 0.5 for any two students?
%%
这次你不需要去寻找预设值，而是要找到两个任意匹配的生日，它可以这样描述：一个教室中有 `n` 个学生，当 `n` 最少为多少的时候，两个学生的生日相同的概率为 0.5。
%%

![生日问题-third.png|700](https://obsidian-image.wwtt.xyz/2025/12/生日问题-third.png)


Note that, in the lectures, it was calculated the probability that **no students share a birthday**. Here, you are dealing with the case that, **at least two students share a birthday**, which is the *complement* of the question discussed in the lecture.
%%
注意，在讲座中，计算的是**没有学生生日相同的概率**。这里你正在处理的情况是，至少**有两个学生的生日相同**，这是讲座中讨论问题的*补充*内容。
%%

Before doing the simulation as with previous problems ask yourself: **Do you think that the value of `n` will be similar to that of the previous problems? If you have to guess would you say it needs to be greater than or lower?**
%%
在这个模拟开始前，有一个关于上个问题的提问：你认为这个 `n` 的值和上一个问题差不读吗？如果必须有个答案，你认为它是比之前大还是比之前小？
%%

To help you out run the next cell to play an interactive version of this problem. The instructions are simple:

   - To start a new simulation click anywhere on the upper panel (just below where the `Figure` headline appears)
   - The upper panel shows randomly generated birthdays and let's you know when there is a match between two students
   - The bottom left panel keeps track of the number of students required to have a match for every run
   - The bottom right panel shows that same information as a histogram
   - **Try running the simulation at least 30 times to get a sense of how this particular problem behaves**

%%
为了帮助你解决这个问题，请运行下面的单元格来体验此问题。说明也很简单：

- 点击面板上任意位置的开始运行模拟（`Figure` 标题的下方）。
- 面板将随机生成的生日并显示，当有两名学生匹配成功的时候会通知你。
- 面板左下方将跟踪每次运行结束时学生的数量。
- 面板右下方以直方图显示左侧相同的信息。
- **至少运行 30 次模拟，以了解此特定问题的行为模式** 。
%%


```python
game_third_prob = utils.third_bday_problem()
```

> [!result]
![生日问题-第二次随机.png](https://obsidian-image.wwtt.xyz/2025/12/生日问题-第二次随机.png)

Now you should have a hypothesis of the number of students in the classroom needed for the match. Test your intuition by generating the simulated probabilities as before:
%%
现在你应该对这个教室中至少有多少学生的数量有个假设了，通过生成模拟的概率来验证你的直觉，方法和前面相同：
%%

```python
def problem_3(n_students):
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Get array containing unique bdays
    unique_bdays = np.array(list(set(gen_bdays)))
    
    # Check that both the original and unique arrays have the same length 
    # (if so then no two students share the same bday)
    return len(unique_bdays) != len(gen_bdays)
```


```python
# Generate the simulated probability for every classroom
simulated_probs_3 = [simulate(problem_3, n_students=n) for n in utils.small_classroom_sizes]

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_3, utils.small_classroom_sizes)
```

> [!result]
> ![概率与学生数量的对比-3.png](https://obsidian-image.wwtt.xyz/2025/12/概率与学生数量的对比-3.png)

## Analytical solution

This problem is a bit different from the previous two, so you will need to make some calculations again. Now, the idea is to perform the calculation by steps, selecting one student at time. So let's define $Q_i$ as the probability that the $i$ -th student has birthday different from the previous students. Note that $Q_1 = 1$, because there is no previous student. Note also that,
%%
这个问题和前两个问题有所不同，所以你需要再次做一些计算。现在，思路是分步进行计算，一次选择一个学生。所以让我们定义 $Q_i$ 为第 $i$ 个学生，它的生日自然与前面的学生不同。注意当 $Q_1 = 1$ 的时候，因为此时前面没有学生。同时也要注意：
%%
$$Q_2 = \frac{364}{365}.$$

This is because, given that you've selected one student, the second has $364$ possible values, it has a chance of $\frac{364}{365}$ of not matching birthdays with the first. Now, for $Q_3$, there are $2$ selected students with different birthdays, so the probability that the third student not matches any of the previous two is:
%%
这是因为，既然你选择第一个学生，那么第二个学生就有 $364$ 种可能性，它就有 $\frac{364}{365}$ 的机会不会匹配第一个学生。对于 $Q_{3}$，他就会和前面两个学生的生日有所不同，所以第三个学生不匹配前面两个学生中任意一个的概率是：
%%
$$Q_3 = \frac{363}{365}.$$

Inductively, if we select $n$ students, then 
%%
以此类推，如果我们选择了 $n$ 个学生，那么
%%
$$Q_n = \frac{365 - (n-1)}{365}.$$

Since we assume that we are choosing students independently from each other, then the probability of picking $n$ students that **don't have their birthday in common** is just the product of all $Q_n$. Let's call it $Q$, so:
%%
我们假设我们选择的学生彼此相互独立，那么，选出 `n` 个生日不重复的学生的概率，就是所有 $Q_n$ 的乘积。我们称之为 $Q$，因此：
%%
$$Q = Q_1 \cdot Q_2 \cdot \ldots \cdot Q_{n-1} \cdot Q_n = 1 \cdot \frac{364}{365} \cdot \frac{363}{365} \cdot \ldots \cdot \frac{365 - (n-2)}{365} \cdot \frac{365 - (n-1)}{365}.$$

The desired probability is, therefore, $P:= 1 - Q$, since we want the probability of **at least two students match their birthday** (this is just the complement rule).

%%
由于我们想要得到的是**最后两个学生他们生日相同**的概率，那么期望的概率就是 $P:= 1 - Q$。（互补规则的应用）
%%

Note that we could just write a small program in Python to compute this value for every $n$ and return the first value that achieves the inequality we want ($P \geq \frac{1}{2}$), but for sake of completion we will provide an analytic solution.
%%
注意，我们可以直接用 Python 编写一个小程序来计算每个 n 对应的值，并返回满足我们所需不等式 ($P \geq \frac{1}{2}$) 的第一个值，但为了完整性，我们将提供一个解析解。
%%

We will use the following approximation: $1 - x \approx e^{-x}$ for $x$ small and positive. We can re-write $Q$ as
%%
我们将使用以下近似值：$1 - x \approx e^{-x}$ ，当 $x$ 较小且为正。我们可以将 Q 重写为
%%
$$Q = 1 \cdot \frac{364}{365} \cdot \frac{363}{365} \cdot \ldots \cdot \frac{365 - (n-2)}{365} \cdot \frac{365 - (n-1)}{365} 
    = \left(1 - \frac{1}{365} \right) \cdot \left(1 - \frac{2}{365} \right) \cdot \ldots \cdot \left(1 - \frac{n-2}{365} \right) \cdot \left(1 - \frac{n-1}{365} \right).$$

Thus, using the approximation:
%%
由此可以得到近似值：
%%
$$Q \approx e^{-\frac{1}{365}} \cdot e^{-\frac{2}{365}} \cdot \ldots \cdot e^{-\frac{n-2}{365}} \cdot e^{-\frac{n-1}{365}} = e^{- \frac{1 + 2 + \ldots + (n-1)}{365}}.$$

Using the formula $1 + 2 + \ldots + (n-1) = \frac{n(n-1)}{2}$ (this is the sum of the first $n-1$ terms of a arithmetic progression with first term $1$ and common difference $1$), we have:
%%
带入式子 $1 + 2 + \ldots + (n-1) = \frac{n(n-1)}{2}$ （等差数列）我们得到：
%%
$$Q \approx e^{-\frac{n(n-1)}{730}}.$$

So, $P = 1 - Q = 1 - e^{-\frac{n(n-1)}{730}}$ and $P \geq \frac{1}{2}$ is equivalent to $1 - e^{-\frac{n(n-1)}{730}} \geq \frac{1}{2}$ which is equivalent to $\frac{n(n-1)}{730} \geq \ln 2$, i.e., $n(n-1) \geq 730 \cdot \ln 2$. Since $\ln 2 \approx 0.6931$, so 
%%
所以，$P = 1 - Q = 1 - e^{-\frac{n(n-1)}{730}}$，并且当 $P \geq \frac{1}{2}$ 的时候，替换 P 可以得到： $1 - e^{-\frac{n(n-1)}{730}} \geq \frac{1}{2}$，化简后得到 $\frac{n(n-1)}{730} \geq \ln 2$，也就是说 $n(n-1) \geq 730 \cdot \ln 2$。由此可得 $\ln 2 \approx 0.6931$，那么：
%%
$$n(n-1) \geq 505.96 \geq 505.$$

Solving the quadratic equation $n(n-1) = 505$, the only positive value for $n$ is $n = \frac{1 + \sqrt{2021}}{2} \approx 23.$
%%
解二次方程 $n(n-1) = 505$，也就是 $n^2-n=505$， 得到的唯一正数解为 $n = \frac{1 + \sqrt{2021}}{2} \approx 23$。
%%

Therefore, if $n\geq23$ then $P \geq \frac{1}{2}$.
%%
由此可得，当 $n\geq23$ 时 $P \geq \frac{1}{2}$
%%
## Fourth Problem

The fourth and final one is similar to the third problem but with the difference that you have two classrooms and want to find a match between a student in one classroom and a student in the other, presenting it like a question it will be: given two classrooms with `n` students, what is the value of `n` such that the probability of having a match is greater than or equal to 0.5 for any two students in each classroom?
%%
第四个同时也是最后一个问题和第三个问题类似，不同的点在于你有两个教室，此时想要寻找其中一个班级学生和另一个班级匹配的学生的概率，那么也可以这样描述：给定两个教室各有 `n` 名学生，`n` 为何值时，每间教室中任意两名学生生日相同的概率大于或等于 0.5？
%%

![生日问题-fourth.png|700](https://obsidian-image.wwtt.xyz/2025/12/生日问题-fourth.png)


**Once again try to come up with your own hypothesis before doing the simulation!**
%%
再次尝试在进行模拟之前提出自己的假设！
%%

```python
def problem_4(n_students):
    
    # Generate birthdays for every student in classroom 1
    gen_bdays_1 = np.random.randint(0, 365, (n_students))
    
    # Generate birthdays for every student in classroom 2
    gen_bdays_2 = np.random.randint(0, 365, (n_students))
    
    # Check for any match between both classrooms
    return np.isin(gen_bdays_1, gen_bdays_2).any()
```


```python
# Generate the simulated probability for every classroom
simulated_probs_4 = [simulate(problem_4, n_students=n) for n in utils.small_classroom_sizes]

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_4, utils.small_classroom_sizes)
```

> [!result]
> ![概率与学生数量的对比-4.png](https://obsidian-image.wwtt.xyz/2025/12/概率与学生数量的对比-4.png)

## Analytical solution

The solution to this problem is similar to the first one. Now, instead of only **one** date, there are $n$ dates to compare. 
Remember that if we have only one date $D$ to compare than the probability, let's say $Q_1$ of having **no** student with birthday $D$ is $Q_1 = (1 - \frac{1}{365})^n$ (the complement of $P(\mathcal{S})$ in that case). Now we proceed as problem three, by having independent samples of students. For each student sampled, the probability $Q_i$ is $(1 - \frac{1}{365})^n$, so the probability of no student matches any of the $n$ given dates is therefore
%%
这个问题的解决方法类似第一个问题。现在不仅仅限于一个日期，这里有 $n$ 个日期进行配对。
如果我们只需要有一个日期 $D$ 进行比较，同时我们设没有学生生日为 $D$ 的概率为 $Q_1$，则 $Q_1 = (1 - \frac{1}{365})^n$（这种情况下，Q 为 $P(\mathcal{S})$ 的补集）。
现在我们按问题 3 进行处理模式，独立抽取学生样本。
每个学生的样本，它的概率 $Q_i$ 是 $(1 - \frac{1}{365})^n$，所以没有学生匹配日期 $D$ 的情况下的概率为：
%%
$$Q = Q_1 \cdot Q_2 \cdot \ldots \cdot Q_{n-1} \cdot Q_n = (1 - \frac{1}{365})^{n^2}$$

Using the approximation $1 - x \approx e^{-x}$ for $x$ small,
%%
使用近似值 $1 - x \approx e^{-x}$ 
%%
$$Q \approx e^{-\frac{n^2}{365}}$$

Therefore, $$P(n) \approx 1 - e^{-\frac{n^2}{365}}.$$ 

Thus, $P(n) \geq \frac{1}{2}$ if $n \geq \sqrt{\ln 2 \cdot 365} \approx 15.9 \geq 15$
%%
当 $n \geq \sqrt{\ln 2 \cdot 365} \approx 15.9 \geq 15$ 时， $P(n) \geq \frac{1}{2}$。
%%
**Congratulations! You have finished the ungraded lab on the Birthday problems!**
