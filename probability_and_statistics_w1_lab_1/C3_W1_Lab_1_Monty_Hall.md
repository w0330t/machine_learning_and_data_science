---
aliases: [Ungraded Lab - The Monty Hall Problem]
tags: []
created: 2025-12-16, 13:17:06
modified: 2025-12-17, 15:31:46
banner: https://obsidian-image.wwtt.xyz/2025/12/monty_hall_doors.svg
banner-x: 52
banner-y: 59
content-start: 286
---

# Ungraded Lab - The Monty Hall Problem

Welcome! In this ungraded lab you will see the counter intuitive nature of probability by studying the famous Monty Hall problem. This problem seems very trivial at first glance but it exemplifies the fact that probabilities can have behaviours you might not initially expect.

%%
在这个不计分的实验中你将通过学习研究著名的蒙提霍尔问题来探讨概率反直觉的本质。
乍看之下，这个问题似乎微不足道，但它恰恰说明了概率可能展现出你最初意想不到的特性。
%%

Begin by importing the required libraries for the lab:


```python
import numpy as np
import utils

%matplotlib widget
```

## 1. Problem introduction

Suppose you are in a TV show where you may win a car by playing a game. The game is very simple: you have to choose among three closed doors. One door has the car and the other two have goats.

%%
假设你在某个电视综艺节目中玩一个游戏，奖品是一辆汽车。
这个游戏非常的简单：在三扇门中进行选择。
一扇门后面是汽车，另外两扇门后则是山羊。
%%

The game is played in two steps:

1. The host lets you choose one among the three doors, but you do not open it yet.
2. Then, the host (who knows where the car is) choose one among the two remaining doors and open it, revealing a goat. 


%%
这个游戏分两步：
1. 主持人让你在三扇门中选择一扇门，但是你不能打开它。
2. 然后，主持人（他知道门后的奖品）在剩下的两扇门中选择选择一扇后面是山羊的门，然后打开它。
%%

![monty_hall_doors.svg|500](https://obsidian-image.wwtt.xyz/2025/12/monty_hall_doors.svg)

The time to choose has come and let's suppose you have chosen door number 1. Then, just before they open the door number 1, the Host - who already knows in which door the car is behind - opens door number 3, revealing a goat and leaving doors number 1 and 2 closed. The Host then asks you:

%%
在选择门的时候，假设你选择了 1 号门。
然后在即将打开 1 号门之前的那一刻，主持人（他知道门后的奖品）打开了三号门，他揭示了一只山羊，此时 1 号门和 2 号门依然紧闭。主持人随后向你提问：
%%

**"Would you like to switch your choice to door number 2?"**

%%
你想要改变你的选择吗？要选择 2 号门吗？
%%

This question seems weird, since the host knows which door is the winner, maybe he is trying to trick you into choosing poorly. What would you do? Would you change doors, or you would stick to door number 1?

%%
这个问题看上去有些古怪，由于主持人知道门后的奖品，也许他是尝试引诱你做出错误的选择。
你会怎么做？你会改变选择，或者你会坚持选择 1 号门？
%%

Since you are becoming more familiar with Probability and Statistics, you can think even further. What would give you the highest probability to win? Switching doors or keeping your choice? Does it matter?

%%
由于你比较熟悉概率和统计，你可以进一步思考。
哪个选择会有更高的概率赢？改变选择或者坚持选择？或者那根本无所谓？
%%

Well, you have Python in your hands, so, in this notebook you will simulate this game and answer this question by yourself! At first, you can try the game in real time just below, and you may get some idea of what might goes on!

%%
在你手里还有 Python，所以，在这个 notebook 中你将模拟这个游戏，然后自己回答这个问题！
首先，你可以在真实世界试玩这个游戏，这样你或许能对可能发生的事情有一点概念！
%%
## 2. Try the game for yourself!

By running the next cell you can play the game for a while and try out different strategies. In the left panel you will get the actual game, these are the instructions to play:

%%
通过运行下一单元格，您可以玩一会这个游戏，并尝试不同的策略。
左侧面板将呈现实际游戏内容，以下是游玩指南：
%%


- To start a new game simply select one of the three available doors.
- After you select an initial door the host will open one of the two remaining ones and it will always have a goat behind it.
- Then you can decide if you would like to stay or switch doors.
- If you pick the door opened by the host that game will not count.
- If you click outside any of the three doors then the game will restart and not be counted.
- After the prizes are shown (game has ended) click anywhere on the screen to restart the game.
- If you want to restart the counters, run the cell again

%%
- 在三扇门中选择一扇门即开始游戏。
- 在你选择了第一扇门后，主持人会在剩下的两扇门中打开一扇，门后永远有一只山羊。
- 然后你可以选择第一扇或者另外一扇。
- 如果你选择的是主持人打开的门，游戏将不会计分。
- 如果你点击了三扇门以外的任意位置，游戏将会重启且不会计分。
- 在展示了奖品之后（游戏结束）点击任意位置重启游戏。
- 如果你想重置积分，重新运行这个单元格。
%%

The right panel keeps track of the number of games played and the success rate for both strategies. Try it for a while and see if you can find any patterns!

%%
右侧的面版将会跟踪游戏次数和两种策略的成功率，多尝试几次看看是否能找到规律！
%%

```python
game = utils.monty_hall_game()
```
> [!result]
![蒙提霍尔问题-游戏结果.png](https://obsidian-image.wwtt.xyz/2025/12/蒙提霍尔问题-游戏结果.png)


Before going forward make sure that you played the game enough times to formulate an hypothesis. Is is better to switch doors? Is it better to stay on your initial guess? Or it simply does not make a difference?

%%
在继续之前请确认你游玩了足够可以提出假设的时间。
换另外一扇门会更好吗？或者坚持你最初的猜想？或者这根本没有区别？
%%
## 3. Simulate the game for many iterations

After playing for a while you might have come up with some hypothesis about the preferred strategy to beat this game. Now you will simulate the game for many iterations and see if the success rate varies from one strategy to the other.

%%
玩了一段时间后，你可能已经对赢得这款游戏的优选策略有了一些假设。
现在你将多次模拟游戏运行，看看两种不同策略的胜率
%%

In order to do this, the `monty_hall` function is provided. This function takes a single argument which is a boolean that controls if you decide to switch doors or not. Take a look at the code comments if you want to understand how the implementation works. Notice that the value of `0` is used to represent a goat, while `1` represents a car:

%%
为此下面提供了 `monty_hall` 函数。
这个函数获取一个简单的布尔型参数，用于控制你换门还是不换门的选择。
如果你想知道它如何工作的可以看看代码的注释。
请注意，数值 0 代表山羊，而 1 代表汽车：
%%

```python
def monty_hall(switch):

    # All doors have a goat initially
    # 初始化所有的门都是山羊
    doors = np.array([0, 0, 0])

    # Randomnly decide which door will have a car
    # 随机决定一个门的index为car
    winner_index = np.random.randint(0, 3)

    # Place the car in the winner door
    # 设置这个为car的门
    doors[winner_index] = 1

    # Participant selects a door at random
    # 参与者随机选择一个门
    choice = np.random.randint(0, 3)

    # Get doors that can be opened (host cannot open the door chosen or the one with the car)
    # 获取当前状态下主持人可以打开的门。
    openable_doors = [i for i in range(3) if i not in (winner_index, choice)]

    # Host opens one of the available doors at random
    # 在上面可以打开的门中随机打开一个
    door_to_open = np.random.choice(openable_doors)

    # Switch to the other available door (the one that is not the original choice or the opened one)
    # 切换到另一扇可用的门（既非最初选择也非已开启的那扇）。
    if switch:
        choice = [i for i in range(3) if i not in (choice, door_to_open)][0]

    # Return 1 if you open a door with a car, 0 otherwise
    # 返回选择的门的对应值，1是汽车，0是其他。
    return doors[choice]
```

You can use the function above to simulate one run of the game. However this would not be very practical, it is better to use the function to try a bunch of different runs at once and save the results. This way you can know for sure if one strategy beats the other after consistently using it.

%%
您可以使用上面的函数来模拟游戏的一次运行。
然而这并没有什么意义。
最好是模拟运行很多次并将结果保存下来。
用这种方法你可以确认，一种策略是否优于另外一种。
%%

You can pass the above function to another function that lets you decide a strategy and perform simulations for 1, 10, 100 and 1000 runs. As you increase the number of runs you will see that the strategies converge to their true success rate: 

%%
你可以将上面的函数传入另外一个函数，这个函数可以让你决定使用哪一种策略，并一次性模拟 1 次，10 次，100 次，1000 次。
当你增加运行次数，你将看到这个策略结果的收敛，从而看到真正的胜率。
%%

```python
utils.success_rate_plot(monty_hall)
```

> [!result]
> ### Switch is True (1000 simulations):
> ![蒙提霍尔问题-switch-is-true.png](https://obsidian-image.wwtt.xyz/2025/12/蒙提霍尔问题-switch-is-true.png)
> 
> ### Switch is False (1000 simulations):
> ![蒙提霍尔问题-switch-is-false.png](https://obsidian-image.wwtt.xyz/2025/12/蒙提霍尔问题-switch-is-false.png)
## 4. Analytical Solution

Now you are familiar with the problem and you have gotten a strong evidence that the best strategy is to **switch doors** because it will make you win about 67% of the times! 

%%
现在你已经熟悉这个问题了，你得到了一个关于最佳策略的有力证据，那就是**换门**，它让你的胜率达到了 67%
%%

You now will see it analytically! For this, first let's make some definitions. 

%%
你现在将看到它的分析！首先让我们做一些定义。
%%

Define the events:

$E_1$ = the car is behind door 1
$E_2$ = the car is behind door 2
$E_3$ = the car is behind door 3

%%
定义事件：

$E_1$ = 汽车在一号门后面
$E_2$ = 汽车在二号门后面
$E_3$ = 汽车在三号门后面
%%


Or, in a more concise way: $E_i$ = the car is behind door $i$ for $i = 1,2,3$. 

%%
或者用更简洁的方式： $E_i$ = 汽车在 $i$ 号门后面，其中 $i = 1,2,3$。
%%

Note that these events are **mutually exclusive**, in other words, you cannot have a car simuntaneously in two doors, because of the rules of the game. This means that,

%%
注意这些事件都是**互斥**的，换句话说，你不能同时有汽车在两个门后面，由于游戏规则使然，这意味着：
%%

$$
\begin{align}
P(E_1 \cap E_2) = 0 \\
P(E_1 \cap E_3) = 0 \\
P(E_2 \cap E_3) = 0
\end{align}
$$ You can say it also by writing that 

%%
那么你同样可以写作：
%%
$$P(E_i \cap E_j) = 0  \text{      for } i \neq j.$$

Another fact, due to the rules of the game, is that **the car is behind one of the three doors**, so 

%%
另外，正如游戏规则所描述的那样，这辆汽车在**三个门其中一个门**的后面，所以：
%%
$$P(E_1 \cup E_2 \cup E_3) = 1.$$

This is, in fact, the **sample space**, or **universe**, $\Omega$, because it is the set of all possible outcomes.

%%
这实际上就是这个问题的**样本空间**，或称为**全集** $\Omega$，因为它是所有可能结果的集合。
%%

Let's suppose you've chosen **door number 1**. Since there is an equal chance of the car being behind one of the three doors, we know that 

%%
让我们假设你选择了一号门，由于这里汽车在三个门后面的机会是相等的，所以：
%%

$$P(E_1) = \frac{1}{3}.$$

By the **complement rule**, we know that

%%
根据补集规则，得到：
%%

$$P(E_1^c) = 1 - P(E_1) = 1 - \frac{1}{3} = \frac{2}{3}$$
Since the universe is given by $E_1 \cup E_2 \cup E_3$ (the car is behind door 1 OR door 2 OR door 3), then $E_1^c = E_2 \cup E_3$, therefore $P(E_2 \cup E_3) = \frac{2}{3}$. You can have a visual idea in the image below.

%%
由于样本空间为 $E_1 \cup E_2 \cup E_3$，那么 $E_1^c = E_2 \cup E_3$，因此 $P(E_2 \cup E_3) = \frac{2}{3}$。你可以从下面的图中参考
%%

![monty_closed_doors.svg|500](https://obsidian-image.wwtt.xyz/2025/12/monty_closed_doors.svg)


Now that you chose door 1, the Host then opens door 3, revealing a goat and asks you if you want to switch doors. If you don't switch, the probability of winning remains $\frac{1}{3}$ because this is your initial choice. If you **do** switch, then, you can notice that the Host **gave you an additional information**. They showed to you that door 3 does not have a car, which means that 

%%
现在你选择了一号门，主持人打开了三号门，揭示了山羊并询问你是否换门。
如果你不换，赢得游戏的概率仍然保持为 $\frac{1}{3}$，因为这是你最初的选择。
如果你选择换门，你可以注意到，主持人给你**提供了额外的信息**。
他向你展示了三号没有汽车，这意味着：
%%

$$P(E_3) = 0.$$

Now you are mostly done, because as you know, $\frac{2}{3} = P(E_2 \cup E_3) = P(E_2) + P(E_3) - P(E_2 \cap E_3)$. You already know that $P(E_2 \cap E_3) = 0$, because they are mutually exclusive events (the car is behind in only **one** of the three doors), and the Host gave you a very importante piece of additional information: $P(E_3) = 0$. With this, you can easily conclude that:

%%
你快要成功了，因为正如你所知道的，$\frac{2}{3} = P(E_2 \cup E_3) = P(E_2) + P(E_3) - P(E_2 \cap E_3)$。
同时，你已经知道 $P(E_2 \cap E_3) = 0$，因为它们是互斥事件。（汽车只会在三个门中的一个门后面），然后主持人给你了一个非常重要的额外信息：$P(E_3) = 0$。
由此，你可以轻松得出结论：
%%
$$P(win | switch) = P(E_2) = \frac{2}{3}.$$

In other words, the probability that the car is behind door 2, **given that** it is not behind door 3 is $\frac{2}{3} \approx 0.67$ as you have just seen in your simulations!

%%
换句话说，在已知汽车不在 3 号门后的条件下，它在 2 号门后的概率约为 $\frac{2}{3} \approx 0.67$，正如你在模拟中所见！
%%
## 5 Generalized Monty Hall problem (optional)

Let's consider a new game, more general.

%%
让我们考虑一个新游戏，更加广义的。
%%

Now, the game is:
- There are $n$ doors, and you must choose one door.
- Host opens $k$ doors and revealing goats.
- You may or may not change your previously chosen door.

%%
这个游戏是：
- 这里有 `n` 扇门，你必须选择一扇。
- 主持人打开了 `k` 扇门，揭示了山羊。
- 你可以坚持或者更换你之前选择的门。
%%

Would it still be better to switch doors? Would it depend on $k$ or on $n$? 

%%
更好的选择会是换门吗？这依靠 $k$ 或者 $n$ 来判断吗？
%%

## 5.1 Simulation

You can simulate the problem to build your intuition. 

%%
你可以用这个模拟程序来构建你的直觉。
%%

```python
def generalized_monty_hall(switch, n = 3, k = 1):
    if not (0 <= k <= n-2):
        raise ValueError('k must be between 0 and n-2, so the Host can leave at least 1 openable door!')
    
    # All doors have a goat initially
    doors = np.array([0 for _ in range(n)])
    
    # Decide which door will have a car
    winner = np.random.randint(0,n)

    # Place the car in the winner door
    doors[winner] = 1.0
    
    # Participant selects a door at random
    choice = np.random.randint(0,n)
    
    # Get doors that can be opened (host cannot open the door chosen or the one with the car)
    openable_doors = [i for i in range(n) if i not in (winner, choice)]
    
    # Host open k of the available doors at random
    door_to_open = np.random.choice(openable_doors, size = k, replace = False)
    

        # Switch to the other available door (the one that is not the original choice or the opened one)
    if switch:
        choices = [i for i in range(n) if i not in np.array(choice) and i not in np.array(door_to_open)]
        # Player chooses another door at random
        choice = np.random.choice(choices)
    
    # Return 1 if you open a door with a car, 0 otherwise
    return doors[choice]
```


```python
utils.success_rate_plot(generalized_monty_hall)
```

## 5.2 Analytical solution

This section is more advanced, you may skip it if you want to! 

%%
这一部分内容较为深入，若您愿意，可以选择跳过！
%%

Now, the game is:
- There are $n$ doors, and you must choose one door.
- Host opens $k$ doors and revealing goats.
- You may or may not change your previously chosen door.

The question is: is it always better to switch doors? Will it depend on $k$? 

To answer this question analyticaly, first define the following events:

%%
要分析这个问题，首先定义下面的事件。
%%

$$E_i = \text{ the car is behind door i. In this case, } i = 1, \ldots, n.$$

Again, the $E_i$ 's are independent from each other, because there is only $1$ car available.

%%
再说一次，$E_i$ 相互之间是互斥的，因为只有一辆车。
%%

Note that, since the Host never opens the same door the player chose and also never opens the winning door, there is an upper bound for $k$, which is $n-2$, so 

%%
需要注意的是，由于主持人永远不会打开玩家选择的门和有车的门，那么 k 的上限就是 $n-2$，所以：
%%
$$ 0 \leq k \leq n-2.$$ 

Two facts can be assumed:

- The player chooses door $1$
- The host opens doors $2, \ldots, k+1$

%%
可以假设两个事实：
- 玩家选择了一扇门
- 主持人打开了 $2,\dots, k+1$ 扇门
%%

This is because we can always rename the doors to get this result. For instance, if the player chooses door number $10$, we can rename it as door $1$ and door $1$ will become door $10$. This is just to avoid getting too complex on indices notations. In math terminology, it is usually said that we can do this *without loss of generality*, since it will not affect the final result. 

%%
假设玩家选择了十号门，我们可以重命名这个门为一号门，那么一号门就变成了 10 号门。
由于我们总是可以重新命名这些门以得到这个结果。
这只是为了避免在索引符号上过于复杂化。
在数学术语中，通常可以说我们这样做*不失一般性*，因为它不会影响最终结果。
%%

Now that there are $n$ doors, the probability that the car is behind door $1$ is $\frac{1}{n}$, i.e.,

%%
现在这里有 $n$ 扇门，汽车在一号门后面的概率则是 $\frac{1}{n}$，即：
%%
$$P(E_1) = \frac{1}{n}.$$

By the complement rule, the probability that the car is **not** behind door $1$ is:

%%
根据补集规则，汽车**不在**一号门后面的概率则为：
%%
$$P(E_1^c) = 1 - P(E_1) = 1 - \frac{1}{n} = \frac{n-1}{n}.$$

Note that 

$$E_1^c = E_2 \cup E_3 \cup \ldots \cup E_n.$$

There is a notation to simplify the right hand side equation above, we can write it as:

%%
为了简化上面方程，等号的右侧我们可以写成这样：
%%
$$\bigcup_{i = 2}^{n} E_i.$$

This works in the same fashion as a summation symbol, but the opeartion being performed is set union.

%%
它和求和符号相似，但是它执行的是集合合并操作。
%%

So, we know that 
$$P\left(\bigcup_{i = 2}^{n} E_i\right) = \frac{n-1}{n}.$$

Now we can answer the question: What is the probability of winning, given that we switch doors?

Let's take a look on the following image:

%%
现在我们可以回答这个问题了：换门后的胜率多少？先看看下面这张图：
%%
![monty_hall_n_k.svg|800](https://obsidian-image.wwtt.xyz/2025/12/monty_hall_n_k.svg)

If the player switches to a random available door, then they must choose one of the $k+2, k+3, \ldots, n-1, n$. Therefore, the probability of picking the car is:

%%
如果玩家随机选择了另一扇可用的门，那么他必须选择 $k+2, k+3, \ldots, n-1, n$ 中的一扇，那么选中汽车的概率是：
%%

The probability of **not picking the car** in door $1$ $\left(P(E_1^c) = \frac{n-1}{n}\right)$ times the probability of picking the car **now**, which is $\frac{1}{n-k-1}$ because this is the number of remaining doors. 

%%
车不在一号门的概率 $\left(P(E_1^c) = \frac{n-1}{n}\right)$ 乘以**现在**选中汽车的概率 $\frac{1}{n-k-1}$。
（[[条件概率]]，$P(在剩余门中选中车∣车不在一号门) \times P(车不在一号门)$）。
因为这是剩余门的数量。
%%

So, the final probability is given by

$$P(win | switch) = \frac{n-1}{n} \cdot \frac{1}{n-k-1}.$$

It can be rewriten in the following manner:

%%
它用这样的方式来写：
%%
$$P(win | switch) = \frac{n-1}{n} \cdot \frac{1}{n-k-1} = \frac{1}{n} \cdot \frac{n-1}{n-k-1} \geq \frac{1}{n} = P(E_1) = P(win | not\ switch).$$

And the equality only holds when $k = 0$. This means that the host does not open any door.

%%
这是一个不等式，仅 $k = 0$ 的时候它才是一个等式。这意味着主持人没有打开任何一扇门。
%%

Therefore, **it is always better to switch doors**. This may sound counterintuitive at first, but think that switching doors you are using the **new piece of information** that the host gave you, whereas if you choose not to switch, you will be ignoring this new information.

%%
所以，**最好的方案是换门**。
初次听起来可能有些反直觉，但是可以这么思考，换门的情况相当于主持人给你了**一条新的信息**，如果你没有换门，也即是你忽略了这条新信息。
%%

**Congratulations! You have finished the ungraded lab on the Monty Hall problem!**
