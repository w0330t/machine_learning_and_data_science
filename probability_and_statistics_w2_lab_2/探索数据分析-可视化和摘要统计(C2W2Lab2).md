---
aliases: [Exploratory Data Analysis - Data Visualization and Summary Statistics]
tags: []
created: 2025-12-31, 16:59:40
modified: 2026-01-02, 10:22:53
---

# Exploratory Data Analysis - Data Visualization and Summary Statistics

Welcome to the third notebook of the exploratory data analysis (EDA) series. This notebook is a continuation of the rideshare notebook you used last week.
%%
欢迎来到数据探索分析第三部分。这个 Notebook 是上周网约车 Notebook 的延续。
%%
For this notebook you will use the data on ridesharing in the year 2022 in the city of Chicago, which can be found [here](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p/data). You have already worked with this dataset in the previous week. This time you will continue working on the cleaned-up and reduced version of the dataset, which you prepared in the previous notebook.
%%
这个 Notebook 你将继续使用 2022 年芝加哥网约车的数据。你在上周已经使用了这个数据集。这次你将继续在之前 Notebook 处理后的数据集中工作。
%%
### Learning Objectives:
In this notebook you will use the following concepts from the course in a practical setting:
 - Probability
 - Descriptive statistics (mean, median, standard deviation and quartiles)
 - Box plots
 - Joint distribution
 - Marginal distribution
 - Correlation

%%
这个 Notebook 将运用课程中的以下概念：
- 概率
- 描述性统计（均值，中位数，标准差和分位数）
- 线箱图
- 联合分布
- 边缘分布
- 协方差
%%
# 1. Import the Python Libraries

As usual, the first thing you need to do is import the libraries that you will use in this notebook.
%%
如同往常一样，首先导入库。
%%

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Library for displaying a map
import folium 
from folium.plugins import FastMarkerCluster
```

# 2. Load the Dataset

The next step is to open the dataset. This is the reduced and cleaned-up version that you used in the previous notebook.
%%
下一步是打开数据集，这是一个缩减并清理后的版本，出自上一个 Notebook。
%%

```python
# Open the dataset
# Note the parse_dates parameter, which automatically saves the given columns as dates.
# df = pd.read_csv("data/rideshare_2022_cleaned.csv", parse_dates=['trip_start_timestamp', 'date'])

df = pd.read_parquet("data/rideshare_2022_cleaned.parquet")

df['Trip Start Timestamp'] = pd.to_datetime(df['trip_start_timestamp'])
df['Trip End Timestamp'] = pd.to_datetime(df['date'])

# Show the first few lines of the dataset
df.head()
```

> [!result]
> |     | trip_start_timestamp | trip_seconds | trip_miles | fare | tip | additional_charges | trip_total | shared_trip_authorized | trips_pooled | pickup_centroid_latitude | pickup_centroid_longitude | dropoff_centroid_latitude | dropoff_centroid_longitude | date       | weekday  | Trip Start Timestamp | Trip End Timestamp |
> | --- | -------------------- | ------------ | ---------- | ---- | --- | ------------------ | ---------- | ---------------------- | ------------ | ------------------------ | ------------------------- | ------------------------- | -------------------------- | ---------- | -------- | -------------------- | ------------------ |
> | 0   | 2022-01-01           | 3905.0       | 44.5       | 55.0 | 0.0 | 11.25              | 66.25      | 0                      | 1            | 41.972563                | -87.678846                | NaN                       | NaN                        | 2022-01-01 | Saturday | 2022-01-01           | 2022-01-01         |
> | 1   | 2022-01-01           | 2299.0       | 25.0       | 32.5 | 7.0 | 7.18               | 46.68      | 0                      | 1            | 41.878866                | -87.625192                | NaN                       | NaN                        | 2022-01-01 | Saturday | 2022-01-01           | 2022-01-01         |
> | 2   | 2022-01-01           | 275.0        | 1.5        | 7.5  | 0.0 | 1.02               | 8.52       | 0                      | 1            | 41.792357                | -87.617931                | 41.812949                 | -87.617860                 | 2022-01-01 | Saturday | 2022-01-01           | 2022-01-01         |
> | 3   | 2022-01-01           | 243.0        | 1.0        | 5.0  | 0.0 | 2.36               | 7.36       | 0                      | 1            | 41.936310                | -87.651563                | 41.943155                 | -87.640698                 | 2022-01-01 | Saturday | 2022-01-01           | 2022-01-01         |
> | 4   | 2022-01-01           | 364.0        | 1.3        | 5.0  | 0.0 | 2.36               | 7.36       | 0                      | 1            | 41.921855                | -87.646211                | 41.936237                 | -87.656412                 | 2022-01-01 | Saturday | 2022-01-01           | 2022-01-01         |

# 3. Investigate the Summary Statistics

To get a better grasp of the data it is very useful to learn a bit more about the values in each column. In the previous notebook you have already plotted some histograms of individual columns to see how the data is distributed. Now it's time to approach this more systematically. Let's look at the numeric values first. Pandas has a very useful function `.describe()`, which returns a new dataframe with summary statistics for each of the columns. Run the cell below to compute and display summary statistics for your dataset. The output is a new dataframe that contains the count, mean, standard deviation, minimum value, maximum value and 25%, 50% (median) and 75% quartiles for each of the columns. By now, you should be familiar with all of these statistics. If you need a refresher, check out the Week 2 Lesson 1 videos again. 
%%
为了更好的掌握数据，需要深入了解每个列。在上一个 Notebook 中你已经绘制了一些列的直方图，并观察了它们的分布。这次用更系统性的方法。我们先看看数值。Pandas 有一个非常棒的 `.describe()` 函数，它将返回一个 dataframe，里面包含了每列的汇总统计。运行下面的单元格，它将计算并展示你的数据集的统计。它输出的 dataframe 中包含了每列的计数，平均值，标准差，最小值，最大值和四分位数。现在你应该很熟悉这些统计数据。如果你需要复习，则可以再次观看第二周的第一课的视频。
%%

```python
# Calculate and display the summary statistics
df.describe()
```

> [!result]
> |       | trip_start_timestamp          | trip_seconds  | trip_miles    | fare          | tip           | additional_charges | trip_total    | shared_trip_authorized | trips_pooled  | pickup_centroid_latitude | pickup_centroid_longitude | dropoff_centroid_latitude | dropoff_centroid_longitude | date                          | Trip Start Timestamp          | Trip End Timestamp            |
> | ----- | ----------------------------- | ------------- | ------------- | ------------- | ------------- | ------------------ | ------------- | ---------------------- | ------------- | ------------------------ | ------------------------- | ------------------------- | -------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
> | count | 691098                        | 691073.000000 | 691096.000000 | 689952.000000 | 689952.000000 | 689952.000000      | 689952.000000 | 691098.000000          | 691098.000000 | 635075.000000            | 635075.000000             | 632163.000000             | 632163.000000              | 691098                        | 691098                        | 691098                        |
> | mean  | 2022-07-09 12:13:17.168419328 | 1089.008338   | 6.941224      | 18.577024     | 1.264072      | 4.694999           | 24.536095     | 0.022537               | 1.010250      | 41.889642                | -87.671920                | 41.890190                 | -87.674246                 | 2022-07-09 02:16:40.835192832 | 2022-07-09 12:13:17.168419328 | 2022-07-09 02:16:40.835192832 |
> | min   | 2022-01-01 00:00:00           | 1.000000      | 0.000000      | 0.000000      | 0.000000      | 0.000000           | 0.000000      | 0.000000               | 1.000000      | 41.650222                | -87.913625                | 41.650222                 | -87.913625                 | 2022-01-01 00:00:00           | 2022-01-01 00:00:00           | 2022-01-01 00:00:00           |
> | 25%   | 2022-04-11 11:45:00           | 543.000000    | 2.000000      | 10.000000     | 0.000000      | 2.490000           | 13.440000     | 0.000000               | 1.000000      | 41.871016                | -87.689319                | 41.871016                 | -87.691430                 | 2022-04-11 00:00:00           | 2022-04-11 11:45:00           | 2022-04-11 00:00:00           |
> | 50%   | 2022-07-12 09:45:00           | 880.000000    | 4.100000      | 15.000000     | 0.000000      | 3.550000           | 19.020000     | 0.000000               | 1.000000      | 41.893216                | -87.654093                | 41.893216                 | -87.654007                 | 2022-07-12 00:00:00           | 2022-07-12 09:45:00           | 2022-07-12 00:00:00           |
> | 75%   | 2022-10-08 19:30:00           | 1416.000000   | 9.200000      | 22.500000     | 1.000000      | 5.460000           | 29.490000     | 0.000000               | 1.000000      | 41.934762                | -87.631407                | 41.935706                 | -87.631407                 | 2022-10-08 00:00:00           | 2022-10-08 19:30:00           | 2022-10-08 00:00:00           |
> | max   | 2022-12-31 12:45:00           | 34892.000000  | 366.900000    | 637.500000    | 100.000000    | 253.010000         | 656.750000    | 1.000000               | 5.000000      | 42.021224                | -87.530712                | 42.021224                 | -87.530712                 | 2022-12-31 00:00:00           | 2022-12-31 12:45:00           | 2022-12-31 00:00:00           |
> | std   | NaN                           | 782.835520    | 7.773458      | 14.069854     | 2.923235      | 4.314872           | 17.627719     | 0.148421               | 0.113987      | 0.067517                 | 0.070488                  | 0.067239                  | 0.075001                   | NaN                           | NaN                           | NaN                           |

In the dataframe above, you can find a lot of useful information. Carefully inspect the contents of the table to understand the data better. The following questions may help you think about the insights you can get from the summary statistics presented in the table.
1. Check the minimum and maximum value for each column. What are their values and how far apart are they? For example: What is the shortest and longest trip that was taken and what is the difference between them?
2. What is mean value of each column? For example: What is the mean trip length? Is it closer to the minimum or maximum value?
3. What is the standard deviation of each column? For example: how much do the trip lengths vary?
4. Compare the quartiles and the mean. Is the mean above or below the median? How could you explain this (think of the shape of the histograms you saw in the previous notebook and look for long or heavy tails).
%%
在上面的 dataframe 中，你可以找到很多有用的信息。仔细检查表格内容以更好地理解数据。下面的问题可以帮助你更好的观察表格中的统计数据。
5. 检查每列的最小值和最大值。它们的数值和它们的差距是多少？比如：最短和最长的行程分别是多少，它们之间有何差异？
6. 每列的均值是多少？比如 ：行程的均值是多少？它是更接近最小值还是最大值？
7. 每列的标准差是多少？比如行程长短有多大的差异？
8. 比较分位数和均值。均值比中位数大还是小？你可以解释它吗？（想象一下直方图的形状，你可以参考之前的 Notebook，寻找长尾或重尾分布）
%%
Note that the first three questions all relate to the same column `trip_miles`. You can ask the same question about any column, for example: What is the mean/median/highest/lowest tip?
%%
请注意，前三个问题都与同一列 `trip_miles` 相关。你当然可以用任意列提出相同的问题。比如：小费的均值/中位数/最小值/最大值。
%%

## 3.1 Visualize the Summary Statistics Using Boxplots

A great way to understand your data is to visualize it! A commonly used tool to display summary statistics is a boxplot. Fortunately, it is already integrated to `Pandas`, and you can simply draw it by using `DataFrame.boxplot()`. Remember that the box-plot involves infromation about all the quartiles and the maximum and minumum values of the variable, and it looks something like this
%%
可视化是了解你的数据的好办法！一个展示统计数据的常用工具是线箱图。幸运的是，它已经集成在了 `Pandas` 中，你可以简单的使用 `DataFrame.boxplot()` 绘制它。记住，线箱图包含了所有的分位数，最小值，最大值的变量，它看上去像这样：
%%

![box-plot-screenshot.png|370](https://obsidian-image.wwtt.xyz/2026/01/box-plot-screenshot.png)

As with other plots, if you do not specify which variable, or column, you want to plot, it will plot all of them. Because the columns have very different values, it is better to plot one by one, so you can more easily see the information being communicated.
%%
与其他图表一样，如果你没有指定变量或者列，它将绘制所有的列。因为这些列的值均不相同，最好是一个个绘制，这样你就能更轻松地看到所传达的信息。
%%
```python
# Select the column which you want to plot. Change this for a different column name,
# if you are interested in plotting other columns
column_to_plot = 'fare'

# Display the boxplot
plt.figure()
df.boxplot(column_to_plot)
```

> [!result]
>     <Axes:>
   ![Rideshare_Project_Week2_9_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_9_1.png)

The blue box shows the interquartile range (IQR), and the horizontal blue lines in the box show the Q1, Q2 (median), and Q3 quartiles. You can see these three numbers in the dataframe above (25%, 50% and 75%). Check whether they align with the plot. The horizontal black lines outside the box show +/- 1.5 times IQR, which is the default range used to identify outliers. The individual datapoints plotted outside the lines are the outliers.
%%
这个蓝色的箱子就是四分位距（IQR），箱子中的蓝色水平线则是 Q1、Q2（中位数）和 Q3 分位数。你可以看到上面 dataframe 中（25%, 50% 和 75%）的这三个数值。确认它们在图中的位置是否一致。而在箱子外面的黑色水平线则是正负 1.5 倍的 IQR，这是用于识别异常值的默认范围。绘制在线条之外的单个数据点即为离群值。
%%
In the case of `fare` you can see a lot of outlier points, can you figure out why? Remember the distribution of this variable
%%
在这个 `fare` 示例中，你能看到很多异常点，你能找出原因吗？请记住这个变量的分布情况。
%%
```python
df.hist(column_to_plot, density=True)
```

> [!result]
>     array([[<Axes: title={'center': 'fare'}>]], dtype=object)
![Rideshare_Project_Week2_11_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_11_1.png)


As you can see, this distribution is really heavy for smaller values, and has a really long tail, which is the values you are seeing as outliers.
%%
正如你所看到的，该分布在较小值上显得非常密集，同时它有非常长的尾巴，这就是你看到的异常值。
%%
Go ahead and change the `column_to_plot` variable and plot the boxplots for other columns. Do they all have outliers only on one side? can you infer why?
%%
回到上面的代码中，将其他的列赋值到 `column_to_plot` ，然后绘制线箱图。它们是否都只在单侧存在异常值？你能推断出原因吗？
%%
## 3.2 Visualize the Data on different weekdays

If you want to split the data into subsets (for example for given days of the week) and plot a boxplot for each, you can easily do that using the parameter `by`. You just need to set it to the column name you want to use to split the data. Suppose you want to analize the `tip` variable by day of the week. Intuitively, what you are doing is creting classes according to the different days in `weekday`, and then analyzing the data for each of this classes. This way what you are actually doing is exploring the conditional distributions of a variable. In this case, you are looking at the tip given that it's a Monday, the tip given that it's a Tuesday, and so on.
%%
如果你想将数据拆分为子数据集（例如，对于给定的星期几），然后将其绘制为线箱图，可以很简单的时候参数 `by`。只需将其设置为您想要用于分割数据的列名即可。假设你想要根据每周的星期几分析 `tip` 变量。更直观的说，直观上，你所做的是根据每周里的星期几（`weekday`）来创建类别，然后分析每个类别的数据。实际上这种方式正是探索一个变量的条件分布。在这个案例中，你看到的是周一的小费，周二的小费，以此类推。
%%

```python
df.boxplot(column='tip', by='weekday')

# Limit the plot in y direction. Comment this line of code to see full data
plt.ylim(-2, 52)
```

> [!result]
>     (-2.0, 52.0)
![Rideshare_Project_Week2_13_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_13_1.png)

If you look closely, there are a couple of things that look a bit off in these boxplots. For example, what is happening with the Sunday data when you are looking at tips? It seems that there is no box in the boxplot. Also if you look more closely, the boxes that you see dont have a horizontal line in the middle. Why do you think this is happening? You can find a hint by thinking about the histogram of the tips that you saw in the previous notebook. Did the distribution have any interesting properties? If you don't remember what it looked like, feel free to create a new cell and draw the histogram. 
%%
如果你观察得足够仔细，会发现这些线箱图中有几处不太对。比如，周日发生了什么导致小费的分布有些奇怪？这个线箱图看上去没有箱子。同样的，如果你更仔细观察，这些盒子中间似乎没有水平的中位数。为什么会这样？你可以通过观察上一个 Notebook 的小费直方图获得提示。这个分布还有什么有趣的性质吗？如果你不记得它是什么样，请创建一个新的单元格并绘制直方图。
%%
Another, more obvious, hint is in the summary statistics. Run the cell below to display them again, but this time you will do it just for the `tip` column, grouped by `weekday`.
%%
另一个更为明显的提示隐藏在汇总统计之中。运行下面的单元格再次确认，但是这次你将只针对 tip 列，并使用 `weekday` 分组。
%%

```python
# Group the data by the day of the week, select the tip column and calculate the descriptive statistics.
df.groupby('weekday')['tip'].describe()
```

> [!result]
> |           | count    | mean     | std      | min | 25% | 50% | 75% | max   |
> | --------- | -------- | -------- | -------- | --- | --- | --- | --- | ----- |
> | weekday   |          |          |          |     |     |     |     |       |
> | Friday    | 115730.0 | 1.284628 | 2.999008 | 0.0 | 0.0 | 0.0 | 1.0 | 70.0  |
> | Monday    | 78922.0  | 1.309952 | 3.009671 | 0.0 | 0.0 | 0.0 | 1.0 | 57.0  |
> | Saturday  | 132688.0 | 1.149222 | 2.656089 | 0.0 | 0.0 | 0.0 | 1.0 | 83.0  |
> | Sunday    | 96823.0  | 1.216395 | 2.880735 | 0.0 | 0.0 | 0.0 | 0.0 | 75.0  |
> | Thursday  | 95561.0  | 1.355553 | 3.124336 | 0.0 | 0.0 | 0.0 | 1.0 | 100.0 |
> | Tuesday   | 82479.0  | 1.291480 | 2.945485 | 0.0 | 0.0 | 0.0 | 1.0 | 61.0  |
> | Wednesday | 87749.0  | 1.296585 | 2.921619 | 0.0 | 0.0 | 0.0 | 1.0 | 70.0  |

You can see that there are a lot of zeroes in this dataframe. Where do they all come from? It turns out that on Sunday more than 75% of the people did not tip. This makes the first three quartiles all show a zero, which is why you couldn't see a box on the boxplot. Remember that the size of the box is the distance between the values in the columns `25%` and `75%`. If these are both zero, the size of the box must also be zero.
%%
你可以在这个 dataframe 中看到许多零。它们是怎么来的？原来周日有超过 75%的人都不付小费。这造成了前面三个分位数全都是零。这就是为什么你在线箱图里看不到箱子。这个箱子的高度是由这个列中的 `25%` 和 `75%` 的差值决定的。如果它们都是零，这个箱子的高度肯定也是零。
%%
On the other days, however, less than 75% of the people did not tip, thus you have a nonzero value of the tip in the third quartile (in the column with name 75%). What does it mean for the plots? You have guessed it: the boxes appear. However, you still can't see the middle line and that is because the median (50% of the data) is the same as the first quaritle (25%), and thus the two lines overlap at zero.
%%
在其他日子里，接近 75% 的人也不会付小费。第三分位数不为零。对于线箱图来说这意味着什么？你可以这么猜测：箱子会出现，但是仍然看不到中位数的线，因为中位数（50%）和第一分位数（25%）相等，并且这两根线的值都为零。
%%
An interesting insight: on Sunday, all of the tips are outliers.
%%
还有一个有趣的现象是：在周日，所有的小费都是异常值。
%%
Now, let's repeat the process but leave out non-tippers. This should give you a better understanding of the distribution of the actual tips.
%%
现在将没有付小费的行程去掉，再运行一次。这会给出一个更好的真实小费的分布情况。
%%

```python
df_tippers = df[df['tip'] > 0]
df_tippers.boxplot(column='tip', by='weekday')
```

> [!result]
>     <Axes: title={'center': 'tip'}, xlabel='weekday'>
![Rideshare_Project_Week2_17_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_17_1.png)

This plot gives you a better insight into the distribution of the tips that actually happened, however it misses an important piece of information: how many people do actually tip? You have calculated this already in the previous notebook.
%%
此图有助于更深入地了解实际发生小费的分布情况。但是它忽略了一个很关键的部分：有多少人付了小费？你已经在之前的 Notebook 中计算过了
%%
Imagine you are a driver. Is there any day of the week that you would like to drive more, as the tips are higher? 
Spliting by day of the week doesn't seem to have much of an impact to make this decision. Maybe you can have a look at the tips given different hours of day. Perhaps there is a higher chance of getitng tipped at a certain hour. Lets see if that's the case by running the cell below. For this, you will need to extract the hour of the trip for the `trip_start_timestamp` column, and save it on a new column in the dataframe.
%%
想象一下你是一个司机。在一周中的哪天你更想工作？小费最高的那天吗？将一周切割为七天似乎对做出这个决定影响不大。或者你可以看看每天中哪个时间段的小费差距。也许某一时刻的小费获取机会更大。让我们在下面的单元格中看看。你需要在 `trip_start_timestamp` 列提取行程的时间。然后将它们保存在一个新的 dataframe。
%%

```python
# Add a column for the hour of day to the dataframe
df["hour"] = df["trip_start_timestamp"].apply(lambda x: x.hour)
# Select only the tippers
df_tippers = df[df['tip'] > 0]

# Plot the boxplot of tips for each hour
plt.figure()
df_tippers.boxplot(column='tip', by='hour')
plt.ylim(-2, 52)

# Calculate the percentage of tippers
percentage_of_tippers_hourly = df_tippers.groupby(["hour"])["tip"].count() / df.groupby(["hour"])["tip"].count() * 100

# Plot the percentage of tippers
plt.figure()
percentage_of_tippers_hourly.plot(marker="o", title="Percentage of Tippers")
```

> [!result]
>     <Axes: title={'center': 'Percentage of Tippers'}, xlabel='hour'>
>     <Figure size 640x480 with 0 Axes>
![Rideshare_Project_Week2_19_2.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_19_2.png)
![Rideshare_Project_Week2_19_3.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_19_3.png)


It seems like the tips are slightly higher in the morning hours. But why would the tips be higher in the early morning? Maybe this has to do with the hour of day, perhaps people feel more empathy in the morning. Before you jump to this conclusion, however, let's see if there's any other explanations worth considering.
%%
看上去清晨的那段时间小费偏高。但是为什么会发生这种情况？或许这与一天中的时段有关，也许人们在凌晨更能感同身受。在你妄下结论之前，让我们看看是否有其他值得考虑的解释。
%%
Let's see if you can get any any extra information by looking at the the lenght of the trip. Plot the length of the trip next to see how it changes through the day:
%%
看看是否可以通过行程的距离来得到其他额外的信息。接下来绘制行程长度，观察其如何随一天时间变化：
%%

```python
df.boxplot(column='trip_miles', by='hour')
plt.ylim(-10, 210)
```

> [!result]
>     (-10.0, 210.0)
![Rideshare_Project_Week2_21_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_21_1.png)

As you can see the trips are longer in the early morning hours. Now this is getting interesting. Maybe this is the reason for the higher average tips? Run the next cell to plot a scatter plot of `trip_miles` vs `tip`. Remember that you are only looking at the tippers (no tippers would just contribute to many additional points at tip=0).
%%
正如你看到的，清晨的行程距离会更长。现在事情更加有趣了，也许这就是平均小费比较高的原因？运行下一个单元格绘制 `trip_miles` 和 `tip` 的散点图。请记住，您仅关注那些给予小费的顾客（未给小费者仅会在小费为零时贡献大量额外积分）。
%%

```python
df_tippers.plot(kind='scatter', x='trip_miles', y='tip', marker=".")
```

> [!result]
> 	<Axes: xlabel='trip_miles', ylabel='tip'>
![Rideshare_Project_Week2_23_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_23_1.png)

Can you say something about the correlation between these two variables?  Remember that the correation was a scaled version of the Covariance, that was scaled to have range between -1 and 1
%%
你可以解释一下这两个变量之间的相关性吗？请记住，相关性是协方差的缩放版本，其缩放范围在-1 至 1 之间。
%%

![corr_coef_screenshot.png|600](https://obsidian-image.wwtt.xyz/2026/01/corr_coef_screenshot.png)


The plot suggests that the riders pay higher tips for longer rides, which would imply a positive correlation. However the correlation is not that obvious, so the correlation will be positive, but not close 1. This correlation makes sense, as longer rides also cost more and thus the tips are likely to be higher. 
%%
这个散点图表面，乘客会在更长的行程中付更高的小费，这意味着它们存在正相关的关系。这个相关性似乎并不明显，所以虽然呈正相关，但距离 1 比较远。这种关联性合乎情理，因为行程越长费用越高，相应地小费也倾向于更多。
%%
You can actually measure the correlation between the two variables using Pandas. Let's do it in the cell below!
%%
你可以在 Pandas 中计算一下这两个变量的相关性。就在下面的单元格中！
%%
```python
(df_tippers.tip).corr(df_tippers.trip_miles)
```

> [!result]
>     0.6368983311939714


You got a correlation of 0.637. Remember that 0 correlation means no correlation at all, while correlation 1 means a perfect linear relationship, with positive slope. In this case, as predicted you get a value somewhere in between, being slightly above 0.5. You could say this is a modeate correlation between variables. 
%%
你得到的相关系数为 0.637。请记住，相关系数为 0 意味着没有关系。相关系数为 1 则是完美的线性相关，且斜率为正。在这个示例中，和预期一样，你得到的值在 0 和 1 之间，略高于 0.5。你可以说这两个变量有一定的相关性。
%%
Now try changing the variables in the plot above to see other columns. For example, you can plot `tip` against `fare`. How does it compare with the `tip` against `trip_miles` plot? Try also checking the correlation and making a comparison. 
%%
现在尝试用其他列的数据绘图。比如你可以使用 `tip` 和 `fare`，让它与 `tip` 和 `trip_miles` 的结果对比。同时尝试检查相关性并进行比较。
%%
# 4. Check the Locations of Rides

Another thing you might want to know is where the rides usually start. The dataset contains the columns "Pickup Centroid Latitude", "Pickup Centroid Longitude", "Dropoff Centroid Latitude", and "Dropoff Centroid Longitude", which tell you the locations of the pickup and dropoff respectively.
%%
另一件事情是你可能想要知道乘客的出发地点。数据中包含了"Pickup Centroid Latitude", "Pickup Centroid Longitude", "Dropoff Centroid Latitude", 和 "Dropoff Centroid Longitude"，它们可以告诉你上下车的位置。
%%
Run the cell below to plot the geographical distribution of the pickup locations.

运行下面的单元格绘制上车的地理分布

```python
# Select the columns you want to plot
latitude = df.dropna()["pickup_centroid_latitude"].to_numpy()
longitude = df.dropna()["pickup_centroid_longitude"].to_numpy()
# Plot the 2D histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
hist = ax.hist2d(longitude, latitude, bins=50, density=True)
ax.set_aspect(1.3, "box")
fig.colorbar(hist[3])
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
```

> [!result]
>     Text(0, 0.5, 'Latitude (degrees)')
![Rideshare_Project_Week2_28_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_28_1.png)

What you have just plotted is a joint distribution! Joint distributions are distributions where a variable (in your case `tip`) is distributed across multiple variables (in your case latitude and longitude).
%%
你刚刚画的是一个联合分布！联合分布是一个变量（在您的案例中是 `tip` ）分布在多个变量（在您的案例中是纬度和经度）上的分布。
%%
Looking at the distribution, it seems you have many rides in the middle right of the plot, which is likely to be downtown Chicago. But then there are also quite a few rides in the top left corner, very far away from everything else. What could there be at that location? 
%%
看看这个分布，有很多乘客在图上中心偏右的位置，看起来像是在芝加哥的下城区。但是左上角似乎也有相当多的乘客，离其他地方都很远，那里有什么吗？
%%
To answer this question, you can actually produce an interactive map. Run the following code to do so, plotting the same points on an actual map of Chicago. In reality this cell is only going to plot a limited number of your data points, even from this downsampled data set, to ensure the map renders quickly enough. Check the locations on the map to see where the majority of the points are and what the location in the upper left could be.
%%
为了回答这个问题，你可以制作一个交互式地图，运行下面的代码，绘制相同的点在这个芝加哥地图中。实际上，这个单元格仅仅会绘制有限的数据点，不过即使是这个下采样的数据集，用于快速的渲染地图还是没问题的。在地图中检查这个地点，确认密集的点和左上角是在什么地方。
%%
Note generating this map is a more resource intensive operation and can sometimes fail. If the map doesn't render after a short wait, you can try re-running the cell.
%%
注意，创建这个地图的需要更加密集的操作，这可能导致运行失败。如果地图没有渲染出来请稍作等待，你同样可以尝试再次运行这个单元格。
%%
```python
# Define the function for plotting an interactive map
def interactive_map(df, n_samples=4000):
    
    points = df[["pickup_centroid_longitude", "pickup_centroid_latitude"]].dropna()[0:n_samples]
    
    latitude = points.iloc[0]["pickup_centroid_latitude"]
    longitude = points.iloc[0]["pickup_centroid_longitude"]
    
    map3 = folium.Map(location=[latitude, longitude], zoom_start=9)

    marker_cluster = FastMarkerCluster([]).add_to(map3)
    
    for index, row in points.iterrows():
        latitude = row["pickup_centroid_latitude"]
        longitude = row["pickup_centroid_longitude"]
        folium.Marker((latitude, longitude), icon=folium.Icon(color="green")).add_to(marker_cluster)

    return map3

# Run the function
# If the map doesn’t render, first try re-running this cell. If that doesn’t work, 
# you can restart the kernel (from the Kernel menu above) and try running the notebook again
interactive_map(df)
```

> [!result]
![芝加哥网约车乘客分布地图.png](https://obsidian-image.wwtt.xyz/2026/01/芝加哥网约车乘客分布地图.png)

If you inspect the map carefully, you probably noticed that the rides from the top left corner come from the Chicago O'Hare International Airport. Run the code below to isolate these rides by their latitude and longitude and inspect them.
%%
如果你检查得足够仔细，你可能会注意到左上角的乘车路线源自芝加哥奥黑尔国际机场。运行以下代码，通过经纬度筛选出这些行程。
%%

```python
# Select all of the rides starting at the airport
airport_rides = df[
    (df["pickup_centroid_longitude"] < -87.9) &
    (df["pickup_centroid_latitude"] > 41.97) &
    (df["pickup_centroid_latitude"] < 41.99)
]

airport_df_tippers = airport_rides[airport_rides['tip'] > 0]

# Plot the boxplot of tips for each hour
plt.figure()
airport_df_tippers.boxplot(column='tip', by='hour')

# Calculate the percentage of tippers
airport_tippers_hourly = airport_df_tippers.groupby(["hour"])["tip"].count() / airport_rides.groupby(["hour"])["tip"].count() * 100

# Plot the percentage of tippers
plt.figure()
airport_tippers_hourly.plot(marker="o", title="Percentage of Tippers on Rides From the Airport")
```

> [!result]
>     <Axes: title={'center': 'Percentage of Tippers on Rides From the Airport'}, xlabel='hour'>
>     <Figure size 640x480 with 0 Axes>
![Rideshare_Project_Week2_32_2.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_32_2.png)
![Rideshare_Project_Week2_32_3.png|500](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week2_32_3.png)

As you can see the percentage of the people who tip is much higher for the rides that start at the airport. Looks like this is a good place to be as a driver!
%%
正如你所看到的，从机场出发的行程里，给小费的人群比例明显更高。看来这是一个对于司机来说非常好的地方。
%%
**Congratulations on finishing this lab.** You have seen the implementation of quite a few concepts covered in this course: probabilities, descriptive statistics, such as mean, median, standard deviation and quartiles, you plotted box plots and a 2D histogram to represent a joint distribution and you looked into marginal distributions. On top of that you have practiced Pandas and plotting. If you liked this exercise, look out for another similar notebook next week!
%%
恭喜你完成了本次实验。你已经见识了本课程中涵盖的多个概念的实现：概率、描述性统计，如均值、中位数、标准差和四分位数，你还绘制了箱线图和二维直方图来表示联合分布，并探究了边际分布。此外，你还练习了 Pandas 和绘图技巧。如果你喜欢这个练习，下周请留意另一个类似的 Notebook！
%%