---
aliases: [Ungraded Lab - Summary Statistics]
tags: []
created: 2025-12-30, 15:41:51
modified: 2025-12-31, 13:40:15
---

# Ungraded Lab - Summary Statistics

In this notebook, you will be working with two distinct datasets. You will notice that relying solely on the main statistical measures such as mean, variance (or standard deviation), and correlation may not always effectively describe the datasets. Therefore, it is always advisable to supplement these measures with visualization techniques and/or other statistical measures to gain a deeper understanding of the data.

%%
在这个 Notebook 中，你可能会注意到，如果依靠主要的统计指标比如平均，方差 （或者是标准差）还有相关性，可能无法有效的描述数据集。因此，明智的做法是：始终在这些数字统计指标的基础上，补充一些可视化绘图方法或其他统计手段，以便更深入地理解数据的本质
%%

You will be working with two well-known datasets: Anscombe's quartet and the Datasaurus Dozen dataset. These datasets are artificially generated and are used to illustrate the fact that some metrics can fail to capture important information present in a dataset. More specifically, these datasets are used to demonstrate how relying solely on metrics can sometimes be misleading. If you're interested, you can read more about Anscombe's quartet and the Datasaurus Dozen dataset at their respective [Wikipedia](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) page and [Autodesk Research](https://damassets.autodesk.net/content/dam/autodesk/research/publications-assets/pdf/same-stats-different-graphs.pdf) article.

%%
你将要分析的是两个著名的数据集：安斯库姆四重奏和恐龙数据集。这些数据集由人工创建，并且展示了一些指标无法捕捉的重要信息。更具体来说，这些数据集展示了如果仅仅使用上述指标会对数据的理解有所偏差。如果你有兴趣，可以在上面的链接中阅读更多关于安斯库姆四重奏和恐龙数据集的信息。
%%

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import utils
%matplotlib widget

```

# 1. First data set - Anscombe's quartet

This first dataset was initially constructed by the statistician Francis Anscombe to demonstrate both the importance of graphing data when analyzing it, and the effect of outliers and other influential observations on statistical properties. (From wikipedia)
%%
第一个数据集最初由统计学家弗朗西斯·安斯库姆构建，旨在展示在数据分析时绘制图表的重要性，以及异常值和其他有影响力的观测值对统计特性的影响。
%%

To read the dataset, which is stored in a `.csv file`, you can use the read_csv function in pandas. This function enables you to load a DataFrame immediately. For further information on this function, you can type help(pd.read_csv) in your code editor.
%%
要读取保存在 `.csv` 文件中的数据集，你可以使用 Pandas 的 `read_csv` 函数。此函数使您能够立即加载并保存到 DataFrame 中。要进一步了解这个函数，你可以在编辑器中输入 `help(pd.read_csv)`。
%%

```python
# This line of code reads the dataset named 'df_anscombe.csv', which is stored in the same directory as this notebook.
df_anscombe = pd.read_csv('df_anscombe.csv')
```

The call `df_anscombe.head()` will show you the first five rows of the data set, so you can have a look on its data.
%%
这个单元格中的 `df_anscombe.head()` 将显示数据集的前五行，这样可以看清数据结构。
%%
```python
df_anscombe.head()
```

> [!result]
> |     | x    | y    | group |
> | --- | ---- | ---- | ----- |
> | 0   | 10.0 | 8.04 | 1     |
> | 1   | 8.0  | 6.95 | 1     |
> | 2   | 13.0 | 7.58 | 1     |
> | 3   | 9.0  | 8.81 | 1     |
> | 4   | 11.0 | 8.33 | 1     |


```python
# Let's determine the number of groups present in this dataset.
df_anscombe.group.nunique()
```

> [!result]
> 	4

This dataset comprises of four groups of data, each containing two components - `x` and `y`. To analyze the data, you can obtain the mean and variance of each group, as well as the correlation between x and y within each group. Pandas provides a built-in function called `DataFrame.describe` that displays common statistics for each variable. To group the data by the group column, you can use the `DataFrame.groupby` function.
%%
这个数据集包含了四个组的数据，每个都包含了两个分量—— `x` 和 `y` 。要分析这个数据，你可以获取每组的均值和方差，以及每一组内 `x` 和 `y` 的相关性。Pandas 准备了一个名叫 `DataFrame.describe` 的内置函数，他可以显示每个变量常用的统计信息。如果想要对数据的 group 列进行分组，你可以使用 `DataFrame.groupby` 函数。
%%
The next block of code first groups the `DataFrame` based on the group column, and then applies the describe function to obtain the common statistics for each variable in each group.
%%
下一个单元格的代码首先对 df 进行了分组，然后对分组执行了 describe 函数，获取的每个分组变量的常用统计信息。
%%
```python
df_anscombe.groupby('group').describe()
```

> [!result]
> |       | x     |      |          |     |     |     |      |      | y     |          |          |      |       |      |      |       |
> | ----- | ----- | ---- | -------- | --- | --- | --- | ---- | ---- | ----- | -------- | -------- | ---- | ----- | ---- | ---- | ----- |
> |       | count | mean | std      | min | 25% | 50% | 75%  | max  | count | mean     | std      | min  | 25%   | 50%  | 75%  | max   |
> | group |       |      |          |     |     |     |      |      |       |          |          |      |       |      |      |       |
> | 1     | 11.0  | 9.0  | 3.316625 | 4.0 | 6.5 | 9.0 | 11.5 | 14.0 | 11.0  | 7.500909 | 2.031568 | 4.26 | 6.315 | 7.58 | 8.57 | 10.84 |
> | 2     | 11.0  | 9.0  | 3.316625 | 4.0 | 6.5 | 9.0 | 11.5 | 14.0 | 11.0  | 7.500909 | 2.031657 | 3.10 | 6.695 | 8.14 | 8.95 | 9.26  |
> | 3     | 11.0  | 9.0  | 3.316625 | 4.0 | 6.5 | 9.0 | 11.5 | 14.0 | 11.0  | 7.500000 | 2.030424 | 5.39 | 6.250 | 7.11 | 7.98 | 12.74 |
> | 4     | 11.0  | 9.0  | 3.316625 | 8.0 | 8.0 | 8.0 | 8.0  | 19.0 | 11.0  | 7.500909 | 2.030579 | 5.25 | 6.170 | 7.04 | 8.19 | 12.50 |


The groups appear to be quite similar, as evidenced by the identical mean and standard deviation values for both `x` and `y` within each group.
%%
这些组看上去非常相似，`x` 组和 `y` 组里面每行的均值和方差都是一样的。
%%
Additionally, you can analyze the correlation between `x` and `y` within each group.
%%
此外，您可以在每个组内分析 x 与 y 之间的相关性。
%%
To obtain the correlation matrix for each group, you can follow the same approach as before. First, group the data by the `group` column using `DataFrame.groupby`, and then apply the `.corr` function.
%%
为了得到每组的协方差矩阵，你可以沿用之前的方法。首先使用 `groupby` 对 `group` 列进行分组，然后使用 `.corr` 函数
%%
```python
df_anscombe.groupby('group').corr()
```

> [!result]
> |group||x|y|
> |---|---|---|---|
> |1|x|1.000000|0.816421|
> ||y|0.816421|1.000000|
> |2|x|1.000000|0.816237|
> ||y|0.816237|1.000000|
> |3|x|1.000000|0.816287|
> ||y|0.816287|1.000000|
> |4|x|1.000000|0.816521|
> ||y|0.816521|1.000000|


As observed, the correlation between `x` and `y` is identical within each group up to three decimal places. Moreover, the high correlation coefficient values suggest a strong linear correlation between `x` and `y` within each group.
%%
观察得知，在每个组内，`x` 与 `y` 的相关系数几乎都是相同的。此外，如此高的相关系数强烈表明 `x` 与 `y` 每组之间线性相关。
%%
Despite the similarities in the statistical measures for the groups, it is still necessary to visualize the data to get a better understanding of the differences, if any.
%%
尽管每个组的统计指标都非常相似，仍然有必要对数据进行可视化，以更好地了解可能存在的差异。
%%
```python
utils.plot_anscombes_quartet()
```

> [!result]
> 
![anscombe_quartet.png](https://obsidian-image.wwtt.xyz/2025/12/anscombe_quartet.png)

Upon visualizing the data, the four groups appear to be quite distinct:
%%
将数据可视化后，这四个群体似乎截然不同：
%%

1. The first group shows a clear linear relationship between `x` and `y`.
2. The second group, on the other hand, exhibits a non-linear pattern, indicating that the usual Pearson correlation may not be appropriate to describe the dataset.
3. The third group would be linear if it were not for a single outlier.
4. The fourth group demonstrates that `y` can have different values for the same `x`, suggesting that there is no clear relationship between `x` and `y`. However, there is also an outlier in this group.

%%
1. 第一组清晰的显示了 `x` 和 `y` 对应的线性关系。
2. 另一方面，第二组呈现出非线性模式，表面通常的皮尔逊关系可能无法描述这个数据集。
3. 第三组如果没有那个离群点，它就是一个完美的线性关系。
4. 第四组展示了同样的 `x` 值不同 `y` 值的情况，这表面 `x` 与 `y` 之间没有清晰的关系。而且这里同样有一个离群值。
%%

These four groups illustrate that summary statistics alone are not sufficient for investigating data. Visualizing the data, analyzing possible outliers, and identifying more complex relationships are essential to gain a better understanding of the underlying patterns in the data.

%%
这四组数据展示了汇总统计的方法用于调查数据的严重不足。可视化数据，分析可能离群的数据，并识别更复杂的关系对于深入理解数据中的潜在模式至关重要。
%%
# 2. Second data set - Datasaurus Dozen

The creation of Anscombe's quartet inspired other authors to generate datasets that have different relationships among its points but share the same summary statistics. One such dataset is the Datasaurus Dozen, which was created by AutoDesk. 
%%
安斯库姆四重奏的构建启发了其他的人，他们同样也创建了一些相同统计特征但数据点不同的数据集，其中比较代表性的是由 AutoDesk 公司创建的恐龙恐龙数据集。
%%
In this case, you will take a different approach. Instead of analyzing summary statistics and then plotting the data points, you will compare two datasets from the dozen and compute their statistics.
%%
在这种情况下，您将采取不同的方法。您不再先分析汇总统计数据再绘制数据点，而是比较这十二个中的两个数据集并计算它们的统计量。
%%
```python
df_datasaurus = pd.read_csv("datasaurus.csv")
```

The next cell will run a widget where you can investigate this dataset, which has different groups in it.
%%
下一个单元格将运行一个小部件，您可以在其中调查此数据集，其中包含不同的组。
%%
```python
utils.display_widget()
```

> [!result]
![datasaurus_dozen.png](https://obsidian-image.wwtt.xyz/2025/12/datasaurus_dozen.png)


As you have observed, the first dataset was not an anomaly; it is possible to have different datasets with the same summary statistics. Hence, it is essential to keep in mind while analyzing data that the summary statistics alone may not provide a complete picture of the underlying patterns and relationships.
%%
正如你所看到的，首先数据集没有异常值，可能存在不同数据集拥有相同的汇总统计量。因此，在分析数据时务必记住，仅凭汇总统计可能无法全面揭示潜在的模式和关系。
%%
Congratulations! You have completed this ungraded lab and now understand that summary statistics may not capture all the necessary information to describe your data accurately. Keep in mind that visualizations and more in-depth analyses are often needed to get a better understanding of your data.
