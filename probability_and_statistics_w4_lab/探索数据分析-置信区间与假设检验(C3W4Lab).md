---
aliases: [Exploratory Data Analysis - Confidence Intervals and Hypothesis Testing]
tags: []
created: 2026-01-18, 19:46:02
modified: 2026-01-19, 15:44:24
---

# Exploratory Data Analysis - Confidence Intervals and Hypothesis Testing

Welcome to the last notebook of the exploratory data analysis (EDA) series. For this notebook you will use the data on rideharing in the year 2022 in the city of Chicago, which can be found [here](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p/data). You have already worked with this dataset in the first and second notebooks of this series. This time you will continue working on the cleaned-up and reduced version of the dataset, which you prepared in week 1.

%% 欢迎来到数据探索分析系列最后的一个 Notebook。在这个 Notebook 中你将使用芝加哥 2022 年的网约车数据。你已经在这个系列的第一个和第二个 Notebook 中使用过它了。这次你将继续使用它清洗并简化后的版本，毕竟你已经在第一周处理过了。 %%
### Learning Objectives:
In this notebook you will use the following concepts from the course in a practical setting:
 - Descriptive statistics (mean, standard deviation)
 - Confidence intervals
 - Two sample t-test
 - Linear regression

%%
在这个 Notebook 中，你将在实际场景中使用以下的概念：
- 描述性统计（均值，标准差）
- 置信区间
- 双样本 t 测试
- 线性回归
%%
# 1. Import the Python Libraries

As usual, the first thing you need to do is import the libraries that you will use in this notebook.

%% 通常来说使用 Notebook 的第一件事是需要导入各种库。 %%

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.formula.api as smf
```

# 2. Load the Dataset

The next step is to open the dataset. The dataset has been downsampled to work smoothly in this environment.

%% 下一步是打开这个数据集。为了在这个环境中工作，这个数据集已经下采样。 %%

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

# 3. Investigate the Daily Number of Rides Through the Year

Have a look at the number of daily rides throughout the year. The goal here is to calculate the confidence interval for the (population) mean. This can help you predict the number of rides in the next year.

%% 观察一下这一年中的每日乘车次数。这里的目的是计算总体均值和置信区间。这可以帮助你越策次年的乘车数据。 %%

```python
# Caclulate the daily number of rides through the whole year
daily_rides = df.groupby('date').size().reset_index(name='daily_rides')

# Show the dataframe
daily_rides
```

> [!result]
> |     | date       | daily_rides |
> | --- | ---------- | ----------- |
> | 0   | 2022-01-01 | 1557        |
> | 1   | 2022-01-02 | 1102        |
> | 2   | 2022-01-03 | 1207        |
> | 3   | 2022-01-04 | 1151        |
> | 4   | 2022-01-05 | 1235        |
> | ... | ...        | ...         |
> | 360 | 2022-12-27 | 1557        |
> | 361 | 2022-12-28 | 1663        |
> | 362 | 2022-12-29 | 1797        |
> | 363 | 2022-12-30 | 2177        |
> | 364 | 2022-12-31 | 2720        |
> 365 rows × 2 columns



Now that you have the number of rides for each day, you can calculate the sample mean and standard deviation. The terms mean and standard deviation have been used very loosely in the previous notebooks of the series, but you know better now. Since this is only a sample of the cab rides in Chicago, everything you get from this data will be an estimation of the true population. In other words, you're calculating sample means and sample variances, rather than population means.

%% 现在有了每日的乘车次数，你可以计算样本均值和标准差了。在该系列之前的学习笔记中，我们曾粗略地使用了均值和标准差这两个术语，但现在你已经有了更准确的理解。由于这只是芝加哥的乘车样本，你得到的所有数据都来自于真实总体的估计。换句话说，你计算的样本均值和样本方差，而不是总体均值。 %%

```python
# Calculate the mean and standard deviation of the number of rides
mean_rides_per_day = daily_rides['daily_rides'].mean()
std_rides_per_day = daily_rides['daily_rides'].std()

print(f'Mean number of rides per day: {mean_rides_per_day:.2f}')
print(f'Standard deviation: {std_rides_per_day:.2f}')
```

> [!result]
>     Mean number of rides per day: 1893.42
> 	Standard deviation: 404.21

In the next cell you will plot the daily rides, and add a horizontal line representing the sample mean you got from the data

%% 下一个单元格将绘制每日的乘车数量，另外添加了一根水平线，代表你获得数据的样本均值。 %%

```python
plt.figure(figsize=(18,6))
# Plot the histogram of the daily rides
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')

# Plot the mean value as a horizontal line
plt.axhline(y=mean_rides_per_day, c='r', label=f'Mean Rides per Day')

plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides['date']))
plt.legend(fontsize=14)
plt.show()
```


    
![Rideshare_Project_Week4_10_0.png](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week4_10_0.png)


Above you have calculated the mean number of rides per day. But how confident can you be in this mean?

%% 上面计算了每日的乘车数量。但是对于这个均值此你有多大的信心？ %%
## 3.1 Confidence Intervals

As you learned in this week, one way of anwering the previous question is by using confidence intervals.  Go ahead and find the 95% confidence interval for the mean number of daily rides. Since you don't know the standard deviation (you actually know nothing of the distribution), you need to construct the confidence interval using the $t$ distribution.

%% 正如你这周学过的，回答上面问题的其中一个方法是使用置信区间。在每日乘车数的均值中找到 95%的置信区间。由于不知道标准差（即对分布情况一无所知。），你需要使用 t 分布构建置信区间。 %%

 ![t-CI-screenshot.png|250](https://obsidian-image.wwtt.xyz/2026/01/t-CI-screenshot.png)


```python
# Define the confidence interval you are interested in
confidence = 0.95

# Calculate the critical value using scipy
critical_value = scipy.stats.t.ppf(1 - (1 - confidence)/2, df=len(daily_rides)-1)

print(f"For the confidence interval of {confidence}, the critical value is {critical_value}")

# Calculate the confidence interval
total_days = daily_rides['date'].count()
confidence_interval = critical_value * std_rides_per_day / np.sqrt(total_days)

print(f"With a {100 * confidence}% confidence you can say that your error will be no more than {confidence_interval:.4f} rides per day.")
```

> [!result]
>     For the confidence interval of 0.95, the critical value is 1.966502568799249
>     With a 95.0% confidence you can say that your error will be no more than 41.6059 rides per day.


Now that you have the mean number of rides and the confidence interval, it would be good to plot them together, to see how they look.

%% 现在已经求得平均乘车次数和置信区间了，绘制它们看看是什么样子。 %%

```python
plt.figure(figsize=(18,6))
# Plot the histogram of the daily rides
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')

# Plot the mean value as a horizontal line
plt.axhline(y=mean_rides_per_day, c='r', label=f'Mean Rides per Day +/- {confidence}% Confidence Interval')
# Plot the confidence interval around the line
plt.fill_between(daily_rides['date'], mean_rides_per_day-confidence_interval,
                 mean_rides_per_day+confidence_interval, color='r', alpha=0.2)

plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides['date']))
plt.legend(fontsize=14)
plt.show()
```

> [!result]
![Rideshare_Project_Week4_14_0.png](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week4_14_0.png)


You can see that your confidence interval is very narrow, which implies that you can be very confident in your estimation of the mean. At least with 95% confidence, the population mean will not be too far from the sample mean. This observation may seem counterintuitive at first, as you can see that the data varies a lot more. But remember, the confidence interval does not describe the data, but the mean of the population. Since you have so many datapoints (one for each day), your estimate of the mean is very precise, even though the standard deviation is relatively large (about 10x larger than your confidence interval for the mean).

%% 可以看到你的置信区间非常狭窄，这意味着你可以对均值的估计非常有信心。至少有 95% 的把握可以确信，总体均值不会与样本均值相差太远。这一观察结果起初可能显得反直觉，正如你看到的，数据的变化要大的多。但是请记住，置信区间不描述数据，但是它意味着总体均值。由于你有很多的数据点（每天都有一个数据），在样本足够大的情况下你的估计将非常的精确，即使标准差相对较大（约比你平均值的置信区间大 10 倍左右）。 %%

Notice an interesting thing: in the last two weeks of the year (holiday season) the number of rides seems to drop quite significantly. Perhaps you could isolate this part of the data and separately estimate a confidence interval for the population mean for only those weeks.

%% 有一个有趣的事情需要注意：在这一年最后的两周（节假日期间）的乘车次数看上去掉得非常显著。也许你可以将这部分数据进行隔离，然后分别估计总体均值和这两周的置信区间。 %%

```python
# Select the data only for holidays
daily_rides_holidays = daily_rides[daily_rides["date"] > "2022-12-17"]

# Compute sample mean and standard deviation for holidays
mean_rides_per_day_holidays = daily_rides_holidays['daily_rides'].mean()
std_rides_per_day_holidays = daily_rides_holidays['daily_rides'].std()

print(f'Mean number of rides per day: {mean_rides_per_day_holidays:.2f} +/- {std_rides_per_day_holidays:.2f}')

# Get the confidence interval for the population mean for the holidays.
critical_value_holidays =  scipy.stats.t.ppf(1 - (1 - confidence)/2, df=len(daily_rides_holidays)-1)
total_days_holidays = daily_rides_holidays['date'].count()
confidence_interval_holidays = critical_value_holidays * std_rides_per_day_holidays / np.sqrt(total_days_holidays)
print(f"With a {100 * confidence}% confidence you can say that your error will be no more than {confidence_interval_holidays} rides per day.")
```

> [!result]
>     Mean number of rides per day: 1725.21 +/- 426.10
>     With a 95.0% confidence you can say that your error will be no more than 246.02055491847958 rides per day.

What you may notice here is that while the mean is lower and you have a similar standard deviation, the confidence interval for the mean is much larger. This is because you used only 14 datapoints (two weeks of data) rather than the whole year.

%% 你可能会注意到，最后两周的均值变低了，但是标准差还是相似的，置信区间的均值更大了。这是因为你只使用了 14 个数据点（两周的数据）而不是全年。 %%

In the cell below, you can plot the same plot again, but with the data for the holidays superimposed for comparison.

%% 在下面的单元格中，绘制了同样的图像，并且叠加了节假日的数据进行比较。 %%

```python
plt.figure(figsize=(18,6))

# Plot the histogram of the daily rides, the mean and the confidence interval
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')
plt.axhline(y=mean_rides_per_day, color='C0', label=f'Mean Rides per Day +/- {confidence}% Confidence Interval')
plt.fill_between(daily_rides['date'], mean_rides_per_day-confidence_interval,
                 mean_rides_per_day+confidence_interval, color='C0', alpha=0.3)

# Plot the histogram of the daily rides, the mean and the confidence interval for the holiday season
plt.bar(daily_rides_holidays['date'], daily_rides_holidays['daily_rides'], label='Rides per Day (Holidays)')
plt.axhline(y=mean_rides_per_day_holidays, color='C1', label='Mean Rides per Day (Holidays) +/- {confidence}% Confidence Interval')

plt.fill_between(daily_rides_holidays['date'], mean_rides_per_day_holidays-confidence_interval_holidays,
                 mean_rides_per_day_holidays+confidence_interval_holidays, color='C1', alpha=0.5)

plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides_holidays['date']))
plt.legend(fontsize=14)
plt.show()
```


    
![Rideshare_Project_Week4_18_0.png](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week4_18_0.png)


This is where confidence intervals come in realy handy as you can talk about the confidence of your estimates. Looking at the plot above you can see that the blue line (mean for the entire year) falls within the orange shaded area (95% confidence interval for the mean of the rides during holidays). Collecting data for more years would give you more datapoints and thus a narrower confidente interval for the same confidenve level. This means you are more certain about you sample mean estimation.

%% 在讨论估计值的置信度时，置信区间就派上用场了。检查上面的图，蓝色的线（全年的均值）落在橘色的阴影中（假日乘车数量均值的95%置信区间）。收集更多年份的数据可以得到更多的数据点，这样在相同的置信水平下可以获得更加狭窄的置信区间。这意味着你的样本估计会更加有把握。 %%
## 3.2 Two Sample t-test

Another thing you probably noticed in the plot are the periodic peaks. If you look closely, they appear with a period of 7 days, which gives you a hint that there are more rides on some days of the week than others. Run the cell below to group the weekdays together and calculate the mean and standard deviation for the number of rides each day.

%% 另外一件事是，你可能注意到了在图上的周期性峰值。如果你观察得足够仔细，它们出现的周期是 7 天一次，这暗示着一周中的某些日子比其他日子有更多的行程。运行下面这个单元格，它将一周内的每天乘车数做了聚合，并计算其均值和方差。 %%

```python
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

daily_rides['weekday'] = daily_rides['date'].apply(lambda x: WEEKDAYS[x.weekday()])

weekday_summary = daily_rides.groupby('weekday')['daily_rides'].describe()
# Reorder the series given weekdays
weekday_summary = weekday_summary.reindex(WEEKDAYS)
weekday_summary
```

> [!result]
> | |count|mean|std|min|25%|50%|75%|max|
|---|---|---|---|---|---|---|---|
|Monday|52.0|1519.480769|145.334839|1182.0|1459.25|1526.5|1610.25|1759.0|
|Tuesday|52.0|1588.000000|163.861371|1151.0|1523.75|1571.5|1671.75|1931.0|
|Wednesday|52.0|1692.961538|205.725971|908.0|1631.00|1702.5|1783.00|2121.0|
|Thursday|52.0|1840.788462|218.758739|1332.0|1777.25|1865.0|1947.00|2303.0|
|Friday|52.0|2229.288462|253.053458|1518.0|2136.00|2238.5|2364.50|2712.0|
|Saturday|53.0|2507.018868|331.347685|1435.0|2394.00|2528.0|2671.00|3320.0|
|Sunday|52.0|1864.596154|251.750675|977.0|1765.25|1910.5|2039.75|2238.0|


You can clearly see that there are a larger number of rides on Fridays and Saturdays than on the rest of the week (perhaps due to people going out on the weekend?). But can you be sure you have enough evidence to claim this? Let's find that out with a hypothesis test!

%% 你可以清晰的看到周五和周六的数据比其他天大得多（可能是由于周末人们外出？）。但你能确定你有足够的证据来证明这一点吗？让我们用假设检验来找到答案！ %%

In this case, you can test your assumption with a two sample t-test. Remember that for this test, you used the following test statistic

%% 在这个案例中，你可以使用双样本 t 测试来测试你的假设。针对此次测试，你使用了以下测试统计量。 %%

![2sample-t-test-screenshot.png|700](https://obsidian-image.wwtt.xyz/2026/01/2sample-t-test-screenshot.png)


This can be done very easily using the `scipy` library. You just need to call the function `scipy.stats.ttest_ind()` and pass it the two samples. If you call $\mu_{FS}$ the population mean for the number of rides on Fridays and Saturdays, and $\mu_{other}$ the population mean for the rest of the weekdays, in this case you can propose the the following hypotheses:

%% 使用 `scipy` 库可以非常简单的完成这个测试。你只需要调用函数 `scipy.stats.ttest_ind()`，然后传入两个样本。定义 $\mu_{FS}$ 为周五和周六的乘车次数的总体平均，然后定义 $\mu_{other}$ 为每周其他天的总体平均，然后你就可以提出如下假设： %%

 - Null hypothesis ($H_0$): $\mu_{FS} \leq \mu_{other}$ (population mean of the first group is smaller or equal than that for the the other group)
 - Alternative hypothesis ($H_1$): $\mu_{FS} > \mu_{other}$ (population mean of the first group is bigger than that for the the other group)

%% 
- 零假设($H_0$)：$\mu_{FS} \leq \mu_{other}$（第一组总体均值小于等于另一组）
- 备择假设($H_1$): $\mu_{FS} > \mu_{other}$（第一组总体均值大于另一组）
 %%

Since you want to prove that the the population mean on Fridays and Saturdays is bigger, that's what you set as the alternative hypothesis.

%% 由于你想证明周五周六的总体均值比较大，所以你将这个条件设置为了备择假设。 %%

The function returns the value of the statistic and the p-value. You can now compare the p-value with your desired significance level, for example $\alpha=0.05$, to determine whether you can reject the null hypothesis.

%% 这个函数返回统计值和 P 值。然后就可以比较 P 值和你设计的置信水平，比如 $\alpha=0.05$，用于确定是否拒绝零假设。 %%

```python
# Create two series, one for the numbers of rides on every friday and saturday and one for the other days
fridays_and_saturdays = daily_rides[daily_rides["weekday"].isin(["Friday", "Saturday"])]["daily_rides"]
other_days = daily_rides[daily_rides["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Sunday"])]["daily_rides"]

# Note that these series contain all days, not the summary from the dataframe in the previous cell
print(f"Number of datapoints for Fridays and Saturdays: {len(fridays_and_saturdays)}")
print(f"Number of datapoints for other days: {len(other_days)}")

# Calculate the t
scipy.stats.ttest_ind(a=fridays_and_saturdays, b=other_days, alternative='greater')
```

> [!result]
>     Number of datapoints for Fridays and Saturdays: 105
>     Number of datapoints for other days: 260
> 
>     Ttest_indResult(statistic=21.56892206841357, pvalue=2.6591073725083493e-67)

Here the output gives you an extremely small p-value, which means you would reject the null hypothesis and say that people actually take more rides on Fridays and Saturdays, even with very tiny significance levels.

%% 这里给出了一个极小的 P 值，这意味着你可以拒绝零假设，并且声明：即使是非常微小的显著性水平下，人们在周五和周六乘坐网约车比其他时间多。 %%
# 4. Calculating the Fares Given Trip Distance and Time

In this section you will try to try to calculate how much the drivers can charge for the rides given your data. Usually rideshare comapanies charge a certain amount per unit time and a certain amount for the distance covered. You can assume that this is also the case here. First you can plot some scatter plots to see how the variables correlate with each other.

%% 在这个部分你将尝试计算，根据现有的数据，司机对车程可以收取多少费用。通常，网约车公司会根据单位时间和行驶距离收取费用。你可以认为示例中也是这样的情况。首先你可以绘制一些散点图，观察一下变量之间的相关性。 %%


```python
fig, ax = plt.subplots(1,2, figsize=(12,4))
df.plot.scatter('fare','trip_seconds', ax=ax[0])
df.plot.scatter('fare','trip_miles', ax=ax[1])
```

> [!result]
>     <Axes: xlabel='fare', ylabel='trip_miles'>
![Rideshare_Project_Week4_24_1.png](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week4_24_1.png)

    
As you can see, these variables seem highly correlated and are good candidates for a linear regression. In the previous week "World Happiness" lab, you used `scikit-learn` to perform linear regression. This is a machine learning oriented library. This week you will use another library called `statsmodels`. You can find the documentation [here](https://www.statsmodels.org/stable/index.html). This library is much more statistics oriented, but using it is just as easy. With just a few lines of code, you can fit the model and display a nice and detailed summary.

%% 正如你所看到的，这些数据看起来高度相关，同时也是线性回归的优秀候选。在上周的 “世界幸福指数” 实验中，你使用了 `scikit-learn` 来执行线性回归。它是一个面向机器学习的库。这周你及那个使用另一个被称为 `statsmodels` 的库。这个库更加面向统计，非常易于使用。只需要短短几行，就可以拟合这个模型，然后展示一个美观且详尽的摘要。 %%

In the cell below you will fit a model, using `trip_seconds` and `trip_miles` as explanatory variables and `fare` as the response.

%% 下面的单元格将拟合一个模型，使用 `trip_seconds` （行程秒数）和 `trip_miles` （行程英里数）作为解释变量，`fare` （车费）作为响应变量。 %%

```python
# Create the model
model = smf.ols(formula='fare ~ trip_seconds + trip_miles', data=df)

# Fit the model
result = model.fit()

# Display the results
print(result.summary())
```

> [!result]
>                                 OLS Regression Results                            
>     ==============================================================================
>     Dep. Variable:                   fare   R-squared:                       0.574
>     Model:                            OLS   Adj. R-squared:                  0.574
>     Method:                 Least Squares   F-statistic:                 4.655e+05
>     Date:                Sun, 18 Jan 2026   Prob (F-statistic):               0.00
>     Time:                        19:42:30   Log-Likelihood:            -2.5085e+06
>     No. Observations:              689925   AIC:                         5.017e+06
>     Df Residuals:                  689922   BIC:                         5.017e+06
>     Df Model:                           2                                         
>     Covariance Type:            nonrobust                                         
>     ================================================================================
>                        coef    std err          t      P>|t|      [0.025      0.975]
>     --------------------------------------------------------------------------------
>     Intercept        6.4621      0.020    329.648      0.000       6.424       6.501
>     trip_miles       0.8693      0.003    343.709      0.000       0.864       0.874
>     ==============================================================================
>     Omnibus:                   495203.284   Durbin-Watson:                   1.666
>     Prob(Omnibus):                  0.000   Jarque-Bera (JB):         25420445.937
>     Skew:                           2.917   Prob(JB):                         0.00
>     Kurtosis:                      32.159   Cond. No.                     2.38e+03
>     ==============================================================================
 >    
>     Notes:
>     [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
>     [2] The condition number is large, 2.38e+03. This might indicate that there are
>     strong multicollinearity or other numerical problems.


One interesting thing about using the StatsModels library it that it outputs a lot of information to help you decide if the model you proposed and trained is a good one. As you can see there are quite a lot of details about the model in this table.  While still very useful, some of them are outside the scope of this course. That said, you can still undersand a great deal of them! Don't be intimidated by all this information, here you will understand how to interpret most of it

%% 一个有趣的事情是关于使用 StatsModels 库，它输出了很多信息来帮助你判定你提出并训练的这个模型是不是一个好模型。正如你所看到的，表格里有相当多有关模型的细节。虽然它们都非常有用，但是一些还是超出了这门课程的范畴。即便如此，你仍然可以理解它们的大部分内容！不要被这些信息吓到，在这里你将学会解读大部分内容。 %%

The first part gives you information about the model:

%% 这个模型给出的第一部分的信息如下： %%

```
Dep. Variable:                   fare  
Model:                            OLS  
Method:                 Least Squares  
Date:                Thu, 18 Jan 2024  
Time:                        20:45:54  
No. Observations:              689925  
Df Residuals:                  689922  
Df Model:                           2                                         
```

You can see that the dependent variable is the column `fare`, you see that it is using an OLS (Ordinary Least Squares) model to represent the linear regression. You can also check the date and time of when the model was trained. You can also check the number of obervations that were used to train the model. The residuals is nothing more than the sum of squared errors of the model, compared against the training data, and Df Model indicates how many explicative variables you used to train. You can forget about the column on the right for now. It gives you a lot of metrics to compare between different models, but it is way outside the scope of the course. 

%% 你可以看到，独立变量为列 `fare`，可以看到它正在使用 OLS（普通最小二乘法）模型来表示线性回归。你也可以看到当模型训练完成后的时间和日期。也能看到您也可以查看用于训练模型所用观测数据的数量。残差即模型相对于训练数据的误差平方和，而模型自由度则代表了用于训练的显性变量数目。你现在可以暂时忘掉右侧那列。它为你提供了大量指标，以比较不同模型之间的差异，但这远远超出了课程的范围。 %%

Now, if you go to the middle section of the summary 

%% 下面是中间部分的摘要 %%

```
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        6.4621      0.020    329.648      0.000       6.424       6.501
trip_seconds     0.0056   2.51e-05    222.342      0.000       0.006       0.006
trip_miles       0.8693      0.003    343.709      0.000       0.864       0.874
```

it is giving you the value of the coefficients of the regression, but it gives you much more than that! Lets analyze column by column:
- `coef`: you get the estimated value for each of the coefficients. 
- `std_err`: shows you the standard error of the estimated value for each coefficient. 

%% 它不仅给出了回归系数的值，还提供了更多信息！让我们逐列分析：
- `coef`: 每个系数的估计值。
- `std_err`: 每个预估系数的标准差。 %%

The following columns all give you measures of how good the estimate is:
- `t`: computes the t statistic for the given coefficient. It is the same t statistic you used both in hypothesis testing and confidence intervals. Note that if the standard error is very smalled compared to the coeficient, this will make for a large statistic. 
- `P>|t|`: this column is very important. It represents the p-value for the t-statistic you got for null hypothesis that the coefficient is actually 0. In other words, it shows how likely your coefficient is measured through your model by chance. In this example, all this probabilities are 0, so it means all the considered variables (and the intercept) play an important role on the regression model
- `[0.025    0.975]`: these two columns give you the lower and upper limmits for a 95% confidence interval for each coefficient. In this example, since you've got a lot of observations and proposed a good model, this intervals are very narrow. 
%% 
以下列出的所有列都提供了评估估计值好坏程度的指标：
- `t`: 根据给定的系数计算 t 统计量。它课程中的假设检验和置信区间使用的 t 统计量相同。请注意，如果标准误差远小于系数，将会产生较大的统计量。
- `P>|t|`: 这一列非常重要。它表示在假设系数实际为零的零假设下，所获得的 t 统计量的 p 值。换句话说，它显示了通过你的模型偶然测量到你的系数的可能性有多大。在这个例子中，所有的概率都是 0，这意味着所有需要考虑的变量（包括截距）在回归模型中都起着重要作用。
- `[0.025    0.975]`: 这两列给出了每个系数的 95%置信区间的下限和上限。在这个例子中，由于你提供了很多观察数据和一个好的模型，置信区间的间隔非常的窄。
 %%

Now let's see what information you can get about the remaining values. `Skew` and `kurtosis` give you the skewness and kurtosis of the residues. Remember that the residue is the squared difference between your observation and the predicted output of the model. It is a good way to know how the errors of the model are distributed. 

%% 现在看看剩下的值里有什么信息。`Skew` 和 `kurtosis` 给出了残差的峰度和偏度。残差是你的观察值和模型输出的预测值的平方差。这是了解模型误差分布情况的一个好办法。 %%

This is the last of the values you can actually completely understand from what you've seen on this course. However, there are more that you can at least intuitively understand. Both  (`Omnibus` and `Prob(Omnibus)`) and (`Jarque-Bera (JB)` and `Prob(JB)`) represent hypothesis tests. They are both very particular tests, where the alternative hypothesis is that the residues are not normally distributed. Of course, they are not like any of the tests you've leaned so far, because it doesn't use the same statistics, but the way in which you interpret the test is still the same. The values of `Omnibus` and `Jarque-Bera (JB)` are the values of the statistic of each test, while `Prob(Omnibus)` and `Prob(JB` are the p-values. Note that in both of this tests, a small p-value indicates that the residues are not normally distributed.

%% 
“这是你根据本课程所学内容能完全理解的最后一个数值。不过，还有一些数值是你至少可以凭直觉去理解的。
`Omnibus`（及其对应的 `Prob(Omnibus)`）和 `Jarque-Bera (JB)`（及其对应的 `Prob(JB)`）都代表假设检验。

它们是非常特定的检验，其备择假设（Alternative Hypothesis）是‘残差不服从正态分布’。当然，它们并不像你目前学过的任何检验，因为它们使用的统计量不同，但你解读检验结果的方式是一样的。

`Omnibus` 和 `Jarque-Bera (JB)` 的值是每个检验的统计量数值，而 `Prob(Omnibus)` 和 `Prob(JB)` 则是对应的 P值（p-values）。请注意，在这两个检验中，**较小的 P值意味着残差不服从正态分布**。”
%%
 
The rest of the values on the summary table are completely outside the scope of the course, but you have plenty of information you can work with!

%% 在摘要表格中剩下的值都是完全不在本课程的范围内，但是你手里已经有了充足的信息。 %%

From the `result` object you can access a lot of the information. For example, run the cell below to access the parameters of the fit.

%% 从 `result` 对象中你可以访问很多信息。比如，运行下面的单元格访问拟合的参数。 %%

```python
result.params
```

> [!result]
>     Intercept       6.462145
>     trip_seconds    0.005589
>     trip_miles      0.869281
>     dtype: float64

You can also access each individual parameter as shown in the cell below.

%% 你还可以访问每个单独的参数，比如在下面的单元格中：%%

```python
starting_fare = result.params["Intercept"]
price_per_second = result.params["trip_seconds"]
price_per_mile = result.params["trip_miles"]

print(f"The starting fare is {starting_fare:.3} USD. In addition the ride costs {price_per_second*60:.3} USD per minute and {price_per_mile:.3} USD per mile.")
```

> [!result]
>     The starting fare is 6.46 USD. In addition the ride costs 0.335 USD per minute and 0.869 USD per mile.


Now that you have the coefficients for your model, you can define a simple fare calculator that can predicts the price of the trip for you based on the distiance and the duration.

%% 现在你已经获得了模型的系数，你可以定义一个简单的车费计算器，它可以根据距离和时长来预测你的行程价格。 %%

```python
def fare_calculator(trip_time, trip_distance):
    return starting_fare + price_per_second * trip_time + price_per_mile * trip_distance

sample_trip_duration = 10 * 60 # 10 minutes
sample_trip_distance = 10 # 10 miles

sample_fare = fare_calculator(sample_trip_duration, sample_trip_distance)

print(f"For a {sample_trip_distance} mile trip that takes {sample_trip_duration/60} minutes, you would pay around {sample_fare:.3} USD.")
```

> [!result]
>     For a 10 mile trip that takes 10.0 minutes, you would pay around 18.5 USD.


Lastly, plot the data and the predictions together to see how well the model performs.

%% 最后，将数据与预测结果绘制在一起，以观察模型的表现。 %%

```python
# Get the x and y data that you used to fit the model and drop nan values
x_y = df[["trip_miles", "trip_seconds", "fare"]].dropna()

# Change this row if you want to choose another x variable (trip_miles or trip_seconds) to plot
x_variable = "trip_seconds"

# Get the plotting data
x_plot =  x_y[x_variable]
y_plot =  x_y["fare"]
y_result = result.predict()

# Plot the data
plt.scatter(x_plot, y_plot, label="Original Data")
plt.scatter(x_plot, y_result, label="Prediction")
plt.xlabel(" ".join(x_variable.split("_")).title(), fontsize=14)
plt.ylabel("Fare", fontsize=14)
plt.legend(fontsize=14)
```

> [!result]
>     <matplotlib.legend.Legend at 0x7f9d40130820>  
![Rideshare_Project_Week4_34_1.png|600](https://obsidian-image.wwtt.xyz/2026/01/Rideshare_Project_Week4_34_1.png)


Of course there is some variance in the data that is not explained by the model, but it didn't do that bad for a model that only uses 2 variables!

%% 当然，数据中的一些变量模型是无法解释的，但是对于一个只使用两个变量的模型来说，它表现得很好。 %%

**Congratulations on finishing this lab.** You have used the implementation of quite a few concepts covered in this course: descriptive statistics (mean, standard deviation), confidence intervals, two sample t-test and linear regression. On top of that you have practiced Pandas and plotting. We hope you have enjoyed this series of notebooks!

%% 恭喜你完成了这个实验。你已经使用了本课程中涵盖的许多概念的实现：描述性统计（平均值、标准差）、置信区间、双样本 t 检验和线性回归。最重要的是，你已经练习了 Pandas 和绘图。我们希望你喜欢这个系列的 notebook！ %%