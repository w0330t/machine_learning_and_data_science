---
aliases: [Exploratory Data Analysis - Understanding Your Dataset]
tags: []
created: 2025-12-21, 09:46:15
modified: 2025-12-21, 22:49:55
---

# Exploratory Data Analysis - Understanding Your Dataset

Welcome to the second notebook of the exploratory data analysis (EDA) series, where you will get your hands dirty applying the skills you have learned in the course on an actual data problem, similar to those you might encouter in real life! This is a part of a series, which contains five notebooks, each of them placed on different weeks of this course. There is very little mathematics instruction in these notebooks, but rather practical implementations of the concepts you learned using various python libraries.

%%
欢迎来到探索数据分析系列（EDA）的第二篇笔记，在这里，你将亲自动手，将课程中学到的知识应用于真实的数据问题中，就像在实际生活中遇到的那样。这是一个系列，一共有 5 个笔记，它们分别安排在本课程的不同周次中。这些笔记本中数学教学的内容很少，主要是利用各种 Python 库将所学概念进行实际应用。
%%

For this notebook you will use the data on ridesharing in the year 2022 in the city of Chicago, which can be found [here](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p/data).

%%
在这个笔记中你将使用 2022 年芝加哥网约车的数据，数据源可以在[这里](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p/data)找到。
%%

We have already downloaded the dataset for you and put it in the folder together with this notebook. If you check the link above, you might notice that the dataset includes hundreds of millions of rows. This translates to tens of gigabytes and is too large for working in this environment. That's why the dataset has been preprocessed to include only the data from 2022 and downsampled by a factor of 100 to easily fit into the environment and make your experience more pleasant.

%%
我们已经为您下载了数据集，并将其与本笔记本一起放入文件夹中。如果你点击率上面的连接，你可以注意到数据集包含了数亿行的数据。这相当于有几十 G 的数据，对于我们这个学习环境而言，它太大了。因此，数据集已预先处理，它仅包含 2022 年的数据，并进行了 100 倍的下采样，以便轻松适应环境，使您的体验更加顺畅。
%%

In this notebook you will mostly use the Pandas library. If you are not familiar with it, you can check out the Pandas tutorial notebook.

%%
在这个笔记中主要使用了 Pandas 的库。如果你不熟悉它，你可以查阅 Pandas 的教程。
%%
### Learning Objectives:

In this notebook you will use the following concepts from the course in a practical setting:
 - Probability
 - Conditional probability
 - Distributions

%%
在本笔记本中，您将在实际环境中运用课程中的以下概念：
- 概率
- 条件概率
- 分布
%%
# 1. Import the Python Libraries

As usual, the first thing you need to do is import the libraries that you will use in this notebook. `pandas` will help you load and manipulate data, while `matplotlib` will be used for plottting.

%%
和往常一样，首先需要导入这个Jupyter Notebook中将要使用的库。`pandas` 将帮助你加载和处理数据，而 `matplotlib` 将用于绘制图表。
%%

```python
import pandas as pd
import matplotlib.pyplot as plt
```

# 2. Load the Dataset

The next step is to load the dataset. The dataset has been downsampled by a factor of 100 to work smoothly in this environment.

%%
下一步则是载入数据集，数据集已按 100 倍进行下采样，以确保在此环境中运行顺畅。
%%

```python
# Open the dataset
df = pd.read_csv("data/rideshare_2022.csv", parse_dates=['Trip Start Timestamp', 'Trip End Timestamp'])

# Show the first five rows of the dataset
df.head()
```

|     | Trip ID                                  | Trip Start Timestamp | Trip End Timestamp  | Trip Seconds | Trip Miles | Pickup Census Tract | Dropoff Census Tract | Pickup Community Area | Dropoff Community Area | Fare | ... | Trip Total | Shared Trip Authorized | Trips Pooled | Pickup Centroid Latitude | Pickup Centroid Longitude | Pickup Centroid Location             | Dropoff Centroid Latitude | Dropoff Centroid Longitude | Dropoff Centroid Location            | len_date |
| --- | ---------------------------------------- | -------------------- | ------------------- | ------------ | ---------- | ------------------- | -------------------- | --------------------- | ---------------------- | ---- | --- | ---------- | ---------------------- | ------------ | ------------------------ | ------------------------- | ------------------------------------ | ------------------------- | -------------------------- | ------------------------------------ | -------- |
| 0   | 04767642defd6a3825d089ae66183906a89b902d | 2022-01-01           | 2022-01-01 01:15:00 | 3905.0       | 44.5       | 1.703104e+10        | NaN                  | 4.0                   | NaN                    | 55.0 | ... | 66.25      | 0                      | 1            | 41.972563                | -87.678846                | POINT (-87.6788459662 41.9725625375) | NaN                       | NaN                        | NaN                                  | 16       |
| 1   | 138de88e19e045d9962f1f669e668f9dcdfbc9fd | 2022-01-01           | 2022-01-01 00:30:00 | 2299.0       | 25.0       | NaN                 | NaN                  | 32.0                  | NaN                    | 32.5 | ... | 46.68      | 0                      | 1            | 41.878866                | -87.625192                | POINT (-87.6251921424 41.8788655841) | NaN                       | NaN                        | NaN                                  | 16       |
| 2   | 249cb7bc8eea309aaa3ef941756df4f62a53a92a | 2022-01-01           | 2022-01-01 00:00:00 | 275.0        | 1.5        | NaN                 | NaN                  | 40.0                  | 38.0                   | 7.5  | ... | 8.52       | 0                      | 1            | 41.792357                | -87.617931                | POINT (-87.6179313803 41.7923572233) | 41.812949                 | -87.617860                 | POINT (-87.6178596758 41.8129489392) | 16       |
| 3   | 36c8a2a4cd85fb32ae32170550d2a4d30b8df8a1 | 2022-01-01           | 2022-01-01 00:15:00 | 243.0        | 1.0        | 1.703106e+10        | 1.703106e+10         | 6.0                   | 6.0                    | 5.0  | ... | 7.36       | 0                      | 1            | 41.936310                | -87.651563                | POINT (-87.6515625922 41.9363101308) | 41.943155                 | -87.640698                 | POINT (-87.640698076 41.9431550855)  | 16       |
| 4   | 493f7bbcba1d96bf10bd579fe1c4b7ddb95fd3a6 | 2022-01-01           | 2022-01-01 00:15:00 | 364.0        | 1.3        | 1.703107e+10        | 1.703106e+10         | 7.0                   | 6.0                    | 5.0  | ... | 7.36       | 0                      | 1            | 41.921855                | -87.646211                | POINT (-87.6462109769 41.9218549112) | 41.936237                 | -87.656412                 | POINT (-87.6564115308 41.9362371791) | 16       |
# 3 Explore the Dataset

In the cell above, you have opened the dataset and displayed the first five rows. Have a closer look at the output of the cell above. The dataset consists of the following columns:

%%
在上面的单元格中，你已经读取了数据集，并输出了它的前 5 行。仔细观察上面输出的单元格。该数据集由下面的列组成。
%%

- `Trip ID`: A unique identifier for the trip. 
- `Trip Start Timestamp`: When the trip started, rounded to the nearest 15 minutes.
- `Trip End Timestamp`: When the trip ended, rounded to the nearest 15 minutes.
- `Trip Seconds`: Time of the trip in seconds.
- `Trip Miles`: Distance of the trip in miles.
- `Pickup Census Tract`: The Census Tract where the trip began. This column often will be blank for locations outside Chicago.
- `Dropoff Census Tract`: The Census Tract where the trip ended. This column often will be blank for locations outside Chicago.
- `Pickup Community Area`: The Community Area where the trip began. This column will be blank for locations outside Chicago.
- `Dropoff Community Area`: The Community Area where the trip ended. This column will be blank for locations outside Chicago.
- `Fare`: The fare for the trip, rounded to the nearest $2.50. 
- `Tip`: The tip for the trip, rounded to the nearest $1.00. Cash tips will not be recorded.
- `Additional Charges`: The taxes, fees, and any other charges for the trip.
- `Trip Total`: Total cost of the trip. This is calculated as the total of the previous columns, including rounding.
- `Shared Trip Authorized`: Whether the customer agreed to a shared trip with another customer, regardless of whether the customer was actually matched for a shared trip.
- `Trips Pooled`: If customers were matched for a shared trip, how many trips, including this one, were pooled. All customer trips from the time the vehicle was empty until it was empty again contribute to this count, even if some customers were never present in the vehicle at the same time. Each trip making up the overall shared trip will have a separate record in this dataset, with the same value in this column.
- `Pickup Centroid Latitude`: The latitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
- `Pickup Centroid Longitude`: The longitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
- `Pickup Centroid Location`: The location of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
- `Dropoff Centroid Latitude`: The latitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
- `Dropoff Centroid Longitude`: The longitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
- `Dropoff Centroid Location`: The location of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.

%%
- `Trip ID`: 行程 ID。
- `Trip Start Timestamp`: 行程开始时间，四舍五入到 15 分钟。
- `Trip End Timestamp`: 行程结束时间，四舍五入到 15 分钟。
- `Trip Seconds`: 行程时间（秒）
- `Trip Miles`: 行程距离（英里）
- `Pickup Census Tract`: 行程起始的普查区域。对于芝加哥以外的地点，此列通常为空白。
- `Dropoff Census Tract`: 行程终止的普查区域。对于芝加哥以外的地点，此列通常为空白。
- `Pickup Community Area`: 行程起始的社区区域。对于芝加哥以外的地点，此列通常为空白。
- `Dropoff Community Area`: 行程终止的社区区域。对于芝加哥以外的地点，此列通常为空白。
- `Fare`: 行程的车费，四舍五入到 2.50 刀。
- `Tip`: 行程的小费，四舍五入到 1 刀，小费为现金则不计入。
- `Additional Charges`: 行程的税费和其他费用。
- `Trip Total`: 行程的总费用，此数值为前几列的总和。含四舍五入。
- `Shared Trip Authorized`: 客户是否同意与其他客户共享行程，不论其是否实际能匹配到共享行程。
- `Trips Pooled`: 如果乘客被匹配为共享行程，包括本次行程在内，该共享行程总计包含多少次行程。从车辆空载开始到再次空载期间的所有乘客行程均计入本次计数，即使某些乘客从未同时出现在车辆内。构成该次完整共享行程的每一段乘客行程，在数据集中都有单独记录，且在本列中具有相同的值。
- `Pickup Centroid Latitude`: 获取上车时人口普查区域或社区区域中心的纬度，若普查区域因隐私原因被隐藏，则指社区区域中心。对于芝加哥以外的地点，此列通常为空。
- `Pickup Centroid Longitude`: 获取上车时人口普查区域或社区区域中心的经度，若普查区域因隐私原因被隐藏，则指社区区域中心。对于芝加哥以外的地点，此列通常为空。
- `Pickup Centroid Location`: 获取上车时人口普查区域或社区区域中心的位置，若普查区域因隐私原因被隐藏，则指社区区域中心。对于芝加哥以外的地点，此列通常为空。
- `Dropoff Centroid Latitude`: 获取下车时人口普查区域或社区区域中心的纬度，若普查区域因隐私原因被隐藏，则指社区区域中心。对于芝加哥以外的地点，此列通常为空。
- `Dropoff Centroid Longitude`: 获取下车时人口普查区域或社区区域中心的经度，若普查区域因隐私原因被隐藏，则指社区区域中心。对于芝加哥以外的地点，此列通常为空。
- `Dropoff Centroid Location`: 获取下车时人口普查区域或社区区域中心的位置，若普查区域因隐私原因被隐藏，则指社区区域中心。对于芝加哥以外的地点，此列通常为空。
%%

Run the cell below to print out the column names and inspect the number of non-null values and the data type of each column. 

%%
运行下面的单元格以打印出列名，并检查非空值的数量和每列的数据类型。
%%

```python
df.info()
```

> [!result]
>     <class 'pandas.core.frame.DataFrame'>
>     RangeIndex: 691098 entries, 0 to 691097
>     Data columns (total 22 columns):
>      #   Column                      Non-Null Count   Dtype         
>     ---  ------                      --------------   -----         
>      0   Trip ID                     691098 non-null  object        
>      1   Trip Start Timestamp        691098 non-null  datetime64[ns]
>      2   Trip End Timestamp          691098 non-null  datetime64[ns]
>      3   Trip Seconds                691073 non-null  float64       
>      4   Trip Miles                  691096 non-null  float64       
>      5   Pickup Census Tract         398943 non-null  float64       
>      6   Dropoff Census Tract        397574 non-null  float64       
>      7   Pickup Community Area       633092 non-null  float64       
>      8   Dropoff Community Area      630431 non-null  float64       
>      9   Fare                        689952 non-null  float64       
>      10  Tip                         689952 non-null  float64       
>      11  Additional Charges          689952 non-null  float64       
>      12  Trip Total                  689952 non-null  float64       
>      13  Shared Trip Authorized      691098 non-null  int64         
>      14  Trips Pooled                691098 non-null  int64         
>      15  Pickup Centroid Latitude    635075 non-null  float64       
>      16  Pickup Centroid Longitude   635075 non-null  float64       
>      17  Pickup Centroid Location    635075 non-null  object        
>      18  Dropoff Centroid Latitude   632163 non-null  float64       
>      19  Dropoff Centroid Longitude  632163 non-null  float64       
>      20  Dropoff Centroid Location   632163 non-null  object        
>      21  len_date                    691098 non-null  int64         
>     dtypes: datetime64 [ns](2), float64(14), int64(3), object(3)
>     memory usage: 116.0+ MB

## 3.1 Select columns of interest

At this point, you have seen what the dataset looks like. Take a moment to think of your next steps. Which columns would you explore further? Is there a column that has a problematic number of null values? Are there any columns that you are not interested in?

%%
现在你已经了解数据集的样子了。想一想接下来的工作。你想进一步研究哪些列？某些列是否存在空值问题？哪些列对你来说没有价值？
%%

For exploratory data analysis it is perfectly fine to select only the columns that you are interested in and drop the remainder. This will not only make your dataframe easier to work with, but also reduce its size, making your operations faster.

%%
对数据进行研究分析，仅选择你感兴趣的列，并舍弃掉其他部分即可。这不仅可以让你更轻松的处理 dataframe，同时缩小它的尺寸，让处理的速度更快。
%%

In the cell below you will select a subset of the columns, which are the ones you will be interested in for this notebook. If you keep only the columns that are pre-selected in the cell below, you will reduce the file size by about a half. This can make a difference of whether you can fit the file into the memory or not, especially with larger files. This code will also rename the columns to remove white spaces.

%%
在下面的单元格中你将选择列的子集，这些列应该都是你在这个 Notebook 中最感兴趣的。如果你仅仅保留下面单元格中的这些列，这会让文件的尺寸减半。这对内存的影响很大，特别是文件很大的时候。下面的代码同时处理的列名的空格问题。
%%

```python
columns_of_interest = ['Trip Start Timestamp', 'Trip Seconds',
       'Trip Miles', 'Fare', 'Tip', 'Additional Charges', 'Trip Total', 'Shared Trip Authorized',
       'Trips Pooled', 'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 'Dropoff Centroid Latitude',
       'Dropoff Centroid Longitude']

df = df[columns_of_interest]

# Rename all the columns to not include whitespace
df = df.rename(columns={i: "_".join(i.split(" ")).lower() for i in df.columns})

# Check the info on the cleaned-up dataset
df.info()
```

> [!result]
>     <class 'pandas.core.frame.DataFrame'>
> 	RangeIndex: 691098 entries, 0 to 691097
> 	Data columns (total 13 columns):
>     #   Column                      Non-Null Count   Dtype         
>     ---  ------                      --------------   -----    
>     0   trip_start_timestamp        691098 non-null  datetime64[ns]
>     1   trip_seconds                691073 non-null  float64       
>     2   trip_miles                  691096 non-null  float64       
>     3   fare                        689952 non-null  float64       
>     4   tip                         689952 non-null  float64       
>     5   additional_charges          689952 non-null  float64       
>     6   trip_total                  689952 non-null  float64       
>     7   shared_trip_authorized      691098 non-null  int64         
>     8   trips_pooled                691098 non-null  int64         
>     9   pickup_centroid_latitude    635075 non-null  float64       
>     10  pickup_centroid_longitude   635075 non-null  float64       
>     11  dropoff_centroid_latitude   632163 non-null  float64       
>     12  dropoff_centroid_longitude  632163 non-null  float64       
>     dtypes: datetime64 [ns](1), float64(10), int64(2)
>     memory usage: 68.5 MB
# 4. Visualize the data

To understand the data better, it often makes sense to visualize it. This helps you understand how the data is distributed. You can start by plotting the number of rides in a given day. For this it would be useful to have another column that just contains the date. The code in the cell below will create a new column which takes the `trip_start_timestamp` and converts it into a date.

%%
为了更好的了解数据，一般来说会将其可视化。它可以帮助你了解数据的分布情况。你可以从绘制某一天的乘车数据开始。因此，增加一个仅包含日期的列将很有帮助。下面的单元格创建了一个新列，它读取了 `trip_start_timestamp` 的数据并将其转换为 date 类型的数据。
%%

```python
df['date'] = pd.to_datetime(df['trip_start_timestamp'].dt.date)

df.head()
```

|     | trip_start_timestamp | trip_seconds | trip_miles | fare | tip | additional_charges | trip_total | shared_trip_authorized | trips_pooled | pickup_centroid_latitude | pickup_centroid_longitude | dropoff_centroid_latitude | dropoff_centroid_longitude | date       |
| --- | -------------------- | ------------ | ---------- | ---- | --- | ------------------ | ---------- | ---------------------- | ------------ | ------------------------ | ------------------------- | ------------------------- | -------------------------- | ---------- |
| 0   | 2022-01-01           | 3905.0       | 44.5       | 55.0 | 0.0 | 11.25              | 66.25      | 0                      | 1            | 41.972563                | -87.678846                | NaN                       | NaN                        | 2022-01-01 |
| 1   | 2022-01-01           | 2299.0       | 25.0       | 32.5 | 7.0 | 7.18               | 46.68      | 0                      | 1            | 41.878866                | -87.625192                | NaN                       | NaN                        | 2022-01-01 |
| 2   | 2022-01-01           | 275.0        | 1.5        | 7.5  | 0.0 | 1.02               | 8.52       | 0                      | 1            | 41.792357                | -87.617931                | 41.812949                 | -87.617860                 | 2022-01-01 |
| 3   | 2022-01-01           | 243.0        | 1.0        | 5.0  | 0.0 | 2.36               | 7.36       | 0                      | 1            | 41.936310                | -87.651563                | 41.943155                 | -87.640698                 | 2022-01-01 |
| 4   | 2022-01-01           | 364.0        | 1.3        | 5.0  | 0.0 | 2.36               | 7.36       | 0                      | 1            | 41.921855                | -87.646211                | 41.936237                 | -87.656412                 | 2022-01-01 |

```python
# Select the column which you want to plot.
column_to_plot = 'date'

# Plot the histogram of the desired column
df.hist(column_to_plot, density=True)
```

> [!result]
> 	array([[<Axes: title={'center': 'date'}>]], dtype=object)
![Rideshare_Project_Week1_11_1.png|500](https://obsidian-image.wwtt.xyz/2025/12/Rideshare_Project_Week1_11_1.png)

What you have plotted above is the distribution of the rides throughout the year. Note the code above also set the `dentsity=True`. This is so that the histogram is scaled to look like a probability density function like the ones you saw on Lesson 2. This means scaling the plot so that the area of the bars equals 1. What does this distribution look like to you? Is it similar to any of the distributions you saw in the videos? 

%%
绘制的图像展示了 2022 年一整年的乘车数据的分布情况。注意代码中同样设置了 `dentsity=True`。这是为了缩放直方图，使其看起来像概率密度函数，就像您在第 2 课中看到的那样。这意味着图像缩放后柱状图的所有面积和为 1。这个看上去像什么样分布？它与你在视频中看到的任何分布相似吗？
%%

Although the distribution is slightly smaller for earlier dates, you could probably say that the rides are quite uniformly distributed throughout the year. Just note that this is not the actual distribution of the dates of cab rides, but rather an estimate based on the observations you have. Since this is real-world data, there are some fluctuations.

%%
尽管日期前期的分布略显稀疏，你可能认为整年的乘车数据呈现均匀分布。请注意，这并不是出租车出行日期的实际分布，是你通过手头这部分**观察样本**的估计。由于这是真实世界的数据，它会有一些波动。
%%

Now change the `column_to_plot` variable above to some other column name to observe the distributions of other variables. Some interesting ones might be `fare`, `tip` or `trip_length`. These variables can tell you how far drivers have to drive and how much they are getting paid for it.

%%
现在改变上面代码中 `column_to_plot` 的变量，使用一些其他的列绘图，来观察它的分布情况。有一些有意思的列比如 `fare`, `tip` 和 `trip_length`。这些变量会告诉你司机们开了多远，他们的收入如何。
%%

Lets look together at the `tip` column.

%%
让我们看看 `tip` （小费）列。
%%

```python
# Select the column which you want to plot.
column_to_plot = 'tip'

# Plot the histogram of the desired column
df.hist(column_to_plot, density=True, bins = 100);
```

> [!result]
> ![Rideshare_Project_Week1_13_0.png|500](https://obsidian-image.wwtt.xyz/2025/12/Rideshare_Project_Week1_13_0.png)


What can you say about the distribution of tips? This one looks a bit weird, right? What could explain this strange distribution? What do you think the large bar on the left corresponds to?

%%
对于小费的分布你有什么看法？这个分布看上去有点奇怪对吗？如何解释这个奇怪的分布？你认为这个大长条代表什么？
%%

What is actually happening here is that the majority of the people do not tip, and that's why you see a large bar at tip = 0. 

%%
事实上，大多数人都不会给小费，那就是为什么你看到了一个 tip = 0 的大长条。
%%

Based on the data, you can calculate the probability of the customer tipping. You can do this by simply calculating the proportion of customers that actually tipped from the total number of rides.

%%
基于这个数据，你可以计算顾客会给小费的可能性（概率）。只需计算实际给小费的顾客占所有乘车次数的比例即可。
%%
```python
# Create a boolean series that distinguishes between tippers and no-tippers
tippers = df['tip'] > 0
# Count the number of tippers
number_of_tippers = tippers.sum()
# Count the total number of rides
total_rides = len(df)

# Calculate the fraction of people who tip
fraction_of_tippers = number_of_tippers / total_rides
print(f'The percentage of riders who tip is {fraction_of_tippers*100:.0f}%.')
```

> [!result]
> 	The percentage of riders who tip is 25%.

In the next cell you will create a new dataframe, where you will remove the non-tippers (the ones who gave a tip of zero). Then you can replot the histogram and see how it looks without the large bar at tip = 0.

%%
在下一个单元格中你将会创建一个新的 dataframe，它将会移除没有付小费的人（小费为 0 的人）。然后重新绘制直方图，你可以看到 tip = 0 的大长条没有了。
%%

```python
# Create a dataframe That only consists of tippers (conditioned on the boolean series)
df_tippers = df[tippers]

# Now re-plot the above histogram, but only for tippers
df_tippers.hist('tip', density=True, bins = 100);
```

> [!result]
> ![Rideshare_Project_Week1_17_0.png|500](https://obsidian-image.wwtt.xyz/2025/12/Rideshare_Project_Week1_17_0.png)


You can see now that the distribution got a much more interesting shape. What you are actually doing here is conditioning the original variable `tip`. You are ploting the distribution of tips given that a tip was actually given, or given that `tip>0` if you want it in mathematical terms. In other words, you are discarding part of your data, where `tip=0`, and finding the distribution of the remaining data.

%%
你可以看到这个分布形成了一个更有趣的形状。你实际上在对原始变量 `tip` 做条件化处理。你绘制的实际是给予小费时的小费分布情况，或者用数学术语来说，这是 `tip > 0` 的分布。换句话说，你丢弃了 `tip = 0` 的数据并寻找剩余数据的分布情况。
%%
# 5. Split the Data Into Interesting Subsets

The next thing you can check is if you can create any subsets of data and have a look at conditional distributions over these subsets. For example, you might be interested, to know whether there are more rides on the weekend than during the week, or if people tip more on weekends. This can help you figure out whether there are any differences in demand during the week and helps you adjust the supply of drivers.

%%
接下来，你可以确认能否创建任何数据子集，并查看它的条件分布。比如，你可能想知道周末的乘客是否是一周中最多的，或者人们是否周末小费给得更多。这可以帮助你了解一周内的需求是否存在差异，然后帮助你调整司机与乘客的供给关系。
%%

For this you will first create a new column called `weekday`, where you will store the information on the day of the week.

%%
为了这个你首先要创建名为 `weekday` 的新列，并且把具体“星期几”存储在这个列中。
%%

```python
# Extracting the day of the week is simple when you have it in datetime format.
df['weekday'] = df["date"].dt.day_name()

df.head()
```

|     | trip_start_timestamp | trip_seconds | trip_miles | fare | tip | additional_charges | trip_total | shared_trip_authorized | trips_pooled | pickup_centroid_latitude | pickup_centroid_longitude | dropoff_centroid_latitude | dropoff_centroid_longitude | date       | weekday  |
| --- | -------------------- | ------------ | ---------- | ---- | --- | ------------------ | ---------- | ---------------------- | ------------ | ------------------------ | ------------------------- | ------------------------- | -------------------------- | ---------- | -------- |
| 0   | 2022-01-01           | 3905.0       | 44.5       | 55.0 | 0.0 | 11.25              | 66.25      | 0                      | 1            | 41.972563                | -87.678846                | NaN                       | NaN                        | 2022-01-01 | Saturday |
| 1   | 2022-01-01           | 2299.0       | 25.0       | 32.5 | 7.0 | 7.18               | 46.68      | 0                      | 1            | 41.878866                | -87.625192                | NaN                       | NaN                        | 2022-01-01 | Saturday |
| 2   | 2022-01-01           | 275.0        | 1.5        | 7.5  | 0.0 | 1.02               | 8.52       | 0                      | 1            | 41.792357                | -87.617931                | 41.812949                 | -87.617860                 | 2022-01-01 | Saturday |
| 3   | 2022-01-01           | 243.0        | 1.0        | 5.0  | 0.0 | 2.36               | 7.36       | 0                      | 1            | 41.936310                | -87.651563                | 41.943155                 | -87.640698                 | 2022-01-01 | Saturday |
| 4   | 2022-01-01           | 364.0        | 1.3        | 5.0  | 0.0 | 2.36               | 7.36       | 0                      | 1            | 41.921855                | -87.646211                | 41.936237                 | -87.656412                 | 2022-01-01 | Saturday |

Now you can count the number of riders on a given day of the week.
%%
现在你可以统计每周的每天分别有多少乘客。
%%

```python
# Count the number of rides each day
daily_ride_counts = df['weekday'].value_counts()

# List of weekdays. You will use it to reorder the counts, as they are in random order.
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Reorder the series given weekdays
daily_ride_counts = daily_ride_counts.reindex(WEEKDAYS)

daily_ride_counts
```

> [!result]
> 	weekday
> 	Monday        79013
> 	Tuesday       82576
> 	Wednesday     88034
> 	Thursday      95721
> 	Friday       115923
> 	Saturday     132872
> 	Sunday        96959
> 	Name: count, dtype: int64

And in the same manner, you will calculate the number of tippers on a given day of the week.

%%
以同样的方式，你将计算一周中每天付了小费的人数。
%%

```python
df_tippers = df[df['tip'] > 0]
# Count the number of tips given each day
daily_tippers_counts = df_tippers['weekday'].value_counts()

# Reorder the series given weekdays
daily_tippers_counts = daily_tippers_counts.reindex(WEEKDAYS)

daily_tippers_counts
```

> [!result]
> 	weekday
> 	Monday       19779
> 	Tuesday      20898
> 	Wednesday    22691
> 	Thursday     24210
> 	Friday       29256
> 	Saturday     33215
> 	Sunday       23294
> 	Name: count, dtype: int64

Now you can calculate the percentage of customers tipping on each day of the week.

%%
现在计算乘客一周内每天付小费人数的百分比
%%

```python
df_daily_aggregation = pd.concat([daily_ride_counts, daily_tippers_counts], axis=1, keys=['ride_count', 'tippers_count'])
df_daily_aggregation["tips_percentage"] = df_daily_aggregation['tippers_count'] / df_daily_aggregation['ride_count'] * 100

df_daily_aggregation
```

|           | ride_count | tippers_count | tips_percentage |
| --------- | ---------- | ------------- | --------------- |
| weekday   |            |               |                 |
| Monday    | 79013      | 19779         | 25.032590       |
| Tuesday   | 82576      | 20898         | 25.307595       |
| Wednesday | 88034      | 22691         | 25.775269       |
| Thursday  | 95721      | 24210         | 25.292256       |
| Friday    | 115923     | 29256         | 25.237442       |
| Saturday  | 132872     | 33215         | 24.997742       |
| Sunday    | 96959      | 23294         | 24.024588       |

What you have just calculated are conditional probabilities: What is the probability of someone tipping, given a certain day of the week? Or if you write it with an equation: $P(tip|weekday)$. 

%%
你刚刚计算的是条件概率：在某周特定的一天，某人给小费的概率是多少？你可以写下它的等式 $P(tip|weekday)$。
%%

Now you can have another look at the numbers and see if there are some important insights!

%%
现在你可以再看看这些数字，看看是否有一些重要的见解！
%%

You can see that there are significantly more rides on Fridays and Saturdays than on the other days of the week, however the percentage of the tippers does not change much.

%%
可以看出，周五和周六的乘车次数明显多于一周中的其他日子，然而小费支付者的比例变化不大。
%%

You can use the cell below to save your modified dataframe. You dont need to do that, as the dataframe for the next lab is already provided.

%%
你可以使用下面的单元格保存你修改后的 dataframe。虽然没有必要这么做，我们已经在下一个实验中准备好了这个 dataframe。
%%

```python
# Uncomment the line below if you want to save your dataframe.
# df.to_csv("data/rideshare_2022_user.csv", index=False)
```

**Congratulations on finishing this lab.** You have used the implementation of quite a few concepts covered in this course: probabilities, distributions and conditional probabilities. On top of that you have practiced Pandas a little bit. If you liked this exercise, look out for another similar notebook next week!

%%
恭喜完成了这个实验。您已经使用了本课程中涉及的许多概念的实现：概率，分布和条件概率。此外，你还稍微练习了一点 Pandas。如果你喜欢这个实践，请期待下周的其他的 Notebook。
%%