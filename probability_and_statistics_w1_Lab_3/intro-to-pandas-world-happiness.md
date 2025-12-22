---
aliases: [Exploratory Data Analysis - Intro to Pandas]
tags: []
created: 2025-12-20, 16:21:20
modified: 2025-12-20, 22:18:17
---

# Exploratory Data Analysis - Intro to Pandas

本文全文使用 `Qwen3-Coder-30B-A3B-Instruct` 翻译并人工润色。

Welcome to the Pandas tutorial lab. This is the first notebook of the exploratory data analysis (EDA) series, where you will get your hands dirty applying the skills you have learned in the course on an actual data problem, similar to those you might encouter in real life! Here you will see and try out some basics of Pandas and get familiar with some of the useful functions that you will use across the other labs and assignments. If you already know Pandas well, feel free to skip this notebook.

%%
欢迎来到 Pandas 教程实验室。这是探索性数据分析(EDA)系列的第一个 Notebook，在这里你将运用在课程中学到的技能来解决实际的数据问题，就像你在现实生活中可能遇到的那样！在这里你将看到并尝试一些 Pandas 的基础知识，并熟悉在其他实验和作业中会用到的一些有用函数。如果你已经很熟悉Pandas，可以跳过这个 Notebook 。
%%

For the demonstration purposes you will use the [World Happiness Report](https://worldhappiness.report/) dataset. The dataset consists of 2199 rows, where each row contains various hapiness-related metrics for a certain country in a given year. Right now you'll just use this dataset to understand some fundamental operations in Pandas. You will see this dataset again later in week 3, where you will dig deeper into the data and explore relationships to better understand which factors seem to best predict happiness.

%%
为了演示目的，您将使用世界幸福报告数据集。该数据集包含2199行，每行包含特定国家在给定年份的各種幸福相关指标。现在您将只使用这个数据集来理解 Pandas 中的一些基本操作。您将在第3周再次看到这个数据集，在那里您将深入研究数据并探索关系，以更好地了解哪些因素似乎最能预测幸福。
%%

This notebook is not a comprehensive guide to Pandas, but rather shows and explains the functions you will use through this course. For a more comprehensive guide on Pandas, please see the [official tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) or check the documentation.

%%
这个 Notebook 不是 Pandas 的全面指南，而是展示了并通过本课程说明你将使用的函数。有关 Pandas 的更全面的指南，请参阅官方教程或查看文档。
%%
# 1. Importing the Libraries

The most important library you will need in this notebook is - you guessed it - `Pandas`. You will also use the `Seaborn` library for plotting the data. To import the libraries run the cell below.

%%
在这个 Notebook 中，你需要的最重要的库是——你猜到了—— Pandas。你还将使用 Seaborn 库来绘制数据。要导入这些库，请运行下面的单元格。
%%

```python
# Import the Pandas library
import pandas as pd
# Import the Seaborn library for plotting
#!pip install seaborn
import seaborn as sns
```

# 2. Importing the Data

Now that you have the pandas library imported, you'll need to load your dataset. The dataset you will use is saved as a `.csv` file and all you need to do to load is call the function `pd.read_csv(filename)`. If you have your data in another format, there exists a variety of functions to load it, you can check the documentation [here](https://pandas.pydata.org/pandas-docs/stable/reference/io.html).

%%
现在你已经导入了 pandas 库，需要加载你的数据集。你将要读取的数据集为 `.csv` 文件，只需调用函数 `pd.read_csv(filename)` 即可加载。如果你的数据采用其他格式，也有另外的函数可以加载它，你可以在此查看文档。
%%

When you load the dataset, it will be stored as a `DataFrame` type (see the documentation [here](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html)). This is the most commonly used Pandas datastructure that you will use throughout this and other notebooks.

%%
当你加载了数据集后，它将被存储为 `DataFrame` 类型。这是 Pandas 中最常用的数据结构，你将在 Notebook 中频繁的使用它。
%%

```python
# Load the dataset and save it to the df variable
df = pd.read_csv('data/world_happiness.csv')
```

# 3. Basic Operations With a Dataframe

## 3.1 View the Dataframe
You can use `DataFrame.head()` and `DataFrame.tail()` to view the first or last rows of the frame respectively. By default it will show you five rows, but you can specify the number of rows you want to see as a parameter. Technically, neither of the functions actually display anything, but just return a new dataframe. The dataframe is displayed because Jupyter notebooks show the output of the last row in the cell. You can also display the contents of your dataframe by simply writing `df`. If your dataframe is too long, it will then display only the first and the last few rows.

%%
你可以使用 `DataFrame.head()` 和 `DataFrame.tail()` 分别查看 DataFrame 的前几行或后几行。默认情况下，它会显示五行，你可以将想要显示的行数作为参数指定。从技术上讲，这两个函数实际上并不显示任何内容，只是返回一个新的 DataFrame 。DataFrame 被显示出来是因为 Jupyter Notebook 会显示单元格最后一行的输出。你也可以通过简单地编写 `df` 来显示数据框的内容。如果数据框太长，它将只显示前几行和后几行。
%%

Note that all of this only works if you use it in the last line of code in the cell, because the cells automatically display the output of the last line. If you want to see more than one dataframe by running a single cell or if you want to perform some other tasks after displaying the dataframe, then you better encapsulate it with `print()` or `display()`. `display()` function will print the dataframe, but with the same format as just calling `df`, whereas `print()` will print as plain text. 

%%
请注意，所有这些操作只有在单元格的最后一行代码中使用时才有效，因为单元格会自动显示最后一行的输出。如果你想在运行单个单元格后查看多个数据框，或者在显示数据框后想要执行其他任务，那么最好将其用 `print()` 或 `display()` 包装起来。`display()` 函数会以与直接调用 `df` 相同的格式打印数据框，而 `print()` 会以纯文本格式打印。
%%

Try commenting and uncommenting lines below, to see how this plays out. Try different combiations of rows.

%%
尝试注释和取消注释下面的行，看看它是如何运作的。尝试行的不同组合。
%%

```python
# This line will display the first few rows of the dataframe if there are no lines of code after.
df.head()

# Try uncommenting different combinations of the lines below.
# print("Cats are cool.")
# print(df.head())
# print(df)
# print("Some more text about cats being cool.")
# display(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>year</th>
      <th>Life Ladder</th>
      <th>Log GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy at birth</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Positive affect</th>
      <th>Negative affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>3.724</td>
      <td>7.350</td>
      <td>0.451</td>
      <td>50.5</td>
      <td>0.718</td>
      <td>0.168</td>
      <td>0.882</td>
      <td>0.414</td>
      <td>0.258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>4.402</td>
      <td>7.509</td>
      <td>0.552</td>
      <td>50.8</td>
      <td>0.679</td>
      <td>0.191</td>
      <td>0.850</td>
      <td>0.481</td>
      <td>0.237</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.614</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.121</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.164</td>
      <td>0.731</td>
      <td>0.480</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
    </tr>
  </tbody>
</table>
</div>



Now display the last few rows of the dataframe. Pay attention to the additional parameter that specifies the number of rows.

%%
现在显示数据框的最后几行。注意指定行数的附加参数。
%%

```python
# This line will display only the last two rows of the dataframe.
df.tail(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>year</th>
      <th>Life Ladder</th>
      <th>Log GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy at birth</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Positive affect</th>
      <th>Negative affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2197</th>
      <td>Zimbabwe</td>
      <td>2021</td>
      <td>3.155</td>
      <td>7.657</td>
      <td>0.685</td>
      <td>54.050</td>
      <td>0.668</td>
      <td>-0.076</td>
      <td>0.757</td>
      <td>0.610</td>
      <td>0.242</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>Zimbabwe</td>
      <td>2022</td>
      <td>3.296</td>
      <td>7.670</td>
      <td>0.666</td>
      <td>54.525</td>
      <td>0.652</td>
      <td>-0.070</td>
      <td>0.753</td>
      <td>0.641</td>
      <td>0.191</td>
    </tr>
  </tbody>
</table>
</div>



## 3.2 Index and Column Names

In the `DataFrame`, the data is stored in a two dimensional grid (rows and columns). The rows are indexed and the columns are named. To see the index or the column names, you can use `DataFrame.index` or `DataFrame.columns` respectively.

%%
在DataFrame中，数据存储在一个二维网格（行和列）中。行有索引，列有名称。要查看索引或列名，可以分别使用DataFrame.index或DataFrame.columns。
%%

```python
df.index
```

> [!result]
>     RangeIndex(start=0, stop=2199, step=1)

As you can see, the index is a range of numbers between 0 (inclusive) and 2199 (not inclusive).

%%
如你所见，索引是介于0（包含）和2199（不包含）之间的一组数字。
%%

Run the cell below to see the column names.


```python
df.columns
```

> [!result]
>     Index(['Country name', 'year', 'Life Ladder', 'Log GDP per capita',
>            'Social support', 'Healthy life expectancy at birth',
>            'Freedom to make life choices', 'Generosity',
>            'Perceptions of corruption', 'Positive affect', 'Negative affect'],
>            dtype='object')



The column names are saved as strings. As you can see, they can include spaces. This can lead to difficulties when accessing the columns (you will see this very soon), so it is a good idea to rename them to get rid of the spaces. A common practice is to replace them with underscores. To rename the columns, you can use `DataFrame.rename()` and pass the columns you want to rename in a dictionary.

%%
列名被保存为字符串。如你所见，列名可以包含空格。这在访问列时可能会出现问题（你很快就会看到），因此重新命名列以去掉空格是个好主意。一种常见的做法是用下划线替换空格。要重命名列，你可以使用 `DataFrame.rename()` 并传入一个字典来指定要重命名的列。
%%

In the next example, you will see how you can automatically replace all spaces with underscores

%%
在下一个示例中，您将看到如何自动将所有空格替换为下划线
%%

```python
# A dictionary mapping old column names to new column names. In addition to replacing spaces
# with underscores, you will make all of the text lowercase.
columns_to_rename = {i: "_".join(i.split(" ")).lower() for i in df.columns}
# Note that this dictionary is created automatically from the column names.
# You can also create it by hand and rename only the columns you want to rename
# For example, see the commented line below:
# columns_to_rename = {"Country name": "country_name", "Life Ladder": "life_ladder"}

# Rename the columns
df = df.rename(columns=columns_to_rename)
# Display the new dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>3.724</td>
      <td>7.350</td>
      <td>0.451</td>
      <td>50.5</td>
      <td>0.718</td>
      <td>0.168</td>
      <td>0.882</td>
      <td>0.414</td>
      <td>0.258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>4.402</td>
      <td>7.509</td>
      <td>0.552</td>
      <td>50.8</td>
      <td>0.679</td>
      <td>0.191</td>
      <td>0.850</td>
      <td>0.481</td>
      <td>0.237</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.614</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.121</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.164</td>
      <td>0.731</td>
      <td>0.480</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
    </tr>
  </tbody>
</table>
</div>


## 3.3 Data Types

One cool thing about the DataFrame type is that the columns of the resulting DataFrame can have different `dtypes`. This is something you simply can not do with a Numpy array. You can look at them and if needed to you can change them.

%%
关于 DataFrame 类型的一个酷炫之处在于，生成的 DataFrame 的列可以具有不同的数据类型。这是你无法用 NumPy 数组做到的。你可以查看它们，如果需要的话你还可以修改它们。
%%

```python
df.dtypes
```

> [!result]
>     country_name                         object
>     year                                  int64
>     life_ladder                         float64
>     log_gdp_per_capita                  float64
>     social_support                      float64
>     healthy_life_expectancy_at_birth    float64
>     freedom_to_make_life_choices        float64
>     generosity                          float64
>     perceptions_of_corruption           float64
>     positive_affect                     float64
>     negative_affect                     float64
>     dtype: object



You can see that the columns above are of different types and if you compare it to how the data actually looks like, it seems that the types are correct. Sometimes if your data is incorrectly formatted, the imported types will be wrong. In this case you will want to change the types of the columns manually before proceeding. Check the code below on how you can do that. Note that nothing will change after running the code below, as the data is already of correct types.

%%
你可以看到上面的列是不同类型的，如果你将其与数据实际的外观进行比较，似乎类型是正确的。有时如果数据格式不正确，导入的类型也会错误。在这种情况下，你可能需要在继续之前手动更改列的类型。查看下面的代码，了解如何做到这一点。请注意，运行下面的代码后不会有任何改变，因为数据的类型已经是正确的。
%%

```python
# List all of the columns that should be floats
float_columns = [i for i in df.columns if i not in  ["country_name", "year"]]
# Change the type of all float columns to float
df = df.astype({i: float for i in float_columns})
# Show the types of all columns
df.dtypes
```

> [!result]
>     country_name                         object
>     year                                  int64
>     life_ladder                         float64
>     log_gdp_per_capita                  float64
>     social_support                      float64
>     healthy_life_expectancy_at_birth    float64
>     freedom_to_make_life_choices        float64
>     generosity                          float64
>     perceptions_of_corruption           float64
>     positive_affect                     float64
>     negative_affect                     float64
>     dtype: object


The `df.info()` provides some additional information. In addition to data types it also tells you the number of non-null values per column.

%%
df.info() 提供了一些额外的信息。除了数据类型之外，它还告诉你每列的非空值数量。
%%

```python
df.info()
```

>[!result]
>     <class 'pandas.core.frame.DataFrame'>
>     RangeIndex: 2199 entries, 0 to 2198
>     Data columns (total 11 columns):
>      #   Column                            Non-Null Count  Dtype  
>     ---  ------                            --------------  -----  
>      0   country_name                      2199 non-null   object 
>      1   year                              2199 non-null   int64  
>      2   life_ladder                       2199 non-null   float64
>      3   log_gdp_per_capita                2179 non-null   float64
>      4   social_support                    2186 non-null   float64
>      5   healthy_life_expectancy_at_birth  2145 non-null   float64
>      6   freedom_to_make_life_choices      2166 non-null   float64
>      7   generosity                        2126 non-null   float64
>      8   perceptions_of_corruption         2083 non-null   float64
>      9   positive_affect                   2175 non-null   float64
>      10  negative_affect                   2183 non-null   float64
>     dtypes: float64(9), int64(1), object(1)
>     memory usage: 189.1+ KB


## 3.4 Selecting Columns

One way of selecting a single column is to use `DataFrame.column_name`. Here you can see why it was a good idea that you renamed the columns to not include any whitespaces. This returns a Pandas `Series`, which is a different datatype from a `DataFrame`. You will see how to return a `DataFrame` a bit later.
%%
选择单列的一种方法是使用 `DataFrame.column_name` 。在这里你可以看到，将列重命名为不包含任何空格是一个好主意。这会返回一个Pandas `Series` ，它与 `DataFrame` 的数据类型不同。稍后你将看到如何返回 `DataFrame`。
%%

```python
# Select the life_ladder column and store it in x
x = df.life_ladder

print(f"type(x):\n {type(x)}\n")
print(f"x:\n{x}")
```

>[!result]
>     type(x):
>      <class 'pandas.core.series.Series'>
>     
>     x:
>     0       3.724
>     1       4.402
>     2       4.758
>     3       3.832
>     4       3.783
>             ...  
>     2194    3.616
>     2195    2.694
>     2196    3.160
>     2197    3.155
>     2198    3.296
>     Name: life_ladder, Length: 2199, dtype: float64


Another way to do this is to use square brackets and the name of the column in quortes, much as you would do when accessing an entry in a dictionary. As with dictionaries, you can use double quotes or simple quotes. 

%%
另一种方法是使用方括号和用引号括起来的列名，就像你在字典中查找条目时所做的一样。和字典一样，你可以使用双引号或单引号。
%%

```python
x = df["life_ladder"]

print(f"type(x):\n {type(x)}\n")
print(f"x:\n{x}")
```

>[!result]
>     type(x):
>      <class 'pandas.core.series.Series'>
>     
>     x:
>     0       3.724
>     1       4.402
>     2       4.758
>     3       3.832
>     4       3.783
>             ...  
>     2194    3.616
>     2195    2.694
>     2196    3.160
>     2197    3.155
>     2198    3.296
>     Name: life_ladder, Length: 2199, dtype: float64


Passing a list of labels rather than a single label selects the columns and returns a DataFrame (rather than a Series), with only the selected columns. You can use it to select one or more columns.

%%
传递标签列表而不是单个标签来选择列并返回 DataFrame（而不是Series），只包含选定的列。您可以使用它来选择一个或多个列。
%%

```python
x = df[["life_ladder"]]
# x = df[["life_ladder", "year"]]

print(f"type(x):\n {type(x)}\n")
print(f"x:\n{x}")
```
>[!result]
>     type(x):
>      <class 'pandas.core.frame.DataFrame'>
>     
>     x:
>           life_ladder
>     0           3.724
>     1           4.402
>     2           4.758
>     3           3.832
>     4           3.783
>     ...           ...
>     2194        3.616
>     2195        2.694
>     2196        3.160
>     2197        3.155
>     2198        3.296
>     
>     [2199 rows x 1 columns]


## 3.5 Selecting Rows

Passing a slice `:` selects matching rows and returns a DataFrame with all columns in your original dataframe.

%%
传递切片：选择匹配的行并返回一个 DataFrame，其中包含原始 DataFrame 中的所有列。
%%

```python
df[2:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.614</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.121</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.164</td>
      <td>0.731</td>
      <td>0.480</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
    </tr>
  </tbody>
</table>
</div>



## 3.6 Iterating Over Rows

If you want to iterate over the rows, you can use the `.iterrows()` method. For each row it yields a (index, row) tuple, where the row is a `Series` object containing the data. Note that this does not preserve the data types (dtypes) across the rows (dtypes are preserved across columns for DataFrames).

%%
如果要遍历行，可以使用 .iterrows() 方法。对于每一行，它会生成一个 (index, row) 元组，其中 row 是一个包含数据的 Series 对象。注意，这样不会保留行之间的数据类型（dtypes）（但对于 DataFrame，列之间的数据类型是保留的）。
%%

```python
index, row = next(df.iterrows())
row
```



>[!result]
>     country_name                        Afghanistan
>     year                                       2008
>     life_ladder                               3.724
>     log_gdp_per_capita                         7.35
>     social_support                            0.451
>     healthy_life_expectancy_at_birth           50.5
>     freedom_to_make_life_choices              0.718
>     generosity                                0.168
>     perceptions_of_corruption                 0.882
>     positive_affect                           0.414
>     negative_affect                           0.258
>     Name: 0, dtype: object



## 3.7 Boolean Indexing

Now to the more fun part. If you looked carefully at the dataset that was displayed above, you probably saw that the datapoints are available for different years. What if you are interested only in data from a certain year? Or from a certain country? Or perhaps where a value in a certain column is greater than some predetermined value? You can use boolean indexing.

%%
现在来看更有趣的部分。如果你仔细观察了上面显示的数据集，你可能已经注意到数据点是按不同年份提供的。如果你只对某个特定年份的数据感兴趣呢？或者只对某个特定国家的数据感兴趣呢？又或者只对某个特定列中的值大于某个预定义值的数据感兴趣呢？你可以使用布尔索引。
%%

Run the cell below to select rows where the year equals to 2022. Try to uncomment some other row to see what it does.

%%
运行下面的单元格以选择年份等于2022的行。尝试取消注释其他行来看看它的作用。
%%

```python
df[df["year"] == 2022]
# df[df["life_ladder"] > 5] # Select rows where life_ladder > 5
# df[df["life_ladder"] > 11] # This one should return an empty dataframe
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>Afghanistan</td>
      <td>2022</td>
      <td>1.281</td>
      <td>NaN</td>
      <td>0.228</td>
      <td>54.875</td>
      <td>0.368</td>
      <td>NaN</td>
      <td>0.733</td>
      <td>0.206</td>
      <td>0.576</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Albania</td>
      <td>2022</td>
      <td>5.212</td>
      <td>9.626</td>
      <td>0.724</td>
      <td>69.175</td>
      <td>0.802</td>
      <td>-0.066</td>
      <td>0.846</td>
      <td>0.547</td>
      <td>0.255</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Argentina</td>
      <td>2022</td>
      <td>6.261</td>
      <td>10.011</td>
      <td>0.893</td>
      <td>67.250</td>
      <td>0.825</td>
      <td>-0.128</td>
      <td>0.810</td>
      <td>0.724</td>
      <td>0.284</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Armenia</td>
      <td>2022</td>
      <td>5.382</td>
      <td>9.668</td>
      <td>0.811</td>
      <td>67.925</td>
      <td>0.790</td>
      <td>-0.154</td>
      <td>0.705</td>
      <td>0.531</td>
      <td>0.549</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Australia</td>
      <td>2022</td>
      <td>7.035</td>
      <td>10.854</td>
      <td>0.942</td>
      <td>71.125</td>
      <td>0.854</td>
      <td>0.153</td>
      <td>0.545</td>
      <td>0.711</td>
      <td>0.244</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2104</th>
      <td>Uruguay</td>
      <td>2022</td>
      <td>6.671</td>
      <td>10.084</td>
      <td>0.905</td>
      <td>67.500</td>
      <td>0.878</td>
      <td>-0.052</td>
      <td>0.631</td>
      <td>0.775</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>2120</th>
      <td>Uzbekistan</td>
      <td>2022</td>
      <td>6.016</td>
      <td>8.990</td>
      <td>0.879</td>
      <td>65.600</td>
      <td>0.959</td>
      <td>0.309</td>
      <td>0.616</td>
      <td>0.741</td>
      <td>0.225</td>
    </tr>
    <tr>
      <th>2137</th>
      <td>Venezuela</td>
      <td>2022</td>
      <td>5.949</td>
      <td>NaN</td>
      <td>0.899</td>
      <td>63.875</td>
      <td>0.770</td>
      <td>NaN</td>
      <td>0.798</td>
      <td>0.754</td>
      <td>0.292</td>
    </tr>
    <tr>
      <th>2154</th>
      <td>Vietnam</td>
      <td>2022</td>
      <td>6.267</td>
      <td>9.333</td>
      <td>0.879</td>
      <td>65.600</td>
      <td>0.975</td>
      <td>-0.179</td>
      <td>0.703</td>
      <td>0.774</td>
      <td>0.108</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>Zimbabwe</td>
      <td>2022</td>
      <td>3.296</td>
      <td>7.670</td>
      <td>0.666</td>
      <td>54.525</td>
      <td>0.652</td>
      <td>-0.070</td>
      <td>0.753</td>
      <td>0.641</td>
      <td>0.191</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 11 columns</p>
</div>


Note that now that you selected only the certain rows, the index column does not make much sense anymore because you have a lot of gaps. While this is not a problem, in some cases you might want the index to correspond to the actual row number. To achieve this you can use `reset_inex()`. In other cases you might want to keep the index as it is to more easily refer back to the original dataframe. It all depends on the context of your project. Run the cell below to reset the index and take a look at the output.

%%
请注意，现在你只选择了某些行，索引列就不再有意义了，因为你会有很多空隙。虽然这不是问题，但在某些情况下，你可能希望索引对应实际的行号。要实现这一点，可以使用 `reset_index()`。在其他情况下，你可能希望保持索引不变，以便更容易回顾原始数据框。这完全取决于你项目的上下文。运行下面的单元格来重置索引并查看输出。
%%


```python
new_df = df[df["year"] == 2022]
new_df = new_df.reset_index(drop=True)
new_df
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2022</td>
      <td>1.281</td>
      <td>NaN</td>
      <td>0.228</td>
      <td>54.875</td>
      <td>0.368</td>
      <td>NaN</td>
      <td>0.733</td>
      <td>0.206</td>
      <td>0.576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>2022</td>
      <td>5.212</td>
      <td>9.626</td>
      <td>0.724</td>
      <td>69.175</td>
      <td>0.802</td>
      <td>-0.066</td>
      <td>0.846</td>
      <td>0.547</td>
      <td>0.255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Argentina</td>
      <td>2022</td>
      <td>6.261</td>
      <td>10.011</td>
      <td>0.893</td>
      <td>67.250</td>
      <td>0.825</td>
      <td>-0.128</td>
      <td>0.810</td>
      <td>0.724</td>
      <td>0.284</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Armenia</td>
      <td>2022</td>
      <td>5.382</td>
      <td>9.668</td>
      <td>0.811</td>
      <td>67.925</td>
      <td>0.790</td>
      <td>-0.154</td>
      <td>0.705</td>
      <td>0.531</td>
      <td>0.549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Australia</td>
      <td>2022</td>
      <td>7.035</td>
      <td>10.854</td>
      <td>0.942</td>
      <td>71.125</td>
      <td>0.854</td>
      <td>0.153</td>
      <td>0.545</td>
      <td>0.711</td>
      <td>0.244</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Uruguay</td>
      <td>2022</td>
      <td>6.671</td>
      <td>10.084</td>
      <td>0.905</td>
      <td>67.500</td>
      <td>0.878</td>
      <td>-0.052</td>
      <td>0.631</td>
      <td>0.775</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Uzbekistan</td>
      <td>2022</td>
      <td>6.016</td>
      <td>8.990</td>
      <td>0.879</td>
      <td>65.600</td>
      <td>0.959</td>
      <td>0.309</td>
      <td>0.616</td>
      <td>0.741</td>
      <td>0.225</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Venezuela</td>
      <td>2022</td>
      <td>5.949</td>
      <td>NaN</td>
      <td>0.899</td>
      <td>63.875</td>
      <td>0.770</td>
      <td>NaN</td>
      <td>0.798</td>
      <td>0.754</td>
      <td>0.292</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Vietnam</td>
      <td>2022</td>
      <td>6.267</td>
      <td>9.333</td>
      <td>0.879</td>
      <td>65.600</td>
      <td>0.975</td>
      <td>-0.179</td>
      <td>0.703</td>
      <td>0.774</td>
      <td>0.108</td>
    </tr>
    <tr>
      <th>113</th>
      <td>Zimbabwe</td>
      <td>2022</td>
      <td>3.296</td>
      <td>7.670</td>
      <td>0.666</td>
      <td>54.525</td>
      <td>0.652</td>
      <td>-0.070</td>
      <td>0.753</td>
      <td>0.641</td>
      <td>0.191</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 11 columns</p>
</div>



# 4. Summary Statistics

Later in this course you will learn about summary statistics. For now, this is just to show you that Pandas allows for a very simple way to calculate all sorts of statistics using `describe()`. Run the cell below to see a quick statistical summary of your data. It doesn't matter if you don't know what each row means, you will learn all about it in the coming weeks.

%%
在本课程的后面，您将学习汇总统计。现在，这只是向您展示Pandas提供了一种非常简单的方式来计算各种统计信息，方法是使用describe()。运行下面的单元格，查看您的数据的快速统计摘要。如果不知道每一行的含义也没关系，您将在接下来的几周内学习所有相关内容。
%%


```python
df.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2199.000000</td>
      <td>2199.000000</td>
      <td>2179.000000</td>
      <td>2186.000000</td>
      <td>2145.000000</td>
      <td>2166.000000</td>
      <td>2126.000000</td>
      <td>2083.000000</td>
      <td>2175.000000</td>
      <td>2183.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.161437</td>
      <td>5.479227</td>
      <td>9.389760</td>
      <td>0.810681</td>
      <td>63.294582</td>
      <td>0.747847</td>
      <td>0.000091</td>
      <td>0.745208</td>
      <td>0.652148</td>
      <td>0.271493</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.718736</td>
      <td>1.125527</td>
      <td>1.153402</td>
      <td>0.120953</td>
      <td>6.901104</td>
      <td>0.140137</td>
      <td>0.161079</td>
      <td>0.185835</td>
      <td>0.105913</td>
      <td>0.086872</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2005.000000</td>
      <td>1.281000</td>
      <td>5.527000</td>
      <td>0.228000</td>
      <td>6.720000</td>
      <td>0.258000</td>
      <td>-0.338000</td>
      <td>0.035000</td>
      <td>0.179000</td>
      <td>0.083000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2010.000000</td>
      <td>4.647000</td>
      <td>8.500000</td>
      <td>0.747000</td>
      <td>59.120000</td>
      <td>0.656250</td>
      <td>-0.112000</td>
      <td>0.688000</td>
      <td>0.572000</td>
      <td>0.208000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014.000000</td>
      <td>5.432000</td>
      <td>9.499000</td>
      <td>0.836000</td>
      <td>65.050000</td>
      <td>0.770000</td>
      <td>-0.023000</td>
      <td>0.800000</td>
      <td>0.663000</td>
      <td>0.261000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2018.000000</td>
      <td>6.309500</td>
      <td>10.373500</td>
      <td>0.905000</td>
      <td>68.500000</td>
      <td>0.859000</td>
      <td>0.092000</td>
      <td>0.869000</td>
      <td>0.738000</td>
      <td>0.323000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2022.000000</td>
      <td>8.019000</td>
      <td>11.664000</td>
      <td>0.987000</td>
      <td>74.475000</td>
      <td>0.985000</td>
      <td>0.703000</td>
      <td>0.983000</td>
      <td>0.884000</td>
      <td>0.705000</td>
    </tr>
  </tbody>
</table>
</div>


Not all of the summary statistics always make sense. In your case, for example, you are looking at the summary statistics across various columns. But are you sure you know what the final numbers actually mean? You have data for many different countries, but are you sure that you have the same amount of datapoints for each country or for each year? Also the countries can have vastly different populations, is it fair to just average the numbers out?

%%
并非所有汇总统计都总是有意义的。例如，在这个例子中，您正在查看各个列的汇总统计。但您确定知道这些最终数字的实际含义吗？您有众多不同国家的数据，但您确定每个国家或每年的数据点数量相同吗？此外，各国的人口可能天差地别，简单地对数字求平均值是否公平？
%%
# 5. Plotting
If you want to plot the data, you can use `DataFrame.plot()`. By default it uses the index as the x axis and plots all the numeric columns as y axes. Run the cell below to see the output for your dataframe.

%%
如果你想绘制数据，可以使用 `DataFrame.plot()`。默认情况下，它使用索引作为 x 轴，并将所有数值列作为 y 轴进行绘制。运行下面的单元格以查看你的数据框的输出结果。
%%

```python
# If the plot doesn’t render, first try re-running this cell. If that doesn’t work, 
# you can restart the kernel (from the Kernel menu above) and try running the notebook again
df.plot()
```

> [!result]
>     <Axes: >
![intro-to-pandas-world-happiness_38_1.png](https://obsidian-image.wwtt.xyz/2025/12/intro-to-pandas-world-happiness_38_1.png)


As you can see, in this case the plot is not very useful. The index does not have any specific meaning, and the values of various columns differ greatly (years are all around 2000, but the values in the other columns are much lower) and thus you cannot see much in the plot. Try setting some parameters of the `.plot()` method to see what it allows you to do. You can find the documentation [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html).

%%
如你所见，在这种情况下，图表并不是很有用。索引没有任何特定含义，而且各列的数值差异很大（年份都在2000年左右，但其他列的数值要低得多），因此在图表中你看不出太多东西。尝试设置.plot()方法的一些参数，看看它允许你做什么。你可以在这里找到文档。
%%

Run the cell below to see a scatter plot with specifically chosen x and y variables. On the x axis there is logarithm of the GDP (measuring the wealth) while on the y axis there is the life ladder. This column contains values which are an estimate of self-assessed life quality on a scale of 1 to 10 as given by a survey among the people.

%%
运行下面的单元格查看散点图，其中特别选择了 x 和 y 变量。x 轴上是 GDP 的对数（衡量财富），y 轴上是生活阶梯。该列包含调查中人们对1至10尺度上自我评估生活质量的估计值。
%%

```python
df.plot(kind='scatter', x='log_gdp_per_capita', y='life_ladder')
```

> [!result]
>     <Axes: xlabel='log_gdp_per_capita', ylabel='life_ladder'>
![intro-to-pandas-world-happiness_40_1.png](https://obsidian-image.wwtt.xyz/2025/12/intro-to-pandas-world-happiness_40_1.png)


You can see that there is some sort of trend between the wealth of the country and the happiness of the population and you can say that it looks like that wealthier people are to some extent happier. In week three, you will explore this kind of relationship further. 

%%
你可以看到国家财富与人口幸福感之间存在某种趋势，可以说财富越充裕的人在某种程度上越幸福。在第三周，你将进一步探讨这种关系。
%%

Sometimes it is very insightful to separate the points by colors to highlight different characteristics or some points you are most interested in. Take a look at the example below

%%
有时通过颜色区分点可以非常有洞察力，以突出不同的特征或你最感兴趣的一些点。请看下面的例子
%%


```python
# Create a dictionary to map the country names to colors
cmap = {
    'Brazil': 'Green',
    'Slovenia': 'Orange',
    'India': 'purple'
}

df.plot(
    kind='scatter',
    x='log_gdp_per_capita',
    y='life_ladder',
    c=[cmap.get(c, 'yellow') for c in df.country_name], # Set the colors
    s=2 # Set the size of the points
    )
```

> [!result]
>     <Axes: xlabel='log_gdp_per_capita', ylabel='life_ladder'>
![intro-to-pandas-world-happiness_42_1.png](https://obsidian-image.wwtt.xyz/2025/12/intro-to-pandas-world-happiness_42_1.png)



You can see that even though in general higher GDP means higher value on the life ladder, this is not an universal truth. Comparing Slovenia (orange) with Brazil (green), you can see that people in Brazil earn less, but are on average happier than Slovenians through the years.

%%
可以看到，尽管一般来说GDP越高，生活阶梯上的价值也越高，但这并不是一个普遍真理。比较斯洛文尼亚（橙色）和巴西（绿色），可以看出巴西人的收入较低，但多年来平均来说比斯洛文尼亚人更幸福。
%%

Another very useful task you can do with plots is to visulize the distribution of your data. You will learn how to do this in more detail later, but for example you can easily plot a histogram using Pandas. Ise `DataFrame.hist()` on the dataframe you want to plot. Note that if you have many columns in the dataframe, this command will plot a histogram for each of the columns. You can select a single column from the dataframe if you only want to plot that one.

%%
你可以用图表做的另一个非常有用的任务是可视化数据的分布。你将会在后面更详细地学习如何做到这一点，但例如你可以很容易地使用 Pandas 绘制直方图。在你想绘制的数据框上使用 `DataFrame.hist()`。请注意，如果你的数据框有很多列，此命令将为每一列绘制一个直方图。如果你只想绘制某一列，可以从数据框中选择单列。
%%

```python
df.hist("life_ladder")
```

> [!result]
>     array([[<Axes: title={'center': 'life_ladder'}>]], dtype=object)
![intro-to-pandas-world-happiness_44_1.png](https://obsidian-image.wwtt.xyz/2025/12/intro-to-pandas-world-happiness_44_1.png)
 


What you see in this histogram is a distribution of values in the "life_ladder" column. What do you think about this distribution on the first glance? Are the people generally happy about their quality of life? Note that to answer this question properly, you need to dig a bit deeper into the data: understand where each value comes from, as the values are not single datapoints (single answers by people), but already aggregated values across countries and at various points in time.

%%
你在这个直方图中看到的是"life_ladder"列中的值的分布。你第一眼对这个分布有什么看法？人们对自己的生活质量总体上感到幸福吗？请注意，要正确回答这个问题，你需要更深入地研究数据：了解每个值的来源，因为这些值不是单个数据点（人们的单个回答），而是跨国家和不同时期已聚合的值。
%%

You can use other external libraries to easily produce various advanced plots. One of such libraries is [Seaborn](https://seaborn.pydata.org/). You have already imported it at the beginning of this lab using `import seaborn as sns`. Run the cell below to see one of the many simple and efficient plotting possibilities (you will use this one later on in the other notebooks as well). Since the dataset has many columns it might take a few seconds to run.

%%
你可以使用其他的外部库轻松生成各种高级图表。其中一个这样的库是 Seaborn。你已经在本次实验的开头通过 `import seaborn as sns` 导入了它。运行下面的单元格来查看许多简单高效的绘图功能之一（你稍后在其他 Notebook 中也会用到这个功能）。由于数据集有很多列，运行可能需要几秒钟。
%%

```python
# If the plot doesn’t render, first try re-running this cell. If that doesn’t work, 
# you can restart the kernel (from the Kernel menu above) and try running the notebook again
sns.pairplot(df)
```

> [!result]
>     <seaborn.axisgrid.PairGrid at 0x7fcf9dee4bb0>
![intro-to-pandas-world-happiness_46_1.png](https://obsidian-image.wwtt.xyz/2025/12/intro-to-pandas-world-happiness_46_1.png)



With this kind of plot, you can see pairwise scatter plots for each pair of columns. On the diagonal (where both columns are the same), you don't have a scatter plot (which would only show a line), but a histogram showing the distribution of datapoints.

%%
通过这种图表，你可以看到每一对列之间的散点图。在对角线位置（两列相同），你不会得到散点图（那只会显示一条线），而是显示数据点分布的直方图。
%%

You can see that both the scatter plots and histograms have very different shapes across columns. Think about various insights you could get from this kind of visualization.

%%
你可以看到散点图和直方图在各列上的形状都非常不同。想想从这种可视化中可以获得的各种洞察。
%%
# 6. Operations on Columns

Sometimes the values in the columns are not giving you the information that you need, but there is a way to calculate that information from the values you have.

%%
有时列中的数值并没有给出你需要的信息，但你可以从已有数值中计算出所需信息。
%%

For example you can create a new column, which is a sum of two columns.

%%
例如，您可以创建一个新列，该列是两列的总和。
%%

```python
# Create a new column which is the sum of the year and the value on the life ladder.
df["this_column_makes_no_sense"] = df["year"] + df["life_ladder"]
# Create a new column which is the difference of two columns.
df["net_affect_difference"] = df["positive_affect"] - df["negative_affect"]

df.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
      <th>this_column_makes_no_sense</th>
      <th>net_affect_difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>3.724</td>
      <td>7.350</td>
      <td>0.451</td>
      <td>50.5</td>
      <td>0.718</td>
      <td>0.168</td>
      <td>0.882</td>
      <td>0.414</td>
      <td>0.258</td>
      <td>2011.724</td>
      <td>0.156</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>4.402</td>
      <td>7.509</td>
      <td>0.552</td>
      <td>50.8</td>
      <td>0.679</td>
      <td>0.191</td>
      <td>0.850</td>
      <td>0.481</td>
      <td>0.237</td>
      <td>2013.402</td>
      <td>0.244</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.614</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.121</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
      <td>2014.758</td>
      <td>0.242</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.164</td>
      <td>0.731</td>
      <td>0.480</td>
      <td>0.267</td>
      <td>2014.832</td>
      <td>0.213</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
      <td>2015.783</td>
      <td>0.346</td>
    </tr>
  </tbody>
</table>
</div>


Above you can see your dataframe with both new columns. The first one doesn't make much sense, it's just adding the year to the life ladder. The second one, however, find the net difference between positive and negative affect. Perhaps there's an interesting set of patterns between this new column and other columns that you'd now be able to explore. What other columns might you want to calculate? In general, the ability to create new columns using operations on existing columns can be a powerful tool.

%%
如上所示，你可以看到包含两个新列的 dataframe 。第一个列没有太多意义，它只是将年份加到生活梯子上。然而第二个列计算了正向情感与负向情感之间的净差异。这个新列与其他列之间可能存在一些有趣的模式组合，现在你可以探索这些模式。你还想计算哪些其他列呢？一般来说，利用现有列进行运算来创建新列的能力是一个强大的工具
%%

If you want to perform some more advanced operations on columns, you can use `DataFrame.apply()`, with which you can apply practically any function to a column. Below you can see how to use the `DataFrame.apply()` in various ways. Try to edit `my_function` to perform an operation of your choice.

%%
如果你想对列执行一些更高级的操作，可以使用 `DataFrame.apply()`。借此，你几乎可以将任何函数应用到列中。下面将展示如何以多种方式使用 `DataFrame.apply()` 。你可以尝试编辑 `my_function` 来执行你选择的操作。
%%


```python
# Using df.apply() with a lambda function
# Rescale the life_ladder column to values between 0 and 1 and save it to a new column
df['life_ladder_rescaled'] = df['life_ladder'].apply(lambda x: x / 10)

# Using df.apply() with your own function
# First define a function. The function can do whatever you want. This example will double the column's values
def my_function(x):
    # do stuff to x
    y = x * 2
    return y
# Apply the function.
df['my_function'] = df['life_ladder'].apply(my_function)

# Show the new dataframe
df.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
      <th>this_column_makes_no_sense</th>
      <th>net_affect_difference</th>
      <th>life_ladder_rescaled</th>
      <th>my_function</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>3.724</td>
      <td>7.350</td>
      <td>0.451</td>
      <td>50.5</td>
      <td>0.718</td>
      <td>0.168</td>
      <td>0.882</td>
      <td>0.414</td>
      <td>0.258</td>
      <td>2011.724</td>
      <td>0.156</td>
      <td>0.3724</td>
      <td>7.448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>4.402</td>
      <td>7.509</td>
      <td>0.552</td>
      <td>50.8</td>
      <td>0.679</td>
      <td>0.191</td>
      <td>0.850</td>
      <td>0.481</td>
      <td>0.237</td>
      <td>2013.402</td>
      <td>0.244</td>
      <td>0.4402</td>
      <td>8.804</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.614</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.121</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
      <td>2014.758</td>
      <td>0.242</td>
      <td>0.4758</td>
      <td>9.516</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.164</td>
      <td>0.731</td>
      <td>0.480</td>
      <td>0.267</td>
      <td>2014.832</td>
      <td>0.213</td>
      <td>0.3832</td>
      <td>7.664</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
      <td>2015.783</td>
      <td>0.346</td>
      <td>0.3783</td>
      <td>7.566</td>
    </tr>
  </tbody>
</table>
</div>


**Congratulations on finishing this lab.** If you understand the code above, you are well suited to start working on this week's programming assignment and other labs and assignments throughout the course which use Pandas. If you need a refresher on Pandas in other Exploratory Data Analysis labs, come back to this one and review the skills taught here.

%%
恭喜你完成了这个实验。如果你理解了上面的代码，那你已经做好了准备，可以开始完成本周的编程作业，以及课程中其他使用Pandas的实验和作业。如果你需要复习Pandas的相关内容，可以回到这个实验来复习这里教授的技能。
%%