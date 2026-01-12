---
aliases: [Exploratory Data Analysis - Linear Regression]
tags: []
created: 2026-01-10, 15:57:52
modified: 2026-01-12, 15:37:50
---

# Exploratory Data Analysis - Linear Regression

Welcome to the fourth notebook of the exploratory data analysis (EDA) series! In this notebook you will use the [World Happiness Report](https://worldhappiness.report/) dataset, that you have already seen in the Pandas tutorial notebook. The dataset consists of 2199 rows, where each row contains various hapiness-related metrics for a certain country in a given year. 

%% 欢迎来到数据探索分析系列的第四部分！在这个 Notebook 中你将使用世界幸福报告的数据集，该数据集你已经在 Pandas 教程的 Notebook 中接触过。它有 2199 行，每行包含了某年某个国家各种各样的幸福指数。 %%

In the previous video you have learned about one of the most common applications of Maximum Likelihood Estimation (MLE), which is linear regression. Linear regression is a statistical model that is used to estimate a linear relationship between two or more variables. In case of simple linear regression you have one independent (explanatory) variable and one dependent variable (response), while in case of multiple linear regression, you have more than one explanatory variable. You can read more about linear regression on [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression).

%% 在上一个视频中你已经学习了最大似然估计最常见的应用之一，那就是线性回归。线性回归是估计两个或多个变量之间线性关系的统计模型。在简单线性回归的情况下，您有一个自变量（解释变量）和一个因变量（响应变量），在多元线性回归的情况下，有大于一个的自变量。更多有关线性回归的内容可以查看维基百科。 %%

In this notebook, you will create your own linear regression model and fit it to one and more explanatory variables to predict the response. You will use an open-source, commercially usable machine learning toolkit called [scikit-learn](https://scikit-learn.org/stable/index.html). This toolkit contains implementations of many machine learning and statistical algorithms that you can encounter as a data scientist or machine learning practitioner.

%% 在这个 Notebook 中，你将创建你自己的线性回归模型，将其拟合到一个或多个解释变量以预测响应。你将使用一个开源的，并且可以商业使用的工具包——scikit-learn。这个工具包包含了实现多种机器学习和统计算法的实现，作为数据科学或者机器学习的从业者经常会遇到它。 %%
# 1. Import the Libraries
As usual, first import all the necessary libraries that you will use in the notebook.

%% 和往常一样，第一步先导入必要的库。 %%

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import various functions from scikit-learn to help with the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Import functions to create interactive widgets
import ipywidgets as widgets
from ipywidgets import interact_manual, fixed

# Here are some functions that are abstracted away to not clutter the notebook too much
import utils
```

# 2. Import and Process the Data

You will work with the "World happiness" dataset that you have already seen in the "Introduction to Pandas" notebook. The first thing you need to do is open the notebook and clean up the data. This code will rename columns so that there are no white spaces and drop any missing values.

%% 您在Pandas 入门笔记本中已经看到的“World happiness”数据集。第一件事情是读取并清理数据。下面的代码将重命名数据列并丢弃缺失的数据。 %%

```python
# Open the notebook
df = pd.read_csv('data/world_happiness.csv')

# Rename the column names so they dont contain spaces
df = df.rename(columns={i: "_".join(i.split(" ")).lower() for i in df.columns})

# Drop all of the rows which contain empty values. These will not be good for fitting.
df = df.dropna()

# Show the dataframe
df.head()
```


> [!result]
> ||country_name|year|life_ladder|log_gdp_per_capita|social_support|healthy_life_expectancy_at_birth|freedom_to_make_life_choices|generosity|perceptions_of_corruption|positive_affect|negative_affect|
|---|---|---|---|---|---|---|---|---|---|---|
|0|Afghanistan|2008|3.724|7.350|0.451|50.5|0.718|0.168|0.882|0.414|0.258|
|1|Afghanistan|2009|4.402|7.509|0.552|50.8|0.679|0.191|0.850|0.481|0.237|
|2|Afghanistan|2010|4.758|7.614|0.539|51.1|0.600|0.121|0.707|0.517|0.275|
|3|Afghanistan|2011|3.832|7.581|0.521|51.4|0.496|0.164|0.731|0.480|0.267|
|4|Afghanistan|2012|3.783|7.661|0.521|51.7|0.531|0.238|0.776|0.614|0.268|

Have a closer look at the output of the cell above. The dataset consists of the following columns:

%% 仔细看上面单元格的输出，数据集由如下列组成： %%

- `country_name`: Name of the country where the data was taken.
- `year`: The year when data was taken.
- `life_ladder`: The average of the estimates of life quality on a scale of 1 to 10 as given by a survey. In the survey people subjectively estimate the quality of their own life.
- `log_gdp_per_capita`: Logarithm of gross domestic product (log GDP) in purchasing power parity (PPP).
- `social_support`: National avarage of responses to the binary question: "If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?".
- `healthy_life_expectancy_at_birth`: Life expectancy at birth.
- `freedom_to_make_life_choices`: National avarage to the binary question: "Are you satisfied or dissatisfied with your freedom to choose what you do with your life?".
- `generosity`: Derived from answering the question: "Have you donated money to a charity in the past month?" and GDP.
- `perceptions_of_corruption`: Average of responses to questions about corruption.
- `positive_affect`: Average of answers to three positive affect questions, covering laugh, enjoyment and doing interesting things
- `negative_affect`: Average of answers to three negative affect questions, covering worry, sadness and anger.

%%
- `country_name`:  数据来源国的名字。
- `year`: 数据来源的年份。
- `life_ladder`: 根据问卷调查，受访者以 1 至 10 分制对其生活质量进行主观评估，所得评分的平均值。
- `log_gdp_per_capita`: 以购买力平价（PPP）计算的国内生产总值（GDP）对数。
- `social_support`: 二元问题的全国均值：如果你遇到的困难，是否有亲戚朋友可以帮助你？
- `healthy_life_expectancy_at_birth`: 出生预期寿命
- `freedom_to_make_life_choices`:  二元问题的全国均值：你对自己选择人生的自由感是否满意？
- `generosity`: 源自回答“过去一个月内您是否向慈善机构捐款？”及国内生产总值（GDP）的数据。
- `perceptions_of_corruption`: 对腐败问题的平均回应得分。
- `positive_affect`: 对三个积极情感问题（包括笑、享受和做有趣事情）回答的平均值
- `negative_affect`: 对三个消极情感问题（包括焦虑、悲伤和愤怒）回答的平均值
 %%
 
Detailed explanations of the columns in the data can be found [here](https://happiness-report.s3.amazonaws.com/2023/WHR+23_Statistical_Appendix.pdf).

%% 数据列的详细解释可以看[这里](https://happiness-report.s3.amazonaws.com/2023/WHR+23_Statistical_Appendix.pdf)。 %%
# 3. Inspect the Data

Before you jump into fitting the linear regression model to your data, it makes sense to visualize it to get the feeling of what you are dealing with. The Seaborn `pairplot` is a very useful function that automatically plots the scatter plots between each pair of columns in the dataframe, as well as histograms of the values in each column. Run the cell below to visualize the data.

%% 在你开始对你的数据进行线性拟合之前，将其可视化以便了解正在处理的内容是非常有意义的。Seaborn 的 `pairplot` 是一个非常实用的函数，它会自动绘制 Dataframe 中各列两两之间的散点图，以及每列数值的直方图。运行下面单元格查看数据。%%

```python
# If the plot doesn’t render, first try re-running this cell. If that doesn’t work, 
# you can restart the kernel (from the Kernel menu above) and try running the notebook again
sns.pairplot(df)
```

> [!result]
>     <seaborn.axisgrid.PairGrid at 0x7fea06fbc5b0>
![linear-regression-world-happiness_7_1.png](https://obsidian-image.wwtt.xyz/2026/01/linear-regression-world-happiness_7_1.png)

    

You can see that some of the scatter plots seem quite elongated and might show some significant correlation between the two variables. These pairs may be good candidates for independent-dependent variable pairs. But just looking at the points can be dangerous. You need to have an idea of what you would like to predict and which variables may be a good choice for explanatory variables. Take another look at the column names and think about which ones you would use as explanatory variables and what you would want to predict given this dataset. You can also get some ideas from the [official report](https://happiness-report.s3.amazonaws.com/2023/WHR+23.pdf).

%% 你可以看到，一些散点图看起来非常细长，这表明两个变量可能显著相关。这些配对可能是因变量和自变量配对的。但是如果只看这些点其实是比较危险的。你需要明确想要预测的目标，以及哪些变量适合作为解释变量。再次审视一下这些列名，并思考一下将哪些列用作解释变量，以及基于这个数据集，想要预测什么。你同样可以查阅[官方报告](https://happiness-report.s3.amazonaws.com/2023/WHR+23.pdf)获得一些灵感。 %%
# 4. Simple Linear Regression

Now that you have a good sense of the data, you can choose your independent (X) and dependent (y) variable. Let's start with the obvious ones: use the GDP per capita to explain the value on the life ladder, which measures the happiness of the people.

%% 现在你应该对数据有一个判断，你可以选择你的自变量和因变量。让我们从最明显的开始：使用人均 GDP 来解释生活阶梯评分来衡量人们的幸福指数。 %%
## 4.1 Define the Variables

In machine learning you would typically not use the same data to train and evaluate your model and that is because you want to know how well the model generalizes to new (previously unseen) data. You can run the cell below to split your data into two groups. It will create the training data, `X_train` and `y_train`, as well as the test data, `X_test` and `y_test`.  Note the uppercase X and lowercase y. This is to emphasize that the X variable is two dimensional, while the y variable is one dimensional. This is for better generalization for when you in fact use more than one explanatory variable. You will see this in action later, using more variables for predicting the happiness.

%% 在机器学习中通常不会使用相同的数据进行训练和评估模型，这是因为你需要知道模型对新数据（从未见过的数据）的泛化能力。运行下面的代码将数据分割成两组。训练数据 `X_train` 和 `y_train`，测试数据 `X_test` 和 `y_test`。注意是大写的 X 和小写的 y。同时这里强调一下， X 变量是二维的，y 的变量是一维的。这是为了当你使用更多的解释变量的时候，进行更好的泛化。稍后你会看到它的实际应用：使用更多的变量预测幸福指数。 %%

```python
# Get the data from the dataframe.
X = df[['log_gdp_per_capita']]
y = df['life_ladder']

# Create the train-test split
# Note the test_size=0.2. This means you will use 20% of the data in your test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Show how X_train and y_train look like
print(f"Independent (X) variable:\n{X_train.values}\n")
print(f"Dependent (y) variable:\n{y_train.values}\n")
print(f"Shape Independent (X) variable: {X_train.shape}")
print(f"Shape Dependent (y) variable: {y_train.shape}")
```

> [!result]
> 	Independent (X) variable:
> 	[[ 9.884]
> 	 [10.594]
> 	 [ 8.521]
> 	 ...
> 	 [10.635]
> 	 [10.424]
> 	 [10.006]]
> 	
> 	Dependent (y) variable:
> 	[5.383 6.027 4.483 ... 6.199 6.183 6.174]
> 	
> 	Shape Independent (X) variable: (1566, 1)
> 	Shape Dependent (y) variable: (1566,)

## 4.2 Create and Fit the Linear Regression Model

Now that you have your train and test data, it is time to create your linear regression model. This is actually very simple and can be done in a single line using `LinearRegression` as shown below.

%% 现在你有训练和测试数据了，是时候创建你的线性回归模型了，这实际上非常简单，使用 `LinearRegression` 一行即可： %%

```python
lr = LinearRegression().fit(X_train, y_train)
```

The first part of the statement, `LinearRegression()`, is instantiating a linear regression model. You could also set some parameters to this function. The `fit(X_train, y_train)` method is then performing the actual training of the model, fitting its parameters to your training data. 

%% 声明的第一部分 `LinearRegression()` 是将线性回归模型实例化。这个函数同样可以设置一些参数。`fit(X_train, y_train)` 方法随后将训练模型，使模型的参数和训练数据拟合。 %%
## 4.3 View Parameters

You can write down the equation for the line that you fit to the data as $\hat{y}=Wx+b$, where $x$ is the explanatory variable (or variables) and $\hat{y}$ is your response, while $W$ and $b$ are the parameters that you fit. In the cell below, you can see how to access the parameters $W$ and $b$ of the model you just trained.

%% 写出拟合数据的线性等式 $\hat{y}=Wx+b$，其中 $x$ 是解释变量（或者多个解释变量），$\hat{y}$ 是响应的预测值，$W$ 和 $b$ 均为拟合的参数，在下面代码中，打印出了在训练完成后 $W$ 和 $b$ 的值。 %%

```python
b = lr.intercept_
w = lr.coef_
print(f"Model parameters:\nw: {w},\nb: {b}")
```

> [!result]
> 	Model parameters:
> 	w: [0.77816718],
> 	b: -1.8135023793672875

## 4.4 Make Predictions and Evaluate the Model

now that you have fit your model, it's time to start using it. In machine learning, a model like this is used not only to describe the data that you already have, but more importantly to create predictions based on new datapoints. To assess how well the model works, you have already split the dataset into the train and test sets, so that you can evaluate how the model performs on previously unseen data.

%% 现在你已经将模型拟合完成，是时候使用它了。在机器学习中，一个模型不仅仅是用来描述已有的数据，更重要的是基于新的数据点进行预测。为了评估模型的性能，会将数据集切分为训练数据和测试数据，这样就可以评估模型在之前未见过的数据上的表现。 %%

The default way to calculate the predictions on the test set is to use `.predict()`. You will use `.predict()` on the whole test set at once. To reinforce how the model actually works, you will compare the predictions given by `.predict()` with ones calculated "by hand", using the parameters W and b that you extracted in the previous cell.

%% 一个默认的做法是使用 `.predict()` 预测测试数据。你将一次性对整个测试集调用 `.predict()` 函数。为了加深你对模型实际工作的理解，你将使用上一个单元格中的参数 $W$ 和 $b$ 进行手动计算，并与 `.predict()` 的计算进行比较。 %%

```python
# Make a prediction using lr.predict()
y_test_preds = lr.predict(X_test)

# Make a prediction by hand using w, b.
y_pred = np.dot(X_test, w) + b

# Check whether both results are the same
print(f"prediction using np.dot() and predictions using lr.predict are the same: {(y_pred == y_test_preds).all()}\n")

# Compare some of the predictions with actual (target) values.
print(f"Last four predictions on the test set:\n{y_pred[:4]}\n" )
print(f"Target values \n{y_test[:4].values}\n")

mae = metrics.mean_absolute_error(y_test, y_test_preds)
print(f"Mean Absolute Error on the test set: {mae:.2f}")
```

> [!result]
> 	prediction using np.dot() and predictions using lr.predict are the same: True
> 	
> 	Last four predictions on the test set:
> 	[4.14336741 6.53934416 6.30511584 4.80247501]
> 	
> 	Target values 
> 	[4.016 7.393 6.5   4.51 ]
> 	
> 	Mean Absolute Error on the test set: 0.57


The results show that at least for the last four rows, the predictions were reasonably close to correct. The average absolute error across the test set is also 0.57 which seems to suggest that while the model can't perfectly predict happiness, it gives a pretty good approximation.

%% 结果显示，至少对于最后四行，预测结果与正确答案相当接近。测试集的平均绝对误差为 0.57，这似乎表明，虽然模型不能完美的预测幸福指数，但它也有一个非常好的近似值。 %%
## 4.5 Plot Results

Now you can plot the predictions together with the training data to get a visual feeling of how the model performs. The blue points on the plot are real data from the training set. The orange points on the plot have their x value taken from the test set, but their y value is a prediction created by the model.

%% 现在，你可以将预测结果与训练数据一起绘制出来，直观地感受模型的表现。图中蓝色的数据点是训练集的真实数据。图中橙色的数据点的 x 值取至测试集数据，它们的 y 值则是由模型预测得到的。 %%

```python
plt.figure()
plt.scatter(X_train, y_train, label = 'Training Data')
plt.scatter(X_test, y_pred, label = 'Predictions on the Test Set')
plt.legend()
```

> [!result]
>     <matplotlib.legend.Legend at 0x7fe9fccfdd30>
![linear-regression-world-happiness_18_1.png|500](https://obsidian-image.wwtt.xyz/2026/01/linear-regression-world-happiness_18_1.png)

    
You can see that all of the predictions lie on a straight line that fits the data best. The data, however, has a lot more variation and is not perfectly explained just by this one line. As you know there are many more variables in this dataset and some other variables may be able to explain this variation. You will see how this plays out using multiple linear regression in the next section.

%% 你可以看到，所有的预测都位于一条最能拟合数据的直线上。另外，有相当多的数据不能仅仅只靠这根线来解释。正如您所知，此数据集中还有更多变量，并且其他一些变量可能能够解释这种变异。在下一节中，您将看到如何使用多元线性回归来解决这个问题。 %%
# 5. Multiple Linear Regression

Often there is more than just one variable that explains the behavior of other variables. In this case you can use multiple linear regression. In the cell below there is a function defined to make your life easier:

%% 常常多个变量比一个变量更能解释其他变量的行为。在这个示例中你会使用多个线性回归。在下面的单元格中，定义了一个函数，以使您的生活更轻松: %%

- `fit_and_plot_linear_model`: This function is very similar to the work you already did above. The key difference is that it can take more than one feature at a time, so you will be able to experiment with building models with two or more explanatory variables. In addition it calculates the feature importance. We will not go into detail about it here, but all you need to know is that importance score tells you the relative importance of explanatory variables. The higher the score, the more important the variable is in predicting the outcome.

%% - `fit_and_plot_linear_model`： 这个函数和你上面已经完成的工作非常类似。关键的区别在它可以一次传入多个特征，所以你能构建两个或者多个解释变量的模型实验。此外它可以计算特征重要性。在此我们不进行详细阐述，你只需了解，重要性得分表明了解释变量的相对重要性。得分越高，变量在预测中就越重要。 %%

Check out the code in the cell below if you want to understand better what is happening. Note that the plotting and feature importance has been abstracted away to the `utils` file, so that it doesnt add unnecessary clutter the code.

%% 如果你想知道如何实现的最好查看下面的代码。注意，绘图和特征重要性已经被抽象到 `utils` 文件，这样可以避免在代码中添加不必要的混乱。 %%

```python
def fit_and_plot_linear_model(data, features):
    
    # Create a list of features you want to use as explanatory variables
    features = list(features)
    
    # Create the linear regression model
    
    # Select the data
    y = data['life_ladder']
    X = data[features]
    # Create a train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Fit the linear regression model
    lr = LinearRegression().fit(X_train, y_train)

    # Calculate the feature imporance
    most_important, feature_importance_df = utils.calculate_feature_importance(features, lr, X_test, y_test)

    # Prepare the data for plotting
    X_plot = data[most_important]
    Y_real = data["life_ladder"]
    X_test_plot = X_test[most_important]
    y_test_preds = lr.predict(X_test)
    
    # Plot the data and the model
    utils.plot_happiness(most_important, X_plot, Y_real, X_test_plot, y_test_preds)

    # Create a plot of feature imporance if there is more than one feature
    if feature_importance_df is not None:
        utils.plot_feature_importance(feature_importance_df)
    
    # Calculate and print out the mean absolute error
    mae = metrics.mean_absolute_error(y_test, y_test_preds)
    print(f"Mean Absolute Error: {mae:.2f}\n")
```

Now that you have your functions defined, it is time to run them. The cell below does that in an interactive way using `widgets` and `interact_manual`. This allows you to run the function many times without typing anything in but rather changing parameters using the widgets.

%% 现在函数构建好了，是时候运行它了。下面的单元格使用 `widgets` 和 `interact_manual` 以交互方式执行此操作。它可以运行这个函数很多次，不需要输入任何东西，如果要修改参数则使用小组件。 %%

In the first line of code you define all of the possible predictor variables and the rest of the code takes care for interactively running the functions. After running the cell below you can select one or more predictors from the list to perform linear regression. 

%% 第一行代码定义了所有可能预测的变量，代码的其余部分负责交互式运行函数。在运行下面的代码前，你需要从列表中选择一个或则多个预测变量执行线性回归。 %%

The function will calculate the linear regression and plot the data and the results on a 2D plot. For the x-axis it will automatically choose the variable with the highest feature importance score.

%% 这个函数将计算线性回归并绘制数据点，最后将得到一个 2 维的绘图，$x$ 轴将自动选择最高特征重要性得分的变量。 %%

**Intrustions to use the widget**
- To select different variables from the menu just do ```Ctrl+click``` on the variables you want to select. Single click will only consider the variable you are clicking on. 
- To select consecutive variables you can also click on the first variable you want and ```Shift+click``` on the last one you want. This will select all variable in between. You can also click on the first one and select with ```Shift+down arrow```.

%% 
小部件的使用说明
- 要从菜单中选择不同的变量，只需按住 Ctrl 键并点击你想要选择的变量。单独点击只会选中你点击的那个变量。  
- 要选择连续的变量，你也可以先点击想要的第一个变量，然后按住 Shift 键点击最后一个变量。这将选中两者之间的所有变量。或者，你也可以先点击第一个变量，然后使用 Shift+向下箭头键进行选择。
%% 

```python
# List of all possible predictor variables
predictors = ['year', 'log_gdp_per_capita', 'social_support', 'healthy_life_expectancy_at_birth', 'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption', 'positive_affect', 'negative_affect']

# Widget for feature selection
feature_selection = widgets.SelectMultiple(
    options=predictors, value=predictors,
    description="Features", disabled=False,
)

# Interactive call to the function
interact_manual(fit_and_plot_linear_model, data=fixed(df), features=feature_selection);
```

> [!result]
> 	选择了所有特征之后的绘图
![GDP拟合幸福感.png|600](https://obsidian-image.wwtt.xyz/2026/01/GDP拟合幸福感.png)
![特征重要性排序.png|600](https://obsidian-image.wwtt.xyz/2026/01/特征重要性排序.png)



You see that when you select more explanatory variables, the points do not lie on a straight line in 2D anymore. This is because another variable (which is not on the 2D plot) contributes to the move away from the line. In a higher dimensional space, the points still lie on a straight line.

%% 当你选择了多个解释变量，这些点在二维空间中不再位于一条直线上了。这是因为有其他的变量（没有显示在图中）的构建将其从线上移开了。在更高维度的空间，这些线依然在一条直线上。 %%

**Congratulations on finishing this lab.** If you understand what is happening above, you are well suited to perform linear regression with scikit-learn. Later in this course you will see linear regression again using another Python library.

%% 恭喜你完成这个实验。如果你了解上述内容，那么你非常适合使用 scikit-learn 做线性回归。之后这门课会让你再次看到线性回归，用另外的 Python 库实现的。 %%
