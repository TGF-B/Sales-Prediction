# Sales Prediction
这次我们来分析一个连续型变量问题。   
假设我们拿到了公司的各渠道广告投入和销售数据，现在需要我们快速分析出分配下一季度的广告预算以及可能带来的业务量。
- 导入数据并查看头部
```python
import pandas as pd
import numpy as np
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
data.head()
```
>         TV  Radio  Newspaper  Sales    
>     0  230.1   37.8       69.2   22.1    
>     1   44.5   39.3       45.1   10.4    
>     2   17.2   45.9       69.3   12.0    
>     3  151.5   41.3       58.5   16.5    
>     4  180.8   10.8       58.4   17.9

导入成功！   
数据集展示的是过去5个月电视，电台，报纸三条广告投放渠道的支出和当月的销售额。   
我们首先来看看三条渠道与销售额的关系。

```python
correlation=data.corr()
print(correlation["Sales"].sort_values(ascending=False))
```
>     Sales        1.000000   
>     TV           0.901208   
>     Radio        0.349631   
>     Newspaper    0.157960   
>     Name: Sales, dtype: float64
显然，三种渠道的回报率从高到低排列，**表现最好的是电视广告。**
这种相关性我们也可以通过散点图的分布离散度来判断。
- 画图
```python
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()
```
依次更换y和size的变量名称，得到三张图：
电视：   
电台：   
报纸：   

从图中我们可以发现，虽然不同渠道的广告投放回报有差别，但是都与销售额呈现正向关系。   
考虑到我们要预测的销售额也是一个连续变量，所以我们尝试**用线性回归的方式来构建模型**。
- 建模
  - 数据集的划分
```python
x=np.array(data.drop(["Sales"],1))
y=np.array(data["Sales"])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                            test_size=0.2,
                                            random_state=42)#用20%的数据做测试集，random_state=42是业内惯例，因为据说数字42是一切问题的根本答案。。。
```
  - 模型训练
```python
model=LinearRegression()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest)
```
返回拟合度：
>     0.9059011844150826
说明模型的预测准确度还可以，接下来我们用新数据来应用,即用一组**季度广告投放预算**来预测可能带来的销售额。   
为此我们查看一下数据集的数字特征：
```python
data.describe()
```
>               TV       Radio   Newspaper       Sales   
>     count  200.000000  200.000000  200.000000  200.000000    
>     mean   147.042500   23.264000   30.554000   15.130500    
>     std     85.854236   14.846809   21.778621    5.283892   
从上表可以看出季度电视，电台和报纸广告投放费用的总和，平均值以及标准值。
我们取三者平均加和作为今年广告投放预算，再根据三者与销量的相关性强度将广告预算重新分配一下。
为此我们索性绘制一张表格：
|     |  TV  |  Radio | Newspaper |
|-----|------|--------|-----------|
|mean|147.04 |  23.26 | 30.55     |
| corr|0.9012|  0.3496| 0.1580    |
|weight|0.637|  0.248 | 0.112     |
|budget|127.94| 49.81 | 22.50     |

- 应用预测模型
我们将求得的预算分配带入模型中：
```python
features=np.array([[127.94,49.81,22.50]])
print(model.predict(features))
```
返回销售额：
>       [16.81370557]

结论： 
1. 电视，电台，报纸三种广告投放渠道中，电视广告的回报收益最大，达到了90.12%。
