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
显然，三种渠道的回报率从高到低排列，最好的是电视广告。
这种相关性我们还可以通过散点图的分布离散度来判断。
- 画图
