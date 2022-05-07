#导入数据
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(data.head())

#投入与销售额的关系
correlation=data.corr()
print(correlation["Sales"].sort_values(ascending=False))

#绘制散点图
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()

#建模
#数据集的划分
x=np.array(data.drop(["Sales"],1))
y=np.array(data["Sales"])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                            test_size=0.2,
                                            random_state=42)#用20%的数据做测试集，random_state=42是业内惯例，因为据说数字42是一切问题的根本答案。。。
#模型训练
model=LinearRegression()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest)
#查看数据集的数字特征
data.describe()

#应用预测模型

features=np.array([[127.94,49.81,22.50]])
print(model.predict(features))
