import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks")

data1 = pd.read_csv("Dataset1.csv")
data2 = pd.read_csv("Dataset2.csv")
data3 = pd.read_csv("Dataset3.csv")
data1=np.array(data1)
data2=np.array(data2)
data3=np.array(data3)
ax = sns.scatterplot(x=data1[:,0],y=data1[:,1])
# plt.scatter(data1[:,0],data1[:,1])
# plt.title("data1")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()