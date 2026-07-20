import math
import numpy as np
import matplotlib.pyplot as plt
 from scipy import stats
# import seaborn as sns
print("Square Root:", math.sqrt(9))
print("factorial", math.factorial(4))
print("factorial", math.pow(4,2))
print("ceil", math.ceil(5.66))
print("Floor", math.floor(5.66))
print("Pi value", math.pi)
print("Euler's number", math.e)

arr = np.array([2,4,6,8,10])
print("Array",arr)
print("mean:",np.mean(arr))
print("Sum:",np.sum(arr))
print("Max:",np.max(arr))

matrix = np.array([[1,2],[3,5]])
print("Matrix:\n",matrix)
print("transpose:\n",matrix.T)


x =[1,2,3,4,5]
y =[10,20,30,40,40]
plt.plot(x,y)
plt.xlabel("X axis")
plt.xlabel("Y axis")
plt.title("simple line graph")
plt.show()

# data =[10,20,30,40,50]
# sns.histplot(data)
# plt.title("seaborn hoistogram")
# plt.show()
# used for statistical and scientific computations.
data =[5,2,4,7,9]
print("Mean;",stats.tmean(data))
print("Median;",stats.scoreatpercentile(data,50))
print("Mode;",stats.mode(data))


# exp1 partb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
uploaded = files.upload()
df =pd.read.csv("titanic Dataset.csv")
print("First Five Records")
print(df.head())