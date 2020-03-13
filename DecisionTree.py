import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('C:\\Users\shashikant\Desktop\polynomial_regression\polynomial.csv')
df

x = df[['level']].values
y = df[['salary']].values

model = DecisionTreeRegressor()
model.fit(x,y)
model.predict([[6.5]])

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.title('Decision tree Algorithm')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.scatter(x,y,color = 'g')
plt.plot(x_grid,model.predict(x_grid),color = 'b')



