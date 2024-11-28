# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2. Set variables for the assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing int the graph.
5.predict the regression for marks by using the representation of the graph.
6.Compare the graph and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Eesha Ranka 
RegisterNumber:24900107 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("C:\\Users\\admin\\Desktop\\jupyter notebook\\student_scores.csv")
print(df.head())
print(df.tail())
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Testing set)")
plt.xlabel("hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("MSE=",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE=",mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
```
 Hours  Scores
0    2.5      21
1    5.1      47
2    3.2      27
3    8.5      75
4    3.5      30
    Hours  Scores
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86
[[2.5]
 [5.1]
 [3.2]
 [8.5]
 [3.5]
 [1.5]
 [9.2]
 [5.5]
 [8.3]
 [2.7]
 [7.7]
 [5.9]
 [4.5]
 [3.3]
 [1.1]
 [8.9]
 [2.5]
 [1.9]
 [6.1]
 [7.4]
 [2.7]
 [4.8]
 [3.8]
 [6.9]
 [7.8]]
[21 47 27 75 30 20 88 60 81 25 85 62 41 42 17 95 30 24 67 69 30 54 35 76
 86]
[17.04289179 33.51695377 74.21757747 26.73351648 59.68164043 39.33132858
 20.91914167 78.09382734 69.37226512]
[20 27 69 30 62 35 24 86 76]
```

![download](https://github.com/user-attachments/assets/37b89932-6986-48bd-a5c3-03976d867d18)

![download](https://github.com/user-attachments/assets/beefc655-05f0-44ab-b494-8c20a1ff7679)

```

MSE= 25.463280738222593
MAE= 4.691397441397446
RMSE= 5.046115410711748


```



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
