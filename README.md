# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data

    Clean and format your data
    Split your data into training and testing sets

2.Define your model

    Use a sigmoid function to map inputs to outputs
    Initialize weights and bias terms

3.Define your cost function

    Use binary cross-entropy loss function
    Penalize the model for incorrect predictions

4.Define your learning rate

    Determines how quickly weights are updated during gradient descent

5.Train your model

    Adjust weights and bias terms using gradient descent
    Iterate until convergence or for a fixed number of iterations

6.Evaluate your model

    Test performance on testing data
    Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters

    Experiment with different learning rates and regularization techniques

8.Deploy your model

    Use trained model to make predictions on new data in a real-world application.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Nisha.J
RegisterNumber: 212223040133 
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Initial data set:
![318799335-cfd0dcc8-a05a-4667-bf00-df32c726d6fe](https://github.com/user-attachments/assets/fecd62bb-a570-41c7-8b9f-6237e1e84a18)

### Data info:
![318799431-1f784783-3223-41e3-b45a-f55168d37536](https://github.com/user-attachments/assets/514b1c55-7547-4c6a-bafc-a7f9fae31f6a)

### Optimization of null values:
![318799669-8003d2b0-c5ac-4c21-a612-b418d155a84e](https://github.com/user-attachments/assets/ce8cc128-2598-44d8-b97c-6f6585a14fae)

### Assignment of x and y values:
![318799745-0f8d9936-9839-49fb-9135-1629166a4dcc](https://github.com/user-attachments/assets/e5b7fff0-64af-485f-b075-af7bc964c823)
![318799806-6679ef6f-9dcf-4519-a7ce-80e914ea8d4d](https://github.com/user-attachments/assets/0a5cf4b6-8de8-4f0e-b5bf-7c476cf3f082)

### Converting string literals to numerical values using label encoder:
![318799913-4046c60c-d6be-48ad-a10f-310eb1a02e4f](https://github.com/user-attachments/assets/eb154090-a8fe-48e4-89db-43a2c8048496)

### Accuracy:
![318800047-15dea803-21d5-470c-9e49-8b3693d99391](https://github.com/user-attachments/assets/b630d7ad-3ac9-4e38-b8b7-8390826e7d23)

### Prediction:
![318800136-d833d4d1-004b-42ae-b790-dcd450b6651e](https://github.com/user-attachments/assets/20619e50-a9d2-42f0-812f-125b8842ec6c)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
