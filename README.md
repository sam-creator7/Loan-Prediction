# Loan-Prediction

**Loan Prediction Project**

This project is aimed at building a machine learning model that will predict whether a loan will be approved or not, based on certain features such as income, loan amount, loan term, credit history, property area, etc.

**Data Import**

The first step of this project is to import the loan data set into Python, which can be done using the pandas library. The data is stored in a csv file and is loaded into a pandas dataframe for further processing.

import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\Desktop\\trimester 9\\ML\\load_train.csv")
print(data)

**Data Cleaning and Preprocessing**

Next, the data is checked for missing values. The missing values are filled with either the mean value for numerical data, or the most frequently occurring value for categorical data.

data.Gender = data.Gender.fillna("Male")
data.Married = data.Married.fillna("Yes")
data.Dependents = data.Dependents.fillna(0)
data.Self_Employed = data.Self_Employed.fillna("No")
data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360.0)
data.Credit_History = data.Credit_History.fillna(1.0)

After filling the missing values, the data is split into the input features (X) and target variable (y). The input features include all the variables except the loan approval status.

X = data.iloc[:, 1: 12].values
y = data.iloc[:, 12].values

The next step is to split the data into the training and test set, which can be done using the train_test_split function from the sklearn.model_selection library.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
Categorical data needs to be encoded before the model can be trained on it. This can be done using the LabelEncoder class from the sklearn.preprocessing library.


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_train[:,0] = labelencoder_X.fit_transform(X_train[:,0])
X_train[:,1] = labelencoder_X.fit_transform(X_train[:,1])
X_train[:,2] = labelencoder_X.fit_transform(X_train[:,2])
X_train[:,3] = labelencoder_X.fit_transform(X_train[:,3])
X_train[:,4] = labelencoder_X.fit_transform(X_train[:,4])
X_train[:,10] = labelencoder_X.fit_transform(X_train[:,10])
The target variable (loan approval status)
