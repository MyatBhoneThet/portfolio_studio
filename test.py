# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from statsmodels.formula.api import ols

# # 1.1
# train_1 = pd.read_csv('./train_1.csv', index_col = 0)
# train_2 = pd.read_csv('./train_2.csv', index_col = 0)
# new_data = pd.concat([train_1,train_2], axis = 0)
# print(new_data.shape)

# # 1.2
# print(new_data.dtypes)
# print(new_data.isnull().count())
# print(new_data.dropna())

# # 1.3
# print(new_data.fillna(new_data.value_counts()))

# # 1.4 
# data = new_data[['ApplicantIncome','LoanAmount']].corr()
# sns.heatmap(data, annot=True, cmap="Blues", fmt="d")
# plt.show()
# new_data.groupby('Loan_Status')[['ApplicantIncome','LoanAmount']]
# sns.scatterplot(data=new_data, x = 'LoanAmount',y = "ApplicantIncome")
# plt.show()

# # 1.5
# model = ols('LoanAmount ~ ApplicantIncome', data = new_data).fit()
# print(model.params)

# sns.regplot(x='ApplicantIncome', y = 'LoanAmount', data = new_data, ci = None)
# plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 2.1
telecom_churn = pd.read_csv('./telecom_churn.csv', index_col = 0)
print(telecom_churn.shape)

# 2.2
X = telecom_churn[["total_day_charge", "total_eve_charge", "total_night_charge", "customer_service_calls"]]
y = telecom_churn["churn"]
knn = KNeighborsClassifier(n_neighbors = 8)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
print(accuracy)
print()

# 2.3
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
print(accuracy)
print()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.33, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
print(accuracy)
print()

# 2.4
X = telecom_churn[["total_day_charge", "total_eve_charge", "total_night_charge", "customer_service_calls"]]
y = telecom_churn["churn"]
knn = KNeighborsClassifier(n_neighbors = 1)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
print(accuracy)
print()
X = telecom_churn[["total_day_charge", "total_eve_charge", "total_night_charge", "customer_service_calls"]]
y = telecom_churn["churn"]
knn = KNeighborsClassifier(n_neighbors = 10)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
print(accuracy)
print()
X = telecom_churn[["total_day_charge", "total_eve_charge", "total_night_charge", "customer_service_calls"]]
y = telecom_churn["churn"]
knn = KNeighborsClassifier(n_neighbors = 20)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
print(accuracy)
print()
X = telecom_churn[["total_day_charge", "total_eve_charge", "total_night_charge", "customer_service_calls"]]
y = telecom_churn["churn"]
knn = KNeighborsClassifier(n_neighbors = 25)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
print(accuracy)
print()

# 2.5
X = np.array([[35.0,17.5,10.1,1],
              [107.0,19.0,24.1,0],
              [113.0,9.9,11.2,2],
              [67.9,5.7,4.5,1]])
y = telecom_churn["churn"]
knn = KNeighborsClassifier(n_neighbors = 25)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print()

# 2.6
CustName = pd.read_csv('./new_data.csv')
X = np.array([[35.0,17.5,10.1,1],
              [107.0,19.0,24.1,0],
              [113.0,9.9,11.2,2],
              [67.9,5.7,4.5,1]])
y = CustName["cust_name"]
knn = KNeighborsClassifier(n_neighbors = 25)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print()
