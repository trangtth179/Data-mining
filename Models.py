import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree


import warnings
warnings.filterwarnings("ignore")


# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('.\credit_cards.csv')
df.head()
df.shape

corr = df.corr()
sns.heatmap(corr,annot=True)
corr = np.array(corr)

sex = df['SEX'].value_counts()
ed = df['EDUCATION'].value_counts()
mg = df['MARRIAGE'].value_counts()
age = df['AGE'].value_counts()
print(sex)
print('----------')
print(ed)
print('----------')
print(mg)
print('----------')
print(age)

df.loc[df['EDUCATION']==5,'EDUCATION']=4     
df.loc[df['EDUCATION']==6,'EDUCATION']=4
df.loc[df['EDUCATION']==0,'EDUCATION']=4
df.loc[df['MARRIAGE']==0,'MARRIAGE']=3
print(df['EDUCATION'].value_counts())
print('---------')
print(df['MARRIAGE'].value_counts())

new_corr = df[['EDUCATION','MARRIAGE','default']].corr()
nc = np.array(new_corr)
nc = nc[2,0:2]
oc = corr[24,3:5]
diff = nc-oc
print('Improvements in correlation:',diff)
df[['EDUCATION','MARRIAGE','default']].corr()

# Loại bỏ biến ID vì k cần sử dụng
df.drop('ID',axis=1, inplace=True)

# Đặt các biến dùng dự đoán bằng biến X
X = df.drop('default',axis=1)

# Đặt nhãn vào biến y
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

#Sử dụng thuật toán Random Forest
rfc = RandomForestClassifier()

# fit
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)

y_pred = rfc.predict(X_test)
y_pred = (y_pred > 0.5)
conf_matr = confusion_matrix(y_test, y_pred)



TP = conf_matr[0,0]; FP = conf_matr[0,1]; TN = conf_matr[1,1]; FN = conf_matr[1,0]
print('Confusion Matrix : ')
print(conf_matr)
print()
print('True Positive (TP)  : ',TP)
print('False Positive (FP) : ',FP)
print('True Negative (TN)  : ',TN)
print('False Negative (FN) : ',FN)

acc = (TP+TN)/(TP+TN+FP+FN)
print('Tỉ lệ dự đoán chính xác của thuật toán Random Forest '+ str(round(acc*100)) + ' %')

pre = TP/(TP+FP)
print('Trong đó có  '+ str(round(pre*100)) + ' % thực sự thanh toán tín dụng cho tháng tới.')

#Thuật toán Decision trees

dtc=tree.DecisionTreeClassifier()
dtc = dtc.fit(X_train, y_train)
tree.plot_tree(dtc) 




y_pred = dtc.predict(X_test)
y_pred = (y_pred > 0.5)
conf_matr = confusion_matrix(y_test, y_pred)

TP = conf_matr[0,0]; FP = conf_matr[0,1]; TN = conf_matr[1,1]; FN = conf_matr[1,0]
print('Confusion Matrix : ')
print(conf_matr)
print()
print('True Positive (TP)  : ',TP)
print('False Positive (FP) : ',FP)
print('True Negative (TN)  : ',TN)
print('False Negative (FN) : ',FN)

acc = (TP+TN)/(TP+TN+FP+FN)
print('Tỉ lệ dự đoán chính xác của thuật toán Decision tree '+ str(round(acc*100)) + ' %')

pre = TP/(TP+FP)
print('Trong đó có  '+ str(round(pre*100)) + ' % thực sự thanh toán tín dụng cho tháng tới.')

#Thuật toán KNeibors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

predictions = knn.predict(X_test)

y_pred = knn.predict(X_test)
y_pred = (y_pred > 0.5)
conf_matr = confusion_matrix(y_test, y_pred)

TP = conf_matr[0,0]; FP = conf_matr[0,1]; TN = conf_matr[1,1]; FN = conf_matr[1,0]
print('Confusion Matrix : ')
print(conf_matr)
print()
print('True Positive (TP)  : ',TP)
print('False Positive (FP) : ',FP)
print('True Negative (TN)  : ',TN)
print('False Negative (FN) : ',FN)

acc = (TP+TN)/(TP+TN+FP+FN)
print('Tỉ lệ dự đoán chính xác của thuật toán KNeibors '+ str(round(acc*100)) + ' %')

pre = TP/(TP+FP)
print('Trong đó có  '+ str(round(pre*100)) + ' % thực sự thanh toán tín dụng cho tháng tới.')

#Thuật toán Logistic regression

lr = LogisticRegression(C=0.1, solver='liblinear')
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)

y_pred = lr.predict(X_test)
y_pred = (y_pred > 0.5)
conf_matr = confusion_matrix(y_test, y_pred)

TP = conf_matr[0,0]; FP = conf_matr[0,1]; TN = conf_matr[1,1]; FN = conf_matr[1,0]
print('Confusion Matrix : ')
print(conf_matr)
print()
print('True Positive (TP)  : ',TP)
print('False Positive (FP) : ',FP)
print('True Negative (TN)  : ',TN)
print('False Negative (FN) : ',FN)

acc = (TP+TN)/(TP+TN+FP+FN)
print('Tỉ lệ dự đoán chính xác của thuật toán Logistic Regression '+ str(round(acc*100)) + ' %')

pre = TP/(TP+FP)
print('Trong đó có  '+ str(round(pre*100)) + ' % thực sự thanh toán tín dụng cho tháng tới.')



