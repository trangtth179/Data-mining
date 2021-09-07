import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import pickle

import warnings
warnings.filterwarnings("ignore")


# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('.\credit_cards.csv')
df.head()

df.shape

corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)
plt.show()
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

# Dropping id column as it's no use
df.drop('ID',axis=1, inplace=True)

# Putting feature variable to X
X = df.drop('default',axis=1)

# Putting response variable to y
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

rfc = RandomForestClassifier()

# fit
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)

# Let's check the report of our default model
print(classification_report(y_test,predictions))
# Printing confusion matrix
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))

# Saving the model to disk
pickle.dump(rfc, open('model.pkl', 'wb'))
