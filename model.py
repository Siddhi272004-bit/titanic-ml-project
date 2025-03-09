import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
#load the titanic dataset

df=pd.read_csv('titanic.csv')

#preprocessing
df.dropna(inplace=True)
df['Sex']=df['Sex'].map({'male':0,'female':1})

#features and target
x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

#train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train model

model=XGBClassifier()
model.fit(x_train,y_train)

#save model as pickle file,to be then accessed by pickle in app.py
pickle.dump(model,open('titanic_model.pkl','wb'))


print(" Model trained and saved as titanic_model.pkl")