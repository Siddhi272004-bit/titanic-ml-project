#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier,for bossting the score we gonna use XGbOOST NOW MASTER OF TABULAR DATA HAHAHA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report

#load dataset
df=pd.read_csv(r"C:/Users/KIIT0001/titanic-ml-project/titanic.csv")

# Preprocessing: Handle Missing Values
df["Sex"]=df["Sex"].map({"male":0,"female":1})

# Select Features and Target,now adding super features too
x=df[["Pclass","Sex","Age","Fare"]]
y=df["Survived"]

# family size
df["FamilySize"]=df["SibSp"]+df["Parch"]+1

# isAlone
# checking first
df["isAlone"]=(df["FamilySize"]==1).astype(int)
# title from name
df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].replace('Mlle', 'Miss')
df["Title"] = df["Title"].replace('Ms', 'Miss')
df["Title"] = df["Title"].replace('Mme', 'Mrs')

# Feature 4: Fare Bins (Discretizing Fare)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])
# qcut() means Quantile Cut.
# 👉 It splits your data equally into 4 parts (quartiles).
# 👉 In simple words, it divides the Fare column into 4 categories i.e. lowest(0),low-medium(1),medium-high(2),highest(3)
# since fare has continuous value with no pattern so Ml gets confused

# 👉 The model now understands:

# 💀 Rich people = High Survival.
# 💀 Poor people = Low Survival.


# Fill Missing Age with Median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill Missing Fare with Median
df["Fare"].fillna(df["Fare"].median(), inplace=True)
#Split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Train Model
model=XGBClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)


# Predict
y_pred=model.predict(x_test)

# Evaluate
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

# Using the model to predict survival on test.csv and generate a new file submission.csv
test_data=pd.read_csv(r"C:/Users/KIIT0001/titanic-ml-project/test.csv")

# Map the 'Sex' column in test_data (just like we did for training data)
test_data["Sex"] = test_data["Sex"].map({"male":0, "female":1})

# Drop unnecessary columns from test data
test_data = test_data[["Pclass","Sex","Age","Fare"]]

# Make predictions
predictions = model.predict(test_data)

# 💯🔥 THIS TIME I'M KEEPING THE PassengerId SAFE!!
test_data_original = pd.read_csv(r"C:/Users/KIIT0001/titanic-ml-project/test.csv")

# Create a new dataframe with PassengerId and Survived
output = pd.DataFrame({
    'PassengerId': test_data_original['PassengerId'],  # THIS IS THE FIX!!
    'Survived': predictions
})

# Save the CSV file
output.to_csv('submission.csv', index=False)
print(" Submission file created successfully!")
