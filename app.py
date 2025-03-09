import streamlit as st
import pandas as pd
import pickle
# deploying the web app


#load your trained model
model=pickle.load(open('titanic_model.pkl','rb'))

#app title
st.title('Titanic Survival Prediction App')

#file uploader
uploaded_file=st.file_uploader("Upload your test.csv file",type=["csv"])

if uploaded_file is not None:
    #Read the uploaded CSV file
    data=pd.read_csv(uploaded_file)

    # Preprocessing (same as what we did while training)
    data['Sex']=data['Sex'].map({'male': 0, 'female': 1})

    # Selecting the same features as we trained on
    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    #Predict the survival
    predictions=model.predict(x)

    #Add the prediction to the data
    data['Survived Prediction']=predictions

    #Show the results
    st.write("Here are the predictions:**")
    st.dataframe(data)


    #Option to download the results
    csv=data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions",data=csv,file_name="predictions.csv")