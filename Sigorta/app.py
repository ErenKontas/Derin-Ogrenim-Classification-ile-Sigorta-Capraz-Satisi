import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

df = pd.read_csv('train.csv')

x = df.drop(['id', 'Response'], axis=1)
y = df[['Response']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age','Driving_License','Region_Code','Previously_Insured','Annual_Premium','Policy_Sales_Channel','Vintage']),
        ('cat', OneHotEncoder(), ['Gender','Vehicle_Age','Vehicle_Damage'])
    ]
)


# Prediction function
def mantar_pred(Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage):
    input_data = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'Driving_License': [Driving_License],
        'Region_Code': [Region_Code],
        'Previously_Insured': [Previously_Insured],
        'Vehicle_Age': [Vehicle_Age],
        'Vehicle_Damage': [Vehicle_Damage],
        'Annual_Premium': [Annual_Premium],
        'Policy_Sales_Channel': [Policy_Sales_Channel],
        'Vintage': [Vintage]
    })
    
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('Sigorta.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

# Streamlit UI
def main():
    st.title("Sigorta Poliçesi Satış Tahmini")
    st.write("Veri Gir")

    Gender = st.selectbox('Gender', df['Gender'].unique())
    Age = st.slider('Age', int(df['Age'].min()), int(df['Age'].max()))
    Driving_License = st.selectbox('Driving_License', df['Driving_License'].unique())
    Region_Code = st.selectbox('Region_Code', df['Region_Code'].unique())
    Previously_Insured = st.selectbox('Previously_Insured', df['Previously_Insured'].unique())
    Vehicle_Age = st.selectbox('Vehicle_Age', df['Vehicle_Age'].unique())
    Vehicle_Damage = st.selectbox('Vehicle_Damage', df['Vehicle_Damage'].unique())
    Annual_Premium = st.slider('Annual_Premium', float(df['Annual_Premium'].min()), float(df['Annual_Premium'].max()))
    Policy_Sales_Channel = st.selectbox('Policy_Sales_Channel', df['Policy_Sales_Channel'].unique())
    Vintage = st.slider('Vintage', int(df['Vintage'].min()), int(df['Vintage'].max()))

    if st.button('Predict'):
        result = mantar_pred(Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage)
        st.write(f'The predicted result is: {result:.2f}')

if __name__ == '__main__':
    main()
