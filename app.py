# scaler is exported as scaler.pkl
# model is exported as co2_emission_best_model.pkl

# import the necessary liraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('CO2_emission_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the trained scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title of the app
st.title("CO2 Emissions Prediction")

# Load the list of countries from a CSV file
filtered_df = pd.read_csv('data.csv')  

#filtered_df = data_df[data_df['Some_Column'] == 'Some_Value']
countries = filtered_df['Entity'].unique().tolist()

# Input features
st.header("Model Deployment")
entity = st.selectbox("Entity", options=countries)  # Replace with actual entities
year = st.number_input("Year", min_value=1980, max_value=2030, value=2000)
elec_generated = st.number_input("Elec_Generated", min_value=0.0, value=1000.0)
renewable_energy = st.number_input("Renewable_Energy",  min_value=0.0, value=5000.0)
non_renewable_energy = st.number_input("Non_Renewable_Energy",  min_value=0.0, value=15000.0)
Energy_Consumption_Per_Capita = st.number_input("Energy_Consumption_Per_Capita",  min_value=0.0, value=1.0)
GDP_Per_Capita = st.number_input("GDP_Per_Capita",  min_value=0.0, value=100000.0)
pry_energy_consumption = st.number_input("Pry_Energy_Consumption", min_value=0.0, value=1000.0)
population = st.number_input("Population", min_value=0, value=10000000)
gdp = st.number_input("GDP", min_value=0.0, value=100000000.0)

# Create a DataFrame for the input features
input_data = np.array([[renewable_energy, non_renewable_energy, Energy_Consumption_Per_Capita,
    GDP_Per_Capita, elec_generated, pry_energy_consumption, population, gdp]])

# Scale the input features
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict CO2 Emissions"):
    prediction = model.predict(input_data_scaled)

    # Reverse log transformation
    predictions = np.expm1(prediction[0])  # Reverse log transformation

    st.balloons()
    
    st.success(f"Predicted CO2 Emissions: {predictions/1000000:.2f} Mil tons")
