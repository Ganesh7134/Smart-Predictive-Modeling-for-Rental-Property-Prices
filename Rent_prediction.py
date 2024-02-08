import streamlit as st
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("House_rent_data.csv")
\
st.title("Smart Predictive Modeling for Rental  Property Prices.")
st.warning("some columns selections having 0 and 1 means **no** and **yes** respectively.")
col1 , col2 = st.columns(2)


with col1:
    type = st.selectbox("type of the house", df["type"].unique(),key=2)
    lift = st.selectbox("select you want lift or not: ", df["lift"].unique(),key = 3)
    swimming = st.selectbox("select you want swimming pool or not: ", df["swimming_pool"].unique(),key = 4)
    parking = st.selectbox("select which parking facility do you want : ", df["parking"].unique(),key = 5)
    building = st.selectbox("select which type of building you want : ", df["building_type"].unique(),key = 6)
    security = st.selectbox("select security facility or not: ", df["security"].unique(),key = 7)
    lease_type = st.selectbox("select lease type: ", df["lease_type"].unique(),key = 8)
    locality = st.selectbox("select locality: ", df["locality"].unique(),key = 9)
    furnishing = st.selectbox("select furnishing type: ", df["furnishing"].unique(),key = 10)
    water_type = st.selectbox("select water facility: ", df["water_supply"].unique(),key = 11)
    facing = st.selectbox("select facing: ", df["facing"].unique(),key = 12)
    negotiable = st.selectbox("select negotiable status: ", df["negotiable"].unique(),key = 13)
    latitude = st.slider("select latitude: ", min_value = df["latitude"].min() , max_value = df["latitude"].max() , key=19)
with col2:
    property_size = st.text_input("Enter the property size: ", key = 1)
    st.warning(f"**min value** : {df['property_size'].min()} and **max_value**: {df['property_size'].max()}")
    property_age = st.text_input("Enter property age :",key=14)
    st.warning(f"**min value** : {df['property_age'].min()} and **max_value**: {df['property_age'].max()}")
    bathrooms = st.text_input("Enter bathrooms count :",key=15)
    st.warning(f"**min value** : {df['bathroom'].min()} and **max_value**: {df['bathroom'].max()}")
    cup_board = st.text_input("Enter cup_board count :",key=16)
    st.warning(f"**min value** : {df['cup_board'].min()} and **max_value**: {df['cup_board'].max()}")
    floor = st.text_input("Enter floor count :",key=17)
    st.warning(f"**min value** : {df['total_floor'].min()} and **max_value**: {df['total_floor'].max()}")
    balconies = st.text_input("Enter how many balconies do you want : ",key=18)
    st.warning(f"**min value** : {df['balconies'].min()} and **max_value**: {df['balconies'].max()}")
    longitude = st.slider("select longitude: ", min_value = df["longitude"].min() , max_value = df["longitude"].max() , key=20)

def result():
    # Load Random Forest model
    with open('randomforest_model.pkl', 'rb') as file:
        rfr = pickle.load(file)

    # Load scaler
    with open('Standard_scaler.pkl', 'rb') as f:
        ss = pickle.load(f)

    # Load OneHotEncoders
    with open('type.pkl', 'rb') as f:
        Ohe1 = pickle.load(f)

    with open('lift.pkl', 'rb') as f:
        Ohe2 = pickle.load(f)

    with open('swimming.pkl', 'rb') as f:
        Ohe3 = pickle.load(f)

    with open('parking.pkl', 'rb') as f:
        Ohe4 = pickle.load(f)

    with open('building.pkl', 'rb') as f:
        Ohe5 = pickle.load(f)

    with open('security.pkl', 'rb') as f:
        Ohe6 = pickle.load(f)

    with open('lease_type.pkl', 'rb') as f:
        Ohe7 = pickle.load(f)

    with open('locality.pkl', 'rb') as f:
        Ohe8 = pickle.load(f)

    with open('furnishing.pkl', 'rb') as f:
        Ohe9 = pickle.load(f)

    with open('water.pkl', 'rb') as f:
        Ohe10 = pickle.load(f)

    with open('facing.pkl', 'rb') as f:
        Ohe11 = pickle.load(f)

    with open('negotiable.pkl', 'rb') as f:
        Ohe12 = pickle.load(f)
    new_test = np.array([[np.log(float(property_size)),np.log(float(property_age)),np.log(float(bathrooms)),np.log(float(cup_board)),np.log(float(floor)),np.log(float(balconies)),latitude,longitude,type,lift,swimming,parking,building,security,lease_type,locality,furnishing,water_type,facing,negotiable]])

    new_test_type = Ohe1.transform(new_test[:, [8]]).toarray()
    new_test_lift = Ohe2.transform(new_test[:, [9]]).toarray()
    new_test_swimming = Ohe3.transform(new_test[:,[10]]).toarray()
    new_test_parking = Ohe4.transform(new_test[:,[11]]).toarray()
    new_test_building = Ohe5.transform(new_test[:,[12]]).toarray()
    new_test_security = Ohe6.transform(new_test[:,[13]]).toarray()
    new_test_lease = Ohe7.transform(new_test[:,[14]]).toarray()
    new_test_local = Ohe8.transform(new_test[:,[15]]).toarray()
    new_test_furnish = Ohe9.transform(new_test[:,[16]]).toarray()
    new_test_water = Ohe10.transform(new_test[:,[17]]).toarray()
    new_test_facing = Ohe11.transform(new_test[:,[18]]).toarray()
    new_test_negotiate = Ohe12.transform(new_test[:,[19]]).toarray()



    new_test = np.concatenate((new_test[:, [0,1,2,3,4,5,6,7]], new_test_type , new_test_lift , new_test_swimming , new_test_parking , new_test_building , new_test_security , new_test_lease , new_test_local , new_test_furnish , new_test_water , new_test_facing , new_test_negotiate), axis=1)
    # print(new_test[0].shape)
    new_test1 = ss.transform(new_test)
    new_pred = rfr.predict(new_test1)
    st.write(f'## :Orange[Rental property price: {np.exp(new_pred)}]')

button = st.button("preducting rental property price",type="primary",use_container_width=True)

if button:
    result()