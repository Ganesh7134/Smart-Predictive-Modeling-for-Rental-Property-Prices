import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import json

df = pd.read_csv("House_rent_data.csv")

container = st.container()
try:
    with container:
        @st.cache_data(ttl=60 * 60)
        def load_lottie_file(filepath : str):
            with open(filepath, "r") as f:
                gif = json.load(f)
            st_lottie(gif, speed=1, width=650, height=450)
            
        load_lottie_file("house.json")
except:
    print("Don't raise exception")

st.title("Smart Predictive Modeling for Rental  Property Prices.")
st.warning("some columns selections having 0 and 1 means **no** and **yes** respectively.")



with st.sidebar:
    locality = st.selectbox("select loction:",df["locality"].unique(),key=30)
    activation_date = st.selectbox("select activation_date:",df["activation_date"].unique(),key=31)
    type = st.selectbox("select property type : ",df["type"].unique(),key=1)
    lease_type = st.selectbox("select lease type : ",df["lease_type"].unique(),key=2)
    gym = st.selectbox("select gym status :",df["gym"].unique(),key=3)
    lift = st.selectbox("select lift status :",df["lift"].unique(),key=4)
    swimming = st.selectbox("select swimming_pool status :",df["swimming_pool"].unique(),key=5)
    negotiate = st.selectbox("select negotiate status :",df["negotiable"].unique(),key=6)
    furnish = st.selectbox("select furnishing category:",df["furnishing"].unique(),key=7)
    parking = st.selectbox("select parking category:",df["parking"].unique(),key=8)
    facing = st.selectbox("select direction of Rental house:",df["facing"].unique(),key=9)
    water = st.selectbox("select water source:",df["water_supply"].unique(),key=10)
    building = st.selectbox("select building type:",df["building_type"].unique(),key=11)
    security = st.selectbox("select security status:",df["security"].unique(),key=12)
    park = st.selectbox("select park status:",df["park"].unique(),key=13)
    Ac = st.selectbox("select Ac status:",df["Ac"].unique(),key=14)
    property_size = st.text_input("Enter property_size:",key=15)
    st.warning(f"**min_value** = {df['property_size'].min()} and **max_value**  ={df['property_size'].max()}")
    property_age = st.text_input("Enter property_age:",key=16)
    st.warning(f"**min_value** = {df['property_age'].min()} and **max_value**  ={df['property_age'].max()}")
    bathroom = st.text_input("Enter bathroom_count:",key=17)
    st.warning(f"**min_value** = {df['bathroom'].min()} and **max_value**  ={df['bathroom'].max()}")
    cup_board = st.text_input("Enter cup_board_count:",key=18)
    st.warning(f"**min_value** = {df['cup_board'].min()} and **max_value**  ={df['cup_board'].max()}")
    lower_floor = st.text_input("Enter lower_floor_count:",key=19)
    st.warning(f"**min_value** = {df['floor'].min()} and **max_value**  ={df['floor'].max()}")
    higher_floor = st.text_input("Enter higher_floor_count:",key=20)
    st.warning(f"**min_value** = {df['total_floor'].min()} and **max_value**  ={df['total_floor'].max()}")
    balcnies = st.text_input("Enter balcony_count:",key=21)
    st.warning(f"**min_value** = {df['balconies'].min()} and **max_value**  ={df['balconies'].max()}")
    latitude = st.selectbox("select any latitude values in selected location:",df.loc[df["locality"] == locality]["latitude"].unique())
    longitude = st.selectbox("select any longitude values in selected location:",df.loc[df["locality"] == locality]["longitude"].unique())

def result():
    with open('randomforest_model.pkl', 'rb') as file:
        rfr = pickle.load(file)
    with open('rent_scale.pkl', 'rb') as f:
        ss = pickle.load(f)
    with open("type_ohe.pkl", "rb") as f:
        Ohe1 = pickle.load(f)
    with open("lease_ohe.pkl", "rb") as f:
        Ohe2 = pickle.load(f)
    with open("gym_ohe.pkl", "rb") as f:
        Ohe3 = pickle.load(f)
    with open("lift_ohe.pkl", "rb") as f:
        Ohe4 = pickle.load(f)
    with open("swimming_ohe.pkl", "rb") as f:
        Ohe5 = pickle.load(f)
    with open("negotiate_ohe.pkl", "rb") as f:
        Ohe6 = pickle.load(f)
    with open("furnish_ohe.pkl", "rb") as f:
        Ohe7 = pickle.load(f)
    with open("parking_ohe.pkl", "rb") as f:
        Ohe8 = pickle.load(f)
    with open("facing_ohe.pkl", "rb") as f:
        Ohe9 = pickle.load(f)
    with open("water_ohe.pkl", "rb") as f:
        Ohe10 = pickle.load(f)
    with open("building_ohe.pkl", "rb") as f:
        Ohe11 = pickle.load(f)
    with open("security_ohe.pkl", "rb") as f:
        Ohe12 = pickle.load(f)
    with open("park_ohe.pkl", "rb") as f:
        Ohe13 = pickle.load(f)
    with open("Ac_ohe.pkl", "rb") as f:
        Ohe14 = pickle.load(f)
    with open("locality_le.pkl", "rb") as f:
        le1 = pickle.load(f)
    with open("ac_date_le.pkl", "rb") as f:
        le2 = pickle.load(f)

    new_test = np.array([[np.log(float(property_size)),np.log(float(property_age)),np.log(float(bathroom)),np.log(float(cup_board)),np.log(float(lower_floor)),np.log(float(higher_floor)),np.log(float(balcnies)),latitude,longitude,type,locality,activation_date,lease_type,gym,lift,swimming,negotiate,furnish,parking,facing,water,building,security,park,Ac]])

    new_test_type = Ohe1.transform(new_test[:, [9]]).toarray()

    new_test_locality = le1.transform(new_test[:,[10]])

    # Reshape the new_test_locality array
    new_test_locality = new_test_locality.reshape(-1,1)


    new_test_ac_date = le2.transform(new_test[:,[11]])

    new_test_ac_date = new_test_ac_date.reshape(-1,1)

    new_test_lease = Ohe2.transform(new_test[:, [12]]).toarray()

    new_test_gym = Ohe3.transform(new_test[:, [13]]).toarray()

    new_test_lift = Ohe4.transform(new_test[:, [14]]).toarray()

    new_test_swimming = Ohe5.transform(new_test[:, [15]]).toarray()

    new_test_negotiate = Ohe6.transform(new_test[:, [16]]).toarray()

    new_test_furnish = Ohe7.transform(new_test[:, [17]]).toarray()

    new_test_parking = Ohe8.transform(new_test[:, [18]]).toarray()

    new_test_facing = Ohe9.transform(new_test[:, [19]]).toarray()

    new_test_water = Ohe10.transform(new_test[:, [20]]).toarray()

    new_test_building = Ohe11.transform(new_test[:, [21]]).toarray()

    new_test_security = Ohe12.transform(new_test[:, [22]]).toarray()

    new_test_park = Ohe13.transform(new_test[:, [23]]).toarray()

    new_test_Ac = Ohe14.transform(new_test[:, [24]]).toarray()

    new_test = np.concatenate((new_test[:, [0,1,2,3,4,5,6,7,8]], new_test_type , new_test_locality , new_test_ac_date , new_test_lease , new_test_gym , new_test_lift , new_test_swimming , new_test_negotiate , new_test_furnish , new_test_parking , new_test_facing , new_test_water,new_test_building,new_test_security,new_test_park,new_test_Ac), axis=1)

    # print(new_test[0].shape)


    new_test1 = ss.transform(new_test)
    new_pred = rfr.predict(new_test1)
    st.title(f'Rental property price Rs: :blue[{int(np.exp(new_pred))}/-]')

button = st.button("preducting rental property price",type="primary",use_container_width=True)

if button:
    result()
