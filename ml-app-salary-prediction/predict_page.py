import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
label_encoding_country=data["label_encoding_country"]
label_encoding_education_level=data["label_encoding_education_level"]
label_encoding_gender=data["label_encoding_gender"]
label_encoding_job_title=data["label_encoding_job_title"]
label_encoding_race=data["label_encoding_race"]

def show_predict_page():
    st.title("Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "USA",
        "China",
        "Australia",
        "UK",
        "Canada",
    )
    races = (
        "White", 
        "Asian", 
        "Korean", 
        "Australian",
        "Chinese", 
        "Black" , 
        "African American", 
        "Mixed",
        "Welsh", 
        "Hispanic", 
    )

    education = (
        "High School",
        "Bachelor",
        "Master",
        "PhD",
    )
    
    job_title = (
        "Other" ,"Software Engineer", "Data Scientist", "Data Analyst", "Software Engineer Manager" ,"Product Manager",
          "Project Engineer" ,"Marketing Manager","Full Stack Engineer","Back end Developer","Front end Developer","Sales Associate",
          "Software Developer","Marketing Coordinator","Human Resources Manager","Marketing Analyst"
          ,"Financial Manager","Web Developer","Operations Manager","Research Scientist","HR Generalist",
    )
    gender = (
        "Female",
        "Male",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    race = st.selectbox("Race", races)
    gender = st.selectbox("Gender", gender)
    job_title = st.selectbox("Job Title", job_title)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[gender,education,job_title,expericence,country ,race, ]])
        X[:,0]=label_encoding_gender.transform(X[:,0])
        X[:,1]=label_encoding_education_level.transform(X[:,1])
        X[:,2]=label_encoding_job_title.transform(X[:,2])
        X[:,4]=label_encoding_country.transform(X[:,4])
        X[:,5]=label_encoding_race.transform(X[:,5])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
