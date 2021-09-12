import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image

model=pickle.load(open('model_knn.pkl','rb'))

st.title('ODI Total Runs Prediction')
st.sidebar.header('Match Data')
image = Image.open('bb.jpeg')
st.image(image, '')

def user_report():
  ball  = st.sidebar.slider('Over ', 10,50, 10 )
  current_run = st.sidebar.slider('current_run', 0,500, 1 )
  wicket  = st.sidebar.slider('wicket ', 0,10, 0 )
  striker_run = st.sidebar.slider('striker_run', 0,300, 0 )
  nonstriker_run = st.sidebar.slider('nonstriker_run', 0,300, 0 )
  runs_last10 = st.sidebar.slider('runs_last10', 0,200, 0)
  wicket_last10 = st.sidebar.slider('wicket_last10', 0,10, 0)


  user_report_data = {
      'Over ':ball ,
      'current_run':current_run,
      'wicket ':wicket ,
      'striker_run':striker_run,
      'nonstriker_run':nonstriker_run,
      'runs_last10':runs_last10,
      'wicket_last10':wicket_last10,
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Match Data')
st.write(user_data)

#if st.button("Predict"):
total_runs = model.predict(user_data)
st.subheader('Predicted Total Runs:')
st.subheader(np.round(total_runs[0], 0))