# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:01:09 2022

@author: elena
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle 
st.title("Iowa property sale prices")

url = r"https://raw.githubusercontent.com/sydneyproject000/dat-sample/main/iowa_mini.csv"

num_rows = st.sidebar.number_input("Select Number of Rows to Load", min_value = 1000, max_value=5000, step=1000)


section = st.sidebar.radio("Choose Application Section", ['Data Explorer', 'Model Explorer'])

print(section)

@st.cache
def load_data(num_rows):
    df=pd.read_csv(url,nrows=num_rows)
    return df
@st.cache
def create_grouping(x_axis,y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df=load_data(num_rows)
    
if section == 'Data Explorer':
    
    
    x_axis = st.sidebar.selectbox("Choose column for x_axis",
                                  df.select_dtypes(include=np.object).columns.tolist())

    y_axis = st.sidebar.selectbox("Choose column for y_axis",
                                  ['SalePrice'])
    
    chart_type = st.sidebar.selectbox("Choose your chart type",['line','bar','area'])
    
    if chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
   
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
        
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    
    st.write(df)

else:
    st.text("Choose Options to the Side to Explore the Model")

    model = load_model()
    
    id_val = st.sidebar.selectbox("Choose Property ID", df['Id'].unique().tolist())
     
    OverallQual_val = st.sidebar.number_input("Overall Quality Score", min_value=1, max_value=10, step=1)
    
    Neighborhood_val = st.sidebar.selectbox("Neighborhood", df['Neighborhood'].unique().tolist())
    sample = {'Id':id_val,'OverallQual': OverallQual_val,'Neighborhood': Neighborhood_val}
    sample=pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]
    
    st.title(f"Predicted Sale Price:{int(prediction)}")