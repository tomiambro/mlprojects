import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.title("PCA in Breast Cancer Dataset")

"""
Interactive app to show the benefits of using Principal Component Analysis
"""

@st.cache
def load_data():
	df = pd.read_csv('breast.csv')
	df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
	return df

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
data = load_data()

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.subheader('Columns')
st.write(data.columns)

st.subheader('Raw data')
st.write(data.head())

fig, ax = plt.subplots()
sns.boxplot(data=data)
st.pyplot(fig)
