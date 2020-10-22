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

@st.cache
def get_pairplot(data):
	fig, ax = plt.subplots(figsize=(20,8))
	sns.pairplot(data=data, hue='diagnosis', palette='Set2', height=1.5)
	# plt.show()
	return fig

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
data = load_data()

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(data.head())

st.subheader('Boxplots')
fig, ax = plt.subplots(figsize=(20,8))
# fig.figure(figsize=(16,5))
sns.boxplot(data=data.iloc[:, :14])
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(20,8))
sns.boxplot(data=data.iloc[:, 14:])
st.pyplot(fig)

data_load_state = st.text('Loading data...')
# st.subheader('Pairplot')
# plot = get_pairplot(data.iloc[:,:3])
# st.pyplot(fig)
data_load_state.text('Loading data...done!')