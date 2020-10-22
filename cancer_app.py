import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

st.title("PCA en el Dataset de Cancer de Mama de Wisconsing")

"""
App interactiva mostrando los beneficios de Principal Component Analysis
"""

@st.cache
def load_data():
	df = pd.read_csv('breast.csv')
	df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
	df.diagnosis = pd.Categorical(df.diagnosis)
	df.diagnosis = df.diagnosis.cat.codes
	return df

@st.cache
def get_pairplot(data):
	fig, ax = plt.subplots(figsize=(20,8))
	sns.pairplot(data=data, hue='diagnosis', palette='Set2', height=1.5)
	# plt.show()
	return fig

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Cargando los datos...')
data = load_data()

# Notify the reader that the data was successfully loaded.
data_load_state.text('Cargando los datos... listo!')

st.subheader('Raw data')
st.write(data.head())

# st.subheader('Boxplots')
# fig, ax = plt.subplots(figsize=(20,8))
# fig.figure(figsize=(16,5))
# sns.boxplot(data=data.iloc[:, :14])
# st.pyplot(fig)

X = data.drop('diagnosis', axis=1)
y = data.diagnosis

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=1)

scaler = StandardScaler()
xtrain_scal = scaler.fit_transform(xtrain)
xtest_scal = scaler.transform(xtest)

st.text("Selecciona entre PCA y Default Features")
sel = st.radio("",('Default', 'PCA'))

if sel == "Default":
	data_load_state = st.text('Entrenando el modelo...')
	m = RandomForestClassifier(random_state=1)
	m.fit(xtrain_scal, ytrain)
	data_load_state.text('Entrenando el modelo... listo!')
	metric = f1_score(ytest, m.predict(xtest_scal), pos_label=1)

else:
	n_comps = st.slider('n_components', 1, xtrain.shape[1], xtrain.shape[1]//2)

	pca = PCA(n_components=n_comps)
	train_pca = pca.fit_transform(xtrain_scal)
	test_pca = pca.transform(xtest_scal)

	data_load_state = st.text('Entrenando el modelo...')
	m = RandomForestClassifier(random_state=1)
	m.fit(train_pca, ytrain)
	data_load_state.text('Entrenando el modelo... listo!')
	metric = f1_score(ytest, m.predict(test_pca), pos_label=1)

st.subheader('Resultados')
st.text(f'F1: {metric}')