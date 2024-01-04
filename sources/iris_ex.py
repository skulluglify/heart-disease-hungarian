import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

iris = load_iris()

target_names = iris.target_names
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.DataFrame(data=iris.data, columns=feature_names)

y = []
for target in iris.target: y.append(target_names[target])

df['target'] = y

X = df.iloc[:, :-1]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X, y)

# STREAMLIT
st.set_page_config(
  page_title = "Iris Classification",
  page_icon = ":sunflower:"
)

st.title("Iris Classification")
st.write("Using Decision Tree Classifier")

sepal_length = st.number_input(label="Sepal Length", min_value=df['sepal_length'].min(), max_value=df['sepal_length'].max())
sepal_width = st.number_input(label="Sepal Width", min_value=df['sepal_width'].min(), max_value=df['sepal_width'].max())
petal_length = st.number_input(label="Petal Length", min_value=df['petal_length'].min(), max_value=df['petal_length'].max())
petal_width = st.number_input(label="Petal Width", min_value=df['petal_width'].min(), max_value=df['petal_width'].max())

predict_btn = st.button("Predict", type="primary")

prediction = ":violet[-]"

if predict_btn:
  inputs = [[sepal_length, sepal_width, petal_length, petal_length]]
  prediction = model.predict(inputs)[0]

st.write("")
st.write("")
st.subheader("Prediction:")
st.subheader(prediction)