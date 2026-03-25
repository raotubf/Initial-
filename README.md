import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
st.header("tip prediction")
dff = sns.load_dataset("tips")
df = sns.load_dataset("tips")

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['day'] = le.fit_transform(df['day'])
df['time'] = le.fit_transform(df['time'])

X = df.drop('tip', axis=1)
y = df['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("Restaurant Tip Recommendation System")
st.subheader("Dataset Preview")
st.write(dff.head())

st.sidebar.header("Select Algorithm")
algo = st.sidebar.selectbox("Algorithm", ["LinearRegression", "DecisionTree", "RandomForest"])

if algo == "LinearRegression":
    model = LinearRegression()
elif algo == "DecisionTree":
    model = DecisionTreeRegressor()
else:
    model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

st.subheader("Enter Customer Details")
col_inp1, col_inp2 = st.columns(2)

with col_inp1:
    bill = st.number_input("Total Bill")
    person = st.slider("Select Person")
    day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
    time = st.selectbox("Time", ["Lunch", "Dinner"])

if st.button("Predict Tip"):
    dayy = ["Thur", "Fri", "Sat", "Sun"].index(day)
    timee = ["Lunch", "Dinner"].index(time)
    
    input_data = np.array([[bill, 0, 0, dayy, timee, person]])
    prediction = model.predict(input_data)
    
    st.success(f"Recommended Tip: {prediction[0]:.2f}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mae:.4f}")
    m2.metric("R2 Score", f"{r2:.4f}")

fig, ax = plt.subplots()
sns.scatterplot(data=df, x="total_bill", y="tip", ax=ax)
st.pyplot(fig)
