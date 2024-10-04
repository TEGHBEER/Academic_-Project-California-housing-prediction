import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error,mean_absolute_error


#loasd the dataaset
cal= fetch_california_housing()
df=pd.DataFrame(cal.data, columns=(cal.feature_names))
df["price"]= cal.target
df.head()

#title of the app
st.title("California House Price Prediction for XYZ Brokerage Company")


# data overview
st.subheader("Data Overview")
st.dataframe(df.head(10))

#train tesr split
X= df.drop("price", axis=1)
y=df["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
                                               
# strandize the data
scaler= StandardScaler()
X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)

# Model Selection
st.subheader(" ## Select a Model")

model= st.selectbox("Choose a model", ["LinearRegression","Ridge", "Lasso", "ElasticNet"])

# Intialize a model

models= {"LinearRegression" :LinearRegression(),
         "Ridge":Ridge(),
         "Lasso": Lasso(),
         "ElasticNet": ElasticNet(alpha=0.001) }

# train the selected dmodel
selectde_model=models[model]

# train the model
selectde_model.fit(X_train_sc,y_train)

#predict the value
y_pred=selectde_model.predict(X_test_sc)

# evaluate the model using metrics
test_mse=mean_squared_error(y_test, y_pred)
test_mae=mean_absolute_error(y_test, y_pred)
test_rmse=np.sqrt(test_mse)
test_r2=r2_score(y_test,y_pred)

# display the metricd for selected model
st.write("Test MSE", test_mse)
st.write("Test MAE", test_mae)
st.write("Test RMSE", test_rmse)
st.write("Test R2", test_r2)


# prompt to enter input values
st.write("Enter the input value to predict the ouse price:")

user_input= {}

for feature in X.columns:
    user_input[feature]= st.number_input(feature)

# convert the dictionary to dataframe
user_input_df= pd.DataFrame([user_input] ) 

# scale the user input
user_input_sc= scaler.transform(user_input_df)

# predict the house price
predicted_price= selectde_model.predict(user_input_sc)

# display the predicted price
st.write(f"predicted house price is {predicted_price[0]*100000}")