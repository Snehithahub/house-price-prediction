#reading data of house price
import pandas as pd
house=pd.read_csv(r"C:\Users\Chandra\Downloads\archive (2)\data.csv")
print(house.info())
print(house.describe())
print(house.head())
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

house['date']=pd.to_datetime(house['date'])
house['year'] = house['date'].dt.year
house['month'] = house['date'].dt.month
house['day_of_week'] = house['date'].dt.dayofweek
house.drop(['date'], axis=1, inplace=True)

nc=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','sqft_above','sqft_basement', 'yr_built', 'yr_renovated']
cc=['waterfront','view','condition','street', 'city','statezip', 'country']

scalar=MinMaxScaler()
for row in nc:
    house[nc]=scalar.fit_transform(house[nc])
print(house['bedrooms'])

for col in cc:
    house[col]=LabelEncoder().fit_transform(house[col])

print(house['view'])
print(house.corr())
print(house)
print(house.columns)

x=house.drop(['price','statezip','month','year','day_of_week'],axis=1)
y=house.price

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7,random_state=1)

from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x_train,y_train)
y_pred=model.predict(x_test)

from catboost import CatBoostRegressor
cb_model = CatBoostRegressor()
cb_model.fit(x_train, y_train)
preds = cb_model.predict(x_test) 
 


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
cb_r2_score=r2_score(Y_valid, preds)
print(cb_r2_score)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
mse=mean_squared_error(y_test,y_pred)
print(f"mean squared error :{mse} ")

import math
rmse=math.sqrt(mse)
print("rsme:",rmse)

r2=r2_score(y_test,y_pred)
print(f"r2 score: {r2}")


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.scatterplot(x=y_test,y=y_pred)
plt.show()
