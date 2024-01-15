import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

#原始資料
NBApoints = pd.read_csv(r"D:\GitHub\Scikit_Learn-NBA\CSV\NBApoints.csv")
#print('原始資料')
#NBApoints.info()

# 1. 將 Pos 和 Tm 欄位轉換為數值型態
label_encoder = preprocessing.LabelEncoder()
encoded_Pos = label_encoder.fit_transform(NBApoints['Pos'])
encoded_Tm = label_encoder.fit_transform(NBApoints['Tm'])

X =pd.DataFrame([encoded_Pos, NBApoints['Age'], encoded_Tm]).T
y = NBApoints['3P']

lm = LinearRegression()
lm.fit(X, y)
y_Train = lm.predict(X)

test_x = [5, 28, 10]

test_data = pd.DataFrame(np.array(test_x)).T
predicted_3P= lm.predict(test_data)
print(f'{test_x}預測之三分球得球數:{round(predicted_3P[0],4)}')

print("MAE： ", mae(y, y_Train))
print("MSE： ", mse(y, y_Train))
print("RMSE： ", mse(y, y_Train) ** 0.5)
print("R-squared： ", lm.score(X, y))
