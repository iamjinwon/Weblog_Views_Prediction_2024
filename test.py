# from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

x_train = pd.read_parquet("./x_train.parquet")
y_train = pd.read_parquet("./y_train.parquet")

standard_scaler = preprocessing.StandardScaler()
x_train_scaled = standard_scaler.fit_transform(x_train)

X_train, X_val, y_train, y_val = train_test_split(x_train_scaled, y_train, test_size=0.33)

# # XGBRegressor 모델 초기화
# model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=32, n_jobs=1)

# model.fit(X_train, y_train, 
#           eval_set=[(X_val, y_val)], 
#           early_stopping_rounds=10)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

y_train = y_train.iloc[:, 0].values
y_val = y_val.iloc[:, 0].values

# 기본 모델 훈련
model_lgbm = LGBMRegressor(n_jobs=1)
model_rf = RandomForestRegressor(n_jobs=1)

model_lgbm.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# 기본 모델 예측
preds_lgbm = model_lgbm.predict(X_val)
preds_rf = model_rf.predict(X_val)

# 메타 모델을 위한 새로운 특성 생성
X_meta = np.column_stack((preds_lgbm, preds_rf))

# 메타 모델 훈련
model_meta = LinearRegression()
model_meta.fit(X_meta, y_val)

# 메타 모델 예측
final_preds = model_meta.predict(X_meta)

# 성능 평가
mse = mean_squared_error(y_val, final_preds)
rmse = np.sqrt(mse)  
print(f"Root Mean Squared Error: {rmse}")