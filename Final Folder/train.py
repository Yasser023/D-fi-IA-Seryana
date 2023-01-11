import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
hotels = pd.read_csv('features_hotels.csv', index_col=['hotel_id', 'city'])
#sample = pd.read_csv('sample_submission.csv')
test_set = pd.read_csv('test_set.csv')
#df1 = pd.read_csv('dataset_1.csv')
#df2 = pd.read_csv('dataset_2.csv')
#df3 = pd.read_csv('dataset_3.csv')
#df4 = pd.read_csv('dataset_4.csv')
#df5 = pd.read_csv('dataset_5.csv')
#df6 = pd.read_csv('dataset_6.csv')
#df1000_1 = pd.read_csv('dataset_1000_1.csv')
#df1000_2 = pd.read_csv('dataset_1000_2.csv')
#df1000_3 = pd.read_csv('dataset_1000_3.csv')
#dfv2_1 = pd.read_csv('dataset_v2_1.csv')
#dfv2_2 = pd.read_csv('dataset_v2_2.csv')
#df_yanaser_1 = pd.read_csv('yanaser_1.csv')
#df_yanaser_2 = pd.read_csv('yanaser_2.csv')

#df = pd.concat([df1,df2,df3,df4,df5,df6])
#df = pd.concat([df1,df2,df3,df4,df5,df6,df1000_1,df1000_2,df1000_3,dfv2_1,dfv2_2])
               #,df_yanaser_1,df_yanaser_2])
df = pd.read_csv('trainset.csv')
df_final = df.join(hotels, on=['hotel_id', 'city'])
dataset = df_final.copy()
dataset = dataset.drop(['Unnamed: 0','hotel_id','avatar_id'], axis=1)
test_set_joined = test_set.join(hotels, on=['hotel_id', 'city'])
index = test_set_joined['index']
y = df_final['price']
X = df_final.drop(['Unnamed: 0','price'], axis=1)
X_test = test_set_joined[['hotel_id','stock','city','date','language','mobile', 'avatar_id','group','brand','parking','pool','children_policy']]
X_ids = X[['hotel_id','avatar_id']]
X_test_ids = X_test[['hotel_id','avatar_id']]
X = X.drop(['hotel_id','avatar_id'], axis=1)
X_test = X_test.drop(['hotel_id','avatar_id'], axis=1)


from category_encoders import TargetEncoder
df_encoder = df_final.drop(['Unnamed: 0','hotel_id','avatar_id'], axis=1)
encoder1 = TargetEncoder()
encoder2 = TargetEncoder()
encoder3 = TargetEncoder()
encoder4 = TargetEncoder()
encoder1.fit(df_encoder['city'],df_encoder['price'])
encoder2.fit(df_encoder['language'],df_encoder['price'])
encoder3.fit(df_encoder['group'],df_encoder['price'])
encoder4.fit(df_encoder['brand'],df_encoder['price'])
df_encoder['city_encoded'] = encoder1.transform(df_encoder['city'])
df_encoder['language_encoded'] = encoder2.transform(df_encoder['language'])
df_encoder['group_encoded'] = encoder3.transform(df_encoder['group'])
df_encoder['brand_encoded'] = encoder4.transform(df_encoder['brand'])

XX_test = X_test.copy()
XX_test['city_encoded']=encoder1.transform(XX_test['city'])
XX_test['language_encoded']=encoder2.transform(XX_test['language'])
XX_test['group_encoded']=encoder3.transform(XX_test['group'])
XX_test['brand_encoded']=encoder4.transform(XX_test['brand'])

correspondance = pd.DataFrame({'city':df_encoder['city'].unique(),'encoding':df_encoder['city_encoded'].unique()})

y = df_encoder.iloc[:, 0].values.reshape(-1,1)
df_train = df_encoder.drop(['Unnamed: 0.1','price','city','language','group','brand'], axis=1)
XX_test = XX_test.drop(['city','language','group','brand'], axis=1)
from xgboost import XGBRegressor
xgb_model = XGBRegressor()
#xgb_model.fit(df_train,y)
#filename = "model_xgb.joblib"
#joblib.dump(xgb_model, filename)