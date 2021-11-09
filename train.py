import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("hiring1.csv")
print(dataset)

dataset.experience.fillna(0,inplace=True)
dataset.test_score.fillna(dataset.test_score.mean(),inplace=True)

X=dataset.iloc[:,:3]
y=dataset.salary

def convert_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11, 0:0}
    return word_dict[word]

X['experience']=X.experience.apply(lambda x:convert_to_int(x))
regressor=LinearRegression()
regressor.fit(X,y)

print('model_training is done')

print(regressor.predict([[2,9,8]]))
joblib.dump(regressor,'hiringmodel1.pkl')
