import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("C:\end_to_end_ml_project\Iris.csv")
df.head()

df1=df.iloc[:,1:]

x=df1.iloc[:,:-1]
y=df1["Species"]

le=LabelEncoder()
y= le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,train_size=0.3)

def evaluate_mode(true,predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mean_squared_error(true,predicted))
    r2_square=r2_score(true,predicted)
    return mae,mse,rmse,r2_square

#just to check for best model
'''
models={
    "linear Regression":LinearRegression(),
    "lasso":Lasso(),
    "Ridge":Ridge(),
    "K-Neighbors Regression":KNeighborsRegressor(),
    "Decision Tree":DecisionTreeRegressor(),
    "Random Forest Regressor":RandomForestRegressor(),
}
'''

'''model_list=[]
r2_list=[]
for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(x_train,y_train)
    
    #making prediction
    y_train_pred=model.predict(x_train)
    y_test_pred=model.predict(x_test)
    
    #evaluating train and test dataset
    
    model_train_mae,model_train_mse,model_train_rmse,model_train_r2= evaluate_mode(y_train,y_train_pred)
    model_test_mae,model_test_mse,model_test_rmse,model_test_r2= evaluate_mode(y_test,y_test_pred)
    
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    
    print('model performance for traing set')
    print("- rms error:{:.4f}".format(model_train_rmse))
    print("- mae: {:.4f}".format(model_train_mae))
    print("- r2 score:{:.4f}".format(model_train_r2))
    
    print("------------------------------------------")
    
    print('model performance for testing set')
    print("- rms error:{:.4f}".format(model_test_rmse))
    print("- mae: {:.4f}".format(model_test_mae))
    print("- r2 score:{:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    
    print('='*35)
    print('\n')
    '''
    
model=Ridge().fit(x_train,y_train)
y_pred=model.predict(x_test)
score=r2_score(y_test,y_pred)*100
#print("accuracy of model is=%.2f"%score)

labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def ml_model(a, b, c, d):
    x = np.array([a, b, c, d])  
    prediction = model.predict(x.reshape(1, -1)) 
    predicted_label = labels[int(prediction[0])]  
    return predicted_label 
