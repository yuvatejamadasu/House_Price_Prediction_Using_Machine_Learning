'''
in this development file
we are developing a house price prediction using malti linear regression algo
'''

#import all the librarys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')


#create a class

class HOUSE_PRICE_PREDICTION:
    # create a construtor class for common things
     def __init__(self,data):

         try:
             #take the dataset
              self.df=pd.read_csv(data)

             #to see the column names
              col_nam=self.df.columns

             #to remove unwanted columns in the dataset (date & street)
              self.df=self.df.drop(['date','street','city'],axis=1)

             # in the independent columns street city and country are text so convert into a number using map method
             #to get the unique values of the street
              con_uni=self.df['country'].unique()

              self.df['country']=self.df['country'].map({
                  'USA':0
              }).astype(int)

             #in the independent column stateZip column has both text and number we split that two columns
             #state is one and zipcode in another number
              self.df['state']=self.df['statezip'].str.split().str[0]
              self.df['zipcode']=self.df['statezip'].str.split().str[1]
              #now we split that statezip in two columns state column and zip column after
             #we convert that into two columns delete the statezip column
              self.df=self.df.drop(['statezip'],axis=1)
             #now we convert state column into a number
             #first we get a unique column names in the state
              sta_uni=self.df['state'].unique()
              self.df['state']=self.df['state'].map({
                  'WA':0
              }).astype(int)
              print(self.df.columns)
              print(self.df)
             #now we put X ina independent columns and y in a dependent columns
              self.X=self.df.iloc[ : , 1 : ]
              self.y=self.df.iloc[ : , 0]
              # now we convert the data into a training part and testing part
              self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)


         except Exception as e:
             er_ty, er_msg, er_line=sys.exc_info()
             print(f"the error line was:{er_line} error msg is :{er_msg} error type is :{er_ty}")


     #Training Performance
     def train(self):
         try:
             #give data to algorithm
             # give data to algo
             self.reg = LinearRegression()
             self.reg.fit(self.x_train, self.y_train)
             # to get the m values
             print(f"the m values are:{self.reg.coef_}")
             print(f"the c values are :{self.reg.intercept_}")
             #to predict the training values
             print(f"the training prediction values are:")
             self.train_predict=self.reg.predict(self.x_train)

             print(f"the accuracy of the training data was")
             print(f"{r2_score(self.y_train,self.train_predict)}")
             print(f"the loss of the training data was")
             print(f"{root_mean_squared_error(self.y_train,self.train_predict)}")
         except Exception as e:
             er_ty, er_msg, er_line=sys.exc_info()
             print(f"the error line was:{er_line} error msg is :{er_msg} error type is :{er_ty}")

             # Training Performance

     def test(self):
         try:
             print(f"=====================================================================")
             # to predict the testing values
             print(f"the testing prediction values are:")
             self.test_predict = self.reg.predict(self.x_test)


             print(f"the accuracy of the testing data was")
             print(f"{r2_score(self.y_test, self.test_predict)}")
             print(f"the loss of the testing data was")
             print(f"{root_mean_squared_error(self.y_test, self.test_predict)}")
         except Exception as e:
             er_ty, er_msg, er_line = sys.exc_info()
             print(f"the error line was:{er_line} error msg is :{er_msg} error type is :{er_ty}")

      #own data prediction
     def own_data_point(self):
         try:
             bedrooms=4
             bathrooms=2
             sqft_living=1200
             sqft_lot=1298
             floors=1
             waterfront=1
             view=2
             condition=4
             sqft_above=1509
             sqft_basement=1100
             yr_built=2004
             yr_renovated=2020
             country=0
             state=0
             zipcode=48928

             result=self.reg.predict([[bedrooms,
             bathrooms,
             sqft_living,
             sqft_lot,
             floors,
             waterfront,
             view,
             condition,
             sqft_above,
             sqft_basement,
             yr_built,
             yr_renovated,
             country,
             state,
             zipcode]])

             print(result)

         except Exception as e:
             er_ty, er_msg, er_line = sys.exc_info()
             print(f"the error line was:{er_line} error msg is :{er_msg} error type is :{er_ty}")

     def saving(self):
         try:
             with open("House_Price_Prediction_Model.pkl", "wb") as f:
                 pickle.dump(self.reg, f)

             #now load the data

             with open("House_Price_Prediction_Model.pkl","rb") as p:
                 self.m=pickle.load(p)
         #checking model data prediction
             bedrooms = 4
             bathrooms = 2
             sqft_living = 1200
             sqft_lot = 1298
             floors = 1
             waterfront = 1
             view = 2
             condition = 4
             sqft_above = 1509
             sqft_basement = 1100
             yr_built = 2004
             yr_renovated = 2020
             country = 0
             state = 0
             zipcode = 48928

             model_result = self.m.predict([[bedrooms,
                                         bathrooms,
                                         sqft_living,
                                         sqft_lot,
                                         floors,
                                         waterfront,
                                         view,
                                         condition,
                                         sqft_above,
                                         sqft_basement,
                                         yr_built,
                                         yr_renovated,
                                         country,
                                         state,
                                         zipcode]])

             print(model_result)


         except Exception as e:
             er_ty, er_msg, er_line = sys.exc_info()
             print(f"the error line was:{er_line} error msg is :{er_msg} error type is :{er_ty}")


#write a main method for this class

if __name__=="__main__":
    dataset="House_Price_Prediction_Datasets.csv"
    obj=HOUSE_PRICE_PREDICTION(dataset)
    obj.train()
    obj.test()
    obj.own_data_point()
    obj.saving()















