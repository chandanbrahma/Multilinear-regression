import pandas as pd
data=pd.read_csv('E:\\assignment\\multilinear regression\\50_Startups.csv')
data.info()

## so in our data set we have 4 int column and one object column
data['State'].unique()
data['State'],_ = pd.factorize(data['State'])


##normalising the data 
data_new=(data-data.min())/(data.max()-data.min())
data_new.head()
data_new.describe()

##selecting the target variales and the predictors
x=data_new.iloc[:,:4]
y=data_new.iloc[:,4]


##training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

##fitting the model
from sklearn import linear_model
model= linear_model.LinearRegression()
model.fit(x_train,y_train)


##model parameter study
model_score= model.score(x_train,y_train)
## so we got a r^2 value of 0.95
y_predict= model.predict(x_test)

## data visualisation
import matplotlib.pyplot as plt
P=plt.subplot()
P.scatter(y_predict,y_test)
P.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--')
P.set_xlabel('Actual')
P.set_ylabel('Predicted')
P.set_title("Actual vs Predicted")
plt.show()

##We can see that our R^2 is very good. This means that we have found a well-fitting model to predict the profit.
