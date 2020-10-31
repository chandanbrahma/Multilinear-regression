## importing the data
import pandas as pd
data= pd.read_csv('E:\\assignment\\multilinear regression\\Computer_Data.csv')
data.info()
data.head()

##we have 3 columns which needs to be converted into integer
data['cd'],_=pd.factorize(data['cd'])
data['multi'],_ = pd.factorize(data['multi'])
data['premium'],_ = pd.factorize(data['premium'])

data.info()
## now all ouur data are converted to integer format


##as all  the columns have different values so we need to normalize the data for better result as well as for calculation
data_new= ((data)-data.min())/(data.max()-data.min())
data_new.describe()


##creating x and y column
x=data_new.iloc[:,1:]
y=data_new.iloc[:,0]

##splliting the data into traaining and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)


##importing the linear regression
##building the model and fitting it to the data
from sklearn import linear_model 
model=linear_model.LinearRegression()
model.fit(x_train,y_train)


##checking performance
model_score= model.score(x_train,y_train)
## so we got a r^2 value of 0.779
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