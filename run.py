import numpy as np
from helpers import *
from implementations import *
## Importing data
x_train,x_test,y_train,train_ids,test_ids = load_csv_data("dataset",sub_sample=False)#Extract Data
print("Data extracted")
##
####################
#Pre-processing data
####################
N=x_train.shape[0]

##Removing NANs
nans = np.zeros(x_train.shape)
nans[np.isnan(x_train)]=1
nans_occurence=np.sum(nans,axis=0) #Number of nans by feature
good_features = (nans_occurence < N*0.3) #Keep only features with less than 30% of Nans

##Removing constant features and standardizing data
std_per_feature = np.nanstd(x_train,axis=0)
good_features = np.logical_and(good_features , std_per_feature!=0)
x_tr,mean_per_feature,std_per_feature = standardize(x_train[:,good_features])
x_te = x_test[:,good_features]
x_te -=mean_per_feature
x_te /=std_per_feature
np.nan_to_num(x_tr,copy=False,nan=0) #Replace Nans with 0
np.nan_to_num(x_te,copy=False,nan=0) #Same
#Adjustments
y_train[y_train==-1]=0 # The logistic loss function is implemented for y equal to 1 and 0 not 1 and -1
x_tr=np.insert(x_tr,0,1,axis=1)# Add a constant feature
x_te=np.insert(x_te,0,1,axis=1)#Add a constant feature
print("Data pre-processed")
##Computing the test x
D=x_tr.shape[1]#Number of features
initial_w=np.ones(D)
w,loss = reg_logistic_regression_stoch(y_train,x_tr,1.4,initial_w,200,0.01,64)
prob = sigmoid(x_te.dot(w))
y_test=np.zeros(x_te.shape[0])
y_test[prob >=0.5]=1
y_test[prob <0.5]=-1
create_csv_submission(test_ids,y_test,"prediction.csv")