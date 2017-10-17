import pandas as pd 
import numpy as np
from numpy import linalg 
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg
import random
import math

### Read the csv files for training and test set 

tr=pd.read_csv("train.csv")
ts=pd.read_csv("test.csv")


print ("Training set shape {}".format(tr.shape))
print ("Test set shape {}".format(ts.shape))

ts.insert(80, 'SalePrice', 0)

# Data frame that contains training and test set 
# I'll work on this for a first cleaning and data analysis
tot=pd.concat([tr, ts], axis=0, ignore_index=True)

print ("Tot set shape {}".format(tot.shape))

print " "
print " "


#['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']



#tot=pd.concat([tr, ts], axis=0, ignore_index=True)

### Print a few values
print ('!-----------------------!')
print ('Training data set')
print tr.head()
print " "
print ('!-----------------------!')
print ('Test data set')
print ts.head()
print " "
print ('!-----------------------!')
print ('Tot data set')
print tot.head()
print " "

### Print some information on the two datasets
print ('!-----------------------!')
#tr.info()
print ('!-----------------------!')
#ts.info()
print ('!-----------------------!')
#tot.info()


### Drop the column 'Id' which is for sure a useless feauture

hid=ts['Id']
tot=tot.drop('Id',axis=1)

### A first screening on data: type of each column, a few examples printed, percentage of non-null items per column, and number of uniques items

l, c=tot.shape
print "Column, type, line1, line2, line3, line4, precentage of non-null entries, number of unique entries"
for i in tot.columns:
#    if (float(tot[i].count())/float(l)*100<100):
    print i, tot[i].dtype, list(tot.loc[0:3,i]), float(tot[i].count())/float(l)*100, tot[i].nunique()

#print tr.loc[:,'Alley']

##############################################################

# Nan entries per line
#a=tot.isnull().sum(axis=1)
#print max(a)
# The max value is 16

#############################################################

### Some observations from the previous screening

# MSSubClass int64 [60, 20, 60, 70] 100 15
# MSSubClass should be categorical but it's probably good to keeo the order in the numbers (ordinal variable)

#OverallQual int64 [7, 6, 7, 7] 100 10
#OverallCond int64 [5, 8, 5, 5] 100 9
# Ordinal variables

# YearBuilt int64 [2003, 1976, 2001, 1915] 100 112
# YearRemodAdd int64 [2003, 1976, 2002, 1970] 100 61
# I think that in all the year variables is good to keep the order and not to use 
# one-hot encoding, which would also significatively increase the number of columns 

# the variables below might be considered ordinal and label oncoding might help
# I will need to interpret the entries: 'Gd' means likely 'good'  
#ExterQual object ['Gd', 'TA', 'Gd', 'TA'] 100 4
#ExterCond object ['TA', 'TA', 'TA', 'TA'] 100 5
#BsmtQual object ['Gd', 'Gd', 'Gd', 'TA'] 97.2250770812 4
#BsmtCond object ['TA', 'TA', 'TA', 'Gd'] 97.1908187736  4
# KitchenQual object ['Gd', 'TA', 'Gd', 'Gd'] 99.9657416924 4
#GarageQual object ['TA', 'TA', 'TA', 'TA'] 94.5529290853 5
#GarageCond object ['TA', 'TA', 'TA', 'TA'] 94.5529290853 5

# Replacement according to dictionary
# {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5} 
# Po: poor, Fa: fair, TA: average, Gd: good, Ex: excellent


repl_list=['ExterQual','ExterCond','BsmtQual','BsmtCond','KitchenQual','GarageQual','GarageCond']
repl_dic = {'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5}

#for i in repl_list:
#    print float(tot[i].count())/float(l)*100
#    print tot[i].value_counts()


for i in repl_list:
#since only relatively few values are missing I'll replace Nan with TA:3 which is an average rate
# and the dominant one
    tot[i]=tot[i].fillna('3')
    tot[i].replace(repl_dic, inplace=True)


# For month and year let's keep the encoding below
#MoSold int64 [2, 5, 9, 2] 100 12
#YrSold int64 [2008, 2007, 2008, 2006] 100 5

# SaleCondition might be considered ordinal too
# Leave it the way it is for the moment
#SaleCondition object ['Normal', 'Normal', 'Normal', 'Abnorml'] 100 6


#############################################################

# Let's look at the most recurrent values for some of the catagorical data 
# It could help to guess on how to replace them

#inc_feat= ['MSZoning', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',   ]

### Let's take a look again to see how many data are still missing
#l, c=tot.shape
#print "Column, type, line1, line2, line3, line4, precentage of non-null entries, number of unique entries"
#for i in tot.columns:
#    if (float(tot[i].count())/float(l)*100<100):
#        print i, tot[i].dtype, list(tot.loc[0:3,i]), float(tot[i].count())/float(l)*100, tot[i].nunique()


#############################################################

# According to the previous screening the following variables still miss some values 

#Column, type, line1, line2, line3, line4, precentage of non-null entries, number of unique entries
#MSZoning object ['RL', 'RL', 'RL', 'RL'] 99.8629667694 5
#LotFrontage float64 [65.0, 80.0, 68.0, 60.0] 83.3504624872 128
#Alley object [nan, nan, nan, nan] 6.78314491264 2
#Utilities object ['AllPub', 'AllPub', 'AllPub', 'AllPub'] 99.9314833847 2
#Exterior1st object ['VinylSd', 'MetalSd', 'VinylSd', 'Wd Sdng'] 99.9657416924 15
#Exterior2nd object ['VinylSd', 'MetalSd', 'VinylSd', 'Wd Shng'] 99.9657416924 16
#MasVnrType object ['BrkFace', 'None', 'BrkFace', 'None'] 99.1778006166 4
#MasVnrArea float64 [196.0, 0.0, 162.0, 0.0] 99.2120589243 444
#BsmtExposure object ['No', 'Gd', 'Mn', 'No'] 97.1908187736 4
#BsmtFinType1 object ['GLQ', 'ALQ', 'GLQ', 'ALQ'] 97.2935936965 6
#BsmtFinSF1 float64 [706.0, 978.0, 486.0, 216.0] 99.9657416924 991
#BsmtFinType2 object ['Unf', 'Unf', 'Unf', 'Unf'] 97.2593353888 6
#BsmtFinSF2 float64 [0.0, 0.0, 0.0, 0.0] 99.9657416924 272
#BsmtUnfSF float64 [150.0, 284.0, 434.0, 540.0] 99.9657416924 1135
#TotalBsmtSF float64 [856.0, 1262.0, 920.0, 756.0] 99.9657416924 1058
#Electrical object ['SBrkr', 'SBrkr', 'SBrkr', 'SBrkr'] 99.9657416924 5
#BsmtFullBath float64 [1.0, 0.0, 1.0, 1.0] 99.9314833847 4
#BsmtHalfBath float64 [0.0, 1.0, 0.0, 0.0] 99.9314833847 3
#Functional object ['Typ', 'Typ', 'Typ', 'Typ'] 99.9314833847 7
#FireplaceQu object [nan, 'TA', 'TA', 'Gd'] 51.3532031518 5
#GarageType object ['Attchd', 'Attchd', 'Attchd', 'Detchd'] 94.6214457006 6
#GarageYrBlt float64 [2003.0, 1976.0, 2001.0, 1998.0] 94.5529290853 103
#GarageFinish object ['RFn', 'RFn', 'RFn', 'Unf'] 94.5529290853 3
#GarageCars float64 [2.0, 2.0, 2.0, 3.0] 99.9657416924 6
#GarageArea float64 [548.0, 460.0, 608.0, 642.0] 99.9657416924 603
#PoolQC object [nan, nan, nan, nan] 0.342583076396 3
#Fence object [nan, nan, nan, nan] 19.5614936622 4
#MiscFeature object [nan, nan, nan, nan] 3.59712230216 4
#SaleType object ['WD', 'WD', 'WD', 'WD'] 99.9657416924 9

miss_list=['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']

# replacing missing catagorical features; if only few values are missing replace them with the most frequent item 
for i in miss_list:
        if (tot[i].dtype=='object'):
			if (float(tot[i].count())/float(l)*100 > 97):
	                #    print i, float(tot[i].count())/float(l)*100    
		        #    print tot[i].value_counts()
			#    print tot[i].value_counts().idxmax()
                            a = str(tot[i].value_counts().idxmax())
                            tot[i]=tot[i].fillna(a) 
                        else:
			    tot[i]=tot[i].fillna('Notavailable')

# replacing missing continous variables 

mn=tot['LotFrontage'].mean()
tot['LotFrontage']=tot['LotFrontage'].fillna(mn)  #Replace with average

mn=tot['MasVnrArea'].mean()
tot['MasVnrArea']=tot['MasVnrArea'].fillna(mn)  #Replace with average

tot['BsmtFinSF1']=tot['BsmtFinSF1'].fillna(0)  #Replace with 0 for the moment - no basement 

tot['BsmtFinSF2']=tot['BsmtFinSF2'].fillna(0)  #Replace with 0 for the moment - no basement 

tot['BsmtUnfSF']=tot['BsmtUnfSF'].fillna(0) #Replace with 0 for the moment - no basement

tot['BsmtUnfSF']=tot['BsmtUnfSF'].fillna(0) #Replace with 0 for the moment - no basement
 
tot['TotalBsmtSF']=tot['TotalBsmtSF'].fillna(0) #Replace with 0 for the moment - no basement
 
tot['BsmtFullBath']=tot['BsmtFullBath'].fillna(0) #Replace with 0 for the moment - no basement
 
tot['BsmtHalfBath']=tot['BsmtHalfBath'].fillna(0) #Replace with 0 for the moment - no basement
 
tot['GarageYrBlt']=tot['GarageYrBlt'].fillna(tot['YearBuilt']) # Let's usppose the year of contruction of the house 
 
tot['GarageCars']=tot['GarageCars'].fillna(0) #Replace with 0 for the moment - no garage

tot['GarageArea']=tot['GarageArea'].fillna(0) #Replace with 0 for the moment - no garage

#check if there is still some missing value
#print tot.isnull().values.any()

##################################################
# getting dummy variables corresponding to categorical variables

tot=pd.get_dummies(data=tot)
print tot.head()

##################################################
# regression

tr=tot.iloc[:1460,:]
ts=tot.iloc[1460:,:]


X_train = tr.drop("SalePrice",axis=1)
Y_train = tr["SalePrice"]
X_test  = ts.drop("SalePrice",axis=1)

#reg = KernelRidge(gamma=.0002441,kernel='laplacian',alpha=0.00000000093132257)
#reg = KernelRidge(alpha=2000.)
reg = Ridge(alpha=0.5, normalize=True)
#reg=RandomForestRegressor(n_estimators=100)
#reg.fit(X_train, Y_train)
scores = cross_val_score(reg, X_train, Y_train, scoring='neg_mean_absolute_error', cv=10)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#scores = cross_validation.cross_val_score(reg, X_train, Y_train, scoring='mean_squared_error', cv=10,)

# This will print the mean of the list of errors that were output and 
# provide your metric for evaluation
#print scores.mean()

reg.fit(X_train, Y_train)
prediction=reg.predict(X_test)
#print reg.score(X_train, Y_train)

submission = pd.DataFrame({
        "Id": hid,
        "SalePrice": prediction
    })
submission.to_csv('final.csv', index=False)
