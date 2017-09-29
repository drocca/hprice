import pandas as pd 
import numpy
from numpy import linalg 
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import scipy.linalg
import random
import math

# Meaning of the header
#survival	Survival 
#pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#sex	Sex	
#Age	Age in years	
#sibsp	# of siblings / spouses aboard the Titanic	
#parch	# of parents / children aboard the Titanic	
#ticket	Ticket number	
#fare	Passenger fare	
#cabin	Cabin number	
# embarked	Port of Embarkation
#header: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

### Read the csv files to train and to test 

tr=pd.read_csv("train.csv")
sztr=len(tr.ix[:, 'Id'])

ts=pd.read_csv("test.csv")
#ts.insert(1, 'Survived', 0)
szts=len(pas_ts.ix[:, 'PassengerId'])
psid=pas_ts['PassengerId']

pas_tot=pd.concat([pas_tr, pas_ts], axis=0, ignore_index=True)
sztot=len(pas_tot.ix[:, 'PassengerId'])

### Print some information of the two files
print ('!-----------------------!')
pas_tr.info()
print ('!-----------------------!')
pas_ts.info()
print ('!-----------------------!')
pas_tot.info()

### Print a few values
print ('!-----------------------!')
print pas_tr.head()
print ('!-----------------------!')
print pas_ts.head()


### Drop the column 'PassengerId' which is for sure useless

pas_tr=pas_tr.drop('PassengerId',axis=1)
pas_ts=pas_ts.drop('PassengerId',axis=1)
pas_tot=pas_tot.drop('PassengerId',axis=1)

### Drop the column 'Ticket'
### It is hard to understand how this column would contribute
### The information conatined in it might already be incuded in 'Pclass' or 'Fare' or 'Embarked' 

ticket_dummies_tot  = pd.get_dummies(pas_tot['Ticket'])

pas_tot = pd.concat([pas_tot, ticket_dummies_tot],axis=1)
pas_tr = pd.concat([pas_tr, ticket_dummies_tot.ix[0:890,:]],axis=1)
tmp=ticket_dummies_tot.ix[891:1308,:]
tmp.index=range(418)
pas_ts = pd.concat([pas_ts,tmp],axis=1)

#print "pippo"
#print pas_ts.head
#print ticket_dummies_tot.ix[891:1308,:]

pas_tr=pas_tr.drop('Ticket',axis=1)
pas_ts=pas_ts.drop('Ticket',axis=1)
pas_tot=pas_tot.drop('Ticket',axis=1)

pas_tr=pas_tr.drop('373450',axis=1)
pas_ts=pas_ts.drop('373450',axis=1)
pas_tot=pas_tot.drop('373450',axis=1)

### Transform the data on sex into an array of 0 and 1

def sex_to_integer(sex):
    if sex=='male':
        si=0
    else:
        si=1
    return si

pas_tr['Sex'] = pas_tr.Sex.apply(sex_to_integer)
pas_ts['Sex'] = pas_ts.Sex.apply(sex_to_integer)
pas_tot['Sex'] = pas_tot.Sex.apply(sex_to_integer)

### Define a new useful column with total number of family members

pas_tr['TotFamily']=pas_tr['SibSp']+pas_tr['Parch']+1
pas_ts['TotFamily']=pas_ts['SibSp']+pas_ts['Parch']+1
pas_tot['TotFamily']=pas_tot['SibSp']+pas_tot['Parch']+1

### Create column with titles: Mr., Miss, etc.

def find_title(nm):
    if ("Master" in nm):
        title="Master"
    elif ("Miss." in nm) or ("Mlle" in nm) or ("Ms." in nm):
        title="Miss"
    elif ("Mr." in nm) or ("Don." in nm) or ("Major" in nm) or ("Capt" in nm) or ("Jonkheer" in nm) or ("Rev" in nm) or ("Col" in nm) or ("Sir." in nm):
        title="Mr"
    elif ("Mrs." in nm) or ("Countess" in nm) or ("Mme" in nm) or ("Dona." in nm) or ("Lady." in nm):
        title="Mrs"
    elif ("Dr" in nm):
        title="Dr"
    return title

pas_tr['Title'] = pas_tr.Name.apply(find_title)
pas_ts['Title'] = pas_ts.Name.apply(find_title)

title_dummies_tr  = pd.get_dummies(pas_tr['Title'])
title_dummies_tr.columns = ['Title_1','Title_2','Title_3','Title_4','Title_5']
title_dummies_tr.drop(['Title_5'], axis=1, inplace=True)

title_dummies_ts  = pd.get_dummies(pas_ts['Title'])
title_dummies_ts.columns = ['Title_1','Title_2','Title_3','Title_4','Title_5']
title_dummies_ts.drop(['Title_5'], axis=1, inplace=True)

pas_tr.drop(['Title'],axis=1,inplace=True)
pas_ts.drop(['Title'],axis=1,inplace=True)

pas_tr = pas_tr.join(title_dummies_tr)
pas_ts = pas_ts.join(title_dummies_ts)

### Create column with 1 if the Name has parethesis and 0 if not

def find_par(nm):
    if ("(" in nm):
        par=1
    else:
        par=0
    return par

pas_tr['Par'] = pas_tr.Name.apply(find_par)
pas_ts['Par'] = pas_ts.Name.apply(find_par)

### Manipulations concerning the age
### Ages below 1 are rounded to 0
### A few "intemediate" ages are rounded (example: 45.5--->46)
### Fill in the missing ages
### To fill in missing ages let's consider that a Master is a child
### that a Miss travelling with a Parch is more likely to be a child
### I replace the non defined ages with averages of subgroups

agetot=[]
agekidm=[]
agekidf=[]
agemiss=[]
agemr=[]
agemrs=[]

pas_tot.Age = pas_tot.Age.fillna(-10)
pas_tr.Age = pas_tr.Age.fillna(-10)
pas_ts.Age = pas_ts.Age.fillna(-10)
for i in range(sztot):
    if (pas_tot.ix[i,'Age']>0):
        agetot.append(pas_tot.ix[i,'Age'])
    if ("Master" in pas_tot.ix[i,'Name'] and pas_tot.ix[i,'Age']>0):
        agekidm.append(pas_tot.ix[i,'Age'])
    elif ("Miss." in pas_tot.ix[i,'Name'] and pas_tot.ix[i,'Age']>0 and pas_tot.ix[i,'Parch'] >= 1):
        agekidf.append(pas_tot.ix[i,'Age'])
    elif ("Miss." in pas_tot.ix[i,'Name'] and pas_tot.ix[i,'Age']>0 and pas_tot.ix[i,'Parch'] == 0):
        agemiss.append(pas_tot.ix[i,'Age'])
    elif ("Mr." in pas_tot.ix[i,'Name'] and pas_tot.ix[i,'Age']>0):
        agemr.append(pas_tot.ix[i,'Age'])
    elif ("Mrs." in pas_tot.ix[i,'Name'] and pas_tot.ix[i,'Age']>0):
        agemrs.append(pas_tot.ix[i,'Age'])

meankidm=numpy.mean(agekidm) 
meankidf=numpy.mean(agekidf)
meanmiss=numpy.mean(agemiss)
meanmr=numpy.mean(agemr)
meanmrs=numpy.mean(agemrs)
meantot=numpy.mean(agetot)

for i in range(sztr):
    if ("Master" in pas_tr.ix[i,'Name'] and pas_tr.ix[i,'Age']<0):
        pas_tr.ix[i,'Age']=meankidm
    elif ("Miss." in pas_tr.ix[i,'Name'] and pas_tr.ix[i,'Age']<0 and pas_tr.ix[i,'Parch'] >= 1):
        pas_tr.ix[i,'Age']=meankidf
    elif ("Miss." in pas_tr.ix[i,'Name'] and pas_tr.ix[i,'Age']<0 and pas_tr.ix[i,'Parch'] == 0):
        pas_tr.ix[i,'Age']=meanmiss
    elif ("Mr." in pas_tr.ix[i,'Name'] and pas_tr.ix[i,'Age']<0):
        pas_tr.ix[i,'Age']=meanmr
    elif ("Mrs." in pas_tr.ix[i,'Name'] and pas_tr.ix[i,'Age']<0):
        pas_tr.ix[i,'Age']=meanmrs
    else:
        if (pas_tr.ix[i,'Age']<0):
            pas_tr.ix[i,'Age']=meantot

for i in range(szts):
    if ("Master" in pas_ts.ix[i,'Name'] and pas_ts.ix[i,'Age']<0):
        pas_ts.ix[i,'Age']=meankidm
    elif ("Miss." in pas_ts.ix[i,'Name'] and pas_ts.ix[i,'Age']<0 and pas_ts.ix[i,'Parch'] >= 1):
        pas_ts.ix[i,'Age']=meankidf
    elif ("Miss." in pas_ts.ix[i,'Name'] and pas_ts.ix[i,'Age']<0 and pas_ts.ix[i,'Parch'] == 0):
        pas_ts.ix[i,'Age']=meanmiss
    elif ("Mr." in pas_ts.ix[i,'Name'] and pas_ts.ix[i,'Age']<0):
        pas_ts.ix[i,'Age']=meanmr
    elif ("Mrs." in pas_ts.ix[i,'Name'] and pas_ts.ix[i,'Age']<0):
        pas_ts.ix[i,'Age']=meanmrs
    else:
        if (pas_ts.ix[i,'Age']<0):
            pas_ts.ix[i,'Age']=meantot

#N = pas_tr.ix[:, 'Ticket']
#C = Counter(N)
#print [ [k,]*v for k,v in C.items()]

def round_age(age):
    if age<1:
        ageout=0
    else:
        ageout=round(age,0)
    return ageout

pas_tr['Age'] = pas_tr.Age.apply(round_age)
pas_ts['Age'] = pas_ts.Age.apply(round_age)

### Cabin might not be important
### We will just use a variable 1=cabin assigned 0=cabin not assigned

pas_tr.Cabin = pas_tr.Cabin.fillna(-10)
pas_ts.Cabin = pas_ts.Cabin.fillna(-10)

def dummycabin(cabin):
    if cabin==-10:
        cabinout=0
    else:
        cabinout=1
    return cabinout

pas_tr['Cabin'] = pas_tr.Cabin.apply(dummycabin)
pas_ts['Cabin'] = pas_ts.Cabin.apply(dummycabin)

### Manipulations concerning the fare
### It is reasonable that the fare goes "linearly" with # of family memebers
### accordingly we renormalize it and introduce the 'NFare' column

for i in range(szts):
    print pas_ts.ix[i,'Fare'] , pas_ts.ix[i,'Name']

mn=numpy.mean(pas_ts['Fare'])
pas_ts.Fare = pas_ts.Fare.fillna(mn)

pas_tr['NFare'] = pas_tr['Fare']/pas_tr['TotFamily']
pas_ts['NFare'] = pas_ts['Fare']/pas_ts['TotFamily']


### Plotting data
###########################################
sns.set(font_scale = 0.45)
plt.figure(figsize=[80,12])
plt.subplot(211)
#sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
#sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
#            axlabel='Age')
#df = df.dropna(axis=0)
#plotage=pas_tr.ix[:]
sns.barplot('Age', 'Survived', data=pas_tr)
#sns.distplot('Age', 'Survived',bins=range(0, 81, 1), data=pas_tr)
plt.subplot(212)
sns.countplot(x='Age', data=pas_tr )

#plt.show()

plt.figure(figsize=[12,18])

plt.subplot(321)
sns.barplot('Sex', 'Survived', data=pas_tr)
plt.subplot(322)
sns.countplot(x='Sex', data=pas_tr )

plt.subplot(323)
sns.barplot('Pclass', 'Survived', data=pas_tr)
plt.subplot(324)
sns.countplot(x='Pclass', data=pas_tr )

plt.subplot(325)
sns.barplot('Embarked', 'Survived', data=pas_tr)
plt.subplot(326)
sns.countplot(x='Embarked', data=pas_tr )

#plt.show()

plt.figure(figsize=[12,18])

plt.subplot(221)
sns.barplot('SibSp', 'Survived', data=pas_tr)
plt.subplot(222)
sns.countplot(x='SibSp', data=pas_tr )
plt.subplot(223)
sns.barplot('Parch', 'Survived', data=pas_tr)
#sns.barplot('Fare', 'Survived', data=pas_tr)
plt.subplot(224)
sns.countplot(x='Parch', data=pas_tr )
#sns.countplot(x='Fare', data=pas_tr )

#plt.show()

plt.figure(figsize=[12,18])

plt.subplot(211)
sns.barplot('NFare', 'Survived', data=pas_tr)
plt.subplot(212)
sns.barplot('Cabin', 'Survived', data=pas_tr)

#plt.subplot(8,2,13)
#sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color=surv_col)
#sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
#plt.subplots_adjust(top=0.2, bottom=0.08, left=0.10, right=0.5, hspace=0.85,
#                    wspace=0.65)

#plt.show()
#################################################

### Create dummy variables for Pclass

pclass_dummies_tr  = pd.get_dummies(pas_tr['Pclass'])
pclass_dummies_tr.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_tr.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_ts  = pd.get_dummies(pas_ts['Pclass'])
pclass_dummies_ts.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_ts.drop(['Class_3'], axis=1, inplace=True)

pas_tr.drop(['Pclass'],axis=1,inplace=True)
pas_ts.drop(['Pclass'],axis=1,inplace=True)

pas_tr = pas_tr.join(pclass_dummies_tr)
pas_ts = pas_ts.join(pclass_dummies_ts)

###

### Create dummy variables for Embarked

pas_tr.Embarked = pas_tr.Embarked.fillna('S')
pas_ts.Embarked = pas_ts.Embarked.fillna('S')

embarked_dummies_tr  = pd.get_dummies(pas_tr['Embarked'])
embarked_dummies_tr.columns = ['Embarked_1','Embarked_2','Embarked_3']
embarked_dummies_tr.drop(['Embarked_3'], axis=1, inplace=True)

embarked_dummies_ts  = pd.get_dummies(pas_ts['Embarked'])
embarked_dummies_ts.columns = ['Embarked_1','Embarked_2','Embarked_3']
embarked_dummies_ts.drop(['Embarked_3'], axis=1, inplace=True)

pas_tr.drop(['Embarked'],axis=1,inplace=True)
pas_ts.drop(['Embarked'],axis=1,inplace=True)

pas_tr = pas_tr.join(embarked_dummies_tr)
pas_ts = pas_ts.join(embarked_dummies_ts)

###

pas_tr['Mul']=pas_tr['Age']*pas_tr['Fare']
pas_ts['Mul']=pas_ts['Age']*pas_ts['Fare']

###

pas_tr=pas_tr.drop('Name',axis=1)
pas_ts=pas_ts.drop('Name',axis=1)
#pas_tr=pas_tr.drop('Fare',axis=1)
#pas_ts=pas_ts.drop('Fare',axis=1)

### Print a few values of the modified datastructures
print ('!-----------------------!')
print pas_tr.head()
print ('!-----------------------!')
print pas_ts.head()


### Overwrite pas_tot with the new data structure; useful for feature normalization

pas_tot=pd.concat([pas_tr, pas_ts], axis=0, ignore_index=True)

### Normalize Age and Fare

mn=numpy.mean(pas_tot['Mul'])
st=numpy.std(pas_tot['Mul'])
mx=max(pas_tot['Mul'])
pas_tr['Mul']=(pas_tr['Mul']-mn)/(st)
pas_ts['Mul']=(pas_ts['Mul']-mn)/(st)

mn=numpy.mean(pas_tot['Age'])
st=numpy.std(pas_tot['Age'])
mx=max(pas_tot['Age'])
pas_tr['Age']=(pas_tr['Age']-mn)/(st)
pas_ts['Age']=(pas_ts['Age']-mn)/(st)

mn=numpy.mean(pas_tot['NFare'])
st=numpy.std(pas_tot['NFare'])
mx=max(pas_tot['NFare'])
pas_tr['NFare']=(pas_tr['NFare']-mn)/(st)
pas_ts['NFare']=(pas_ts['NFare']-mn)/(st)

mn=numpy.mean(pas_tot['Fare'])
st=numpy.std(pas_tot['Fare'])
mx=max(pas_tot['Fare'])
pas_tr['Fare']=(pas_tr['Fare']-mn)/(st)
pas_ts['Fare']=(pas_ts['Fare']-mn)/(st)

mn=numpy.mean(pas_tot['TotFamily'])
st=numpy.std(pas_tot['TotFamily'])
mx=max(pas_tot['TotFamily'])
pas_tr['TotFamily']=pas_tr['TotFamily']
pas_ts['TotFamily']=pas_ts['TotFamily']

mn=numpy.mean(pas_tot['SibSp'])
st=numpy.std(pas_tot['SibSp'])
mx=max(pas_tot['SibSp'])
pas_tr['SibSp']=pas_tr['SibSp']
pas_ts['SibSp']=pas_ts['SibSp']

mn=numpy.mean(pas_tot['Parch'])
st=numpy.std(pas_tot['Parch'])
mx=max(pas_tot['Parch'])
pas_tr['Parch']=pas_tr['Parch']
pas_ts['Parch']=pas_ts['Parch']


print pas_tr.head()
#print ('!-----------------------!')
print pas_ts.head()

print pas_tr.head()
#print ('!-----------------------!')
print pas_ts.head()



### Defining training and testing sets
#pas_tr=pas_tr.drop('Age',axis=1)
#pas_ts=pas_ts.drop('Age',axis=1)
#pas_tr=pas_tr.drop('Mul',axis=1)
#pas_ts=pas_ts.drop('Mul',axis=1)


X_train = pas_tr.drop("Survived",axis=1)
Y_train = pas_tr["Survived"]
X_test  = pas_ts.drop("Survived",axis=1)

#X_train, X_test, Y_train, Y_test = train_test_split(dat, trg, test_size=0.4, random_state=0)

#clf = RandomForestClassifier(n_estimators=100)
clf = svm.SVC(kernel='rbf',C=200,gamma=0.01)
#clf = LogisticRegression(tol=0.000001,C=1)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(80, 80), random_state=1)
clf.fit(X_train, Y_train)
scores = cross_val_score(clf, X_train, Y_train, cv=10)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf.fit(X_train, Y_train)
prediction=clf.predict(X_test)
print clf.score(X_train, Y_train)


submission = pd.DataFrame({
        "PassengerId": psid,
        "Survived": prediction
    })
submission.to_csv('final.csv', index=False)


#print "sztr",sztr
#print y  # pas.ix[10, 'Survived']

#y=numpy.zeros((900))
#print pas.as_matrix
#print y
