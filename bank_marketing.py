import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

data=pd.read_csv('Desktop\\bank-additional-full.csv',sep=';')
data_original=data.copy()

#print(data.head(5))
#print(data.columns)
#print(data.dtypes)
print(data.shape)
#print(data.describe())
#print(data.info())
#print(data.isnull().sum())
#print(data)

#univariate #dependant Variable
print(data['y'].value_counts())
print(data['y'].value_counts(normalize=True))  #for in terms of proportionate 
#data['y'].value_counts().plot.bar()


#inependant Variable  #Nominal
plt.figure(1)
plt.subplot(431)
data['job'].replace('unknown',data['job'].mode()[0],inplace=True)
data['job'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Job')
print(data['job'].value_counts(normalize=True))

plt.figure(1)
plt.subplot(432)
data['marital'].replace('unknown',data['marital'].mode()[0],inplace=True)
data['marital'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Marital')
print(data['marital'].value_counts(normalize=True))

plt.figure(1)
plt.subplot(433)
data['education'].replace('unknown',data['education'].mode()[0],inplace=True)
data['education'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='education')
print(data['education'].value_counts(normalize=True))

plt.figure(1)
plt.subplot(434)
#data['default'].replace('unknown',data['default'].apply(lambda x=data['loan']=='yes' or data['housing']=='yes': 'yes' if x==1 else 'no'), inplace=True)
for i in range (0,len(data['default'])):
	if data.iloc[i,4]=='unknown':
		x=(data.iloc[i,5]=='yes') and (data.iloc[i,6]=='yes')
		if x==True:
			data.iloc[i,4]='yes'
		else:
			data.iloc[i,4]='no'
data['default'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Default')
print(data['default'].value_counts(normalize=False))

plt.figure(1)
plt.subplot(435)
for i in range (0,len(data['housing'])):
	if data.iloc[i,5]=='unknown':
		if data.iloc[i,6]=='yes':
			data.iloc[i,5]='no'
		else:
			data.iloc[i,5]='yes'
data['housing'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Housing')
print(data['housing'].value_counts(normalize=True))

plt.figure(1)
plt.subplot(436)
for i in range (0,len(data['loan'])):
	if data.iloc[i,6]=='unknown':
		if data.iloc[i,5]=='yes':
			data.iloc[i,6]='no'
		else:
			data.iloc[i,6]='yes'
data['loan'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Loan')
print(data['loan'].value_counts(normalize=True))

print(data['contact'].value_counts(normalize=True))
plt.figure(1)
plt.subplot(437)
data['contact'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Contact')


print(data['month'].value_counts(normalize=True))
plt.figure(1)
plt.subplot(438)
data['month'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Month')

print(data['day_of_week'].value_counts(normalize=True))
plt.figure(1)
plt.subplot(439)								
data['day_of_week'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Day_of_week')


print(data['previous'].value_counts(normalize=True))
plt.figure(1)
plt.subplot(438)								
data['previous'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Previous')

print(data['poutcome'].value_counts(normalize=True))
plt.figure(1)
plt.subplot(439)								
data['poutcome'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Poutcome')

print(data['emp.var.rate'].value_counts(normalize=True))

print(data['cons.price.idx'].value_counts(normalize=True))


#ordinal
print(data['campaign'].value_counts(normalize=True))
plt.figure(2)
plt.subplot(211)								
data['campaign'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='campaign')

#print(([999]-data['pdays']).value_counts(normalize=False))
a=([999]-data['pdays'])
plt.figure(2)
plt.subplot(212)								
a.value_counts(normalize=False).plot.bar(figsize=(20,10),title='Pdays')
"""
print(data['nr.employed'].value_counts(normalize=False))
plt.figure(2)
plt.subplot(211)								
data['nr.employed'].value_counts(normalize=False).plot.bar(figsize=(20,10),title='Pdays')
"""

#Numerical
plt.figure(3)
data['duration_cbrt']=np.cbrt(data['duration'])
plt.subplot(221)
sns.distplot(data['duration_cbrt'])
plt.subplot(222)
plt.boxplot(data['duration_cbrt'])
#plt.boxplot(data['duration'])

plt.figure(3)
#data['duration_cbrt']=np.cbrt(data['duration'])
plt.subplot(224)
sns.distplot(data['age'])
plt.subplot(224)
plt.boxplot(data['age'])


"""
plt.figure(3)
plt.subplot(533)
sns.distplot(data['emp.var.rate'])
plt.subplot(534)
plt.boxplot(data['emp.var.rate'])

plt.figure(3)
plt.subplot(535)
sns.distplot(data['cons.conf.idx'])
plt.subplot(536)
plt.boxplot(data['cons.conf.idx'])

plt.figure(3)
plt.subplot(537)
sns.distplot(data['euribor3m'])
plt.subplot(538)
plt.boxplot(data['euribor3m'])
"""

#Bivariate
#Categorical X Vs Y
job=pd.crosstab(data['job'], data['y']) 
#print (job)								#retired persons and students MORE Y
job.div(job.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))


marital=pd.crosstab(data['marital'], data['y']) 
#print (marital)								#SINGLE MORE Y BUT NOT SIGNIFICANT
marital.div(marital.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

education=pd.crosstab(data['education'], data['y']) 
#print (education)								#ILLITURATE MORE Y. SIGNIFICANT
education.div(education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

default=pd.crosstab(data['default'], data['y']) 
#print (default)								#default(no) more y
default.div(default.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))


housing=pd.crosstab(data['housing'], data['y']) 
#print (housing)								#NOT SIGNIFICANT
housing.div(housing.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

loan=pd.crosstab(data['loan'], data['y']) 
#print (loan)								#NOT SIGNIFICANT
loan.div(loan.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

contact=pd.crosstab(data['contact'], data['y']) 
#print (contact)								#cellular more Y
contact.div(contact.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

month=pd.crosstab(data['month'], data['y']) 
#print (month)								#March december more significant
month.div(month.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

day_of_week=pd.crosstab(data['day_of_week'], data['y']) 
#print (day_of_week)									#NOT SIGNIFICANT
day_of_week.div(day_of_week.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

previous=pd.crosstab(data['previous'], data['y']) 
#print (previous)									# SIGNIFICANT
previous.div(previous.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

poutcome=pd.crosstab(data['poutcome'], data['y']) 
#print (poutcome)									# SIGNIFICANT
poutcome.div(poutcome.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

campaign=pd.crosstab(data['campaign'], data['y']) 
#print (campaign)									# SIGNIFICANT
campaign.div(campaign.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))


#Numerical  X Vs Y

#data.groupby('y')['duration_cbrt'].mean().plot.bar()
print(data.groupby('y')['duration_cbrt'].mean())				
bins=[0,4,10,20] 
group=['low','medium','high']
data['duration_bin']=pd.cut(data['duration_cbrt'],bins,labels=group)
duration_bin=pd.crosstab(data['duration_bin'],data['y'])
print(duration_bin)												#significant
duration_bin.div(duration_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(5,5))
plt.xlabel('duration')
p=plt.ylabel('Percentage')

#data.groupby('y')['age'].median().plot.bar()
print(data.groupby('y')['age'].median())				

bins=[0,30,60,90] 
group=['Young','Adult','Retired']
data['age_bin']=pd.cut(data['age'],bins,labels=group)
age_bin=pd.crosstab(data['age_bin'],data['y'])					#significant
age_bin.div(age_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(5,5))
plt.xlabel('age')
p=plt.ylabel('Percentage')

#plt.show()
print(data.columns)
data=data.drop(['age','marital','housing','loan','month','day_of_week','pdays','duration','duration_cbrt','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],axis=1)
print(data.columns)

#seperate out X and Y i reain file into two saperate files
data['y'].replace('no',0,inplace=True)
data['y'].replace('yes',1,inplace=True)
x=data.drop('y',1)
y=data.y

print(x.head())
print(y.head())

#make dumm variable 
X=pd.get_dummies(x)

print(X.head())
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=0.3)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
model.fit(x_train,y_train)
pred_cv=model.predict(x_cv)
print('Logistic Regression accuracy is:')
print(accuracy_score(y_cv,pred_cv))


#Stratified K fold cross validation
from sklearn.model_selection import StratifiedKFold
i=1
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in  kf.split(X,y):
	print('\n{} of kfold{}'.format(i,kf.n_splits))
	xtr,xvl=X.loc[train_index],X.loc[test_index]
	ytr,yvl=y[train_index],y[test_index]
	
	model=LogisticRegression(random_state=1)
	model.fit(xtr,ytr)
	pred_test=model.predict(xvl)
	score=accuracy_score(yvl,pred_test)
	print('Logistic accuracy_score',score)
	i+=1

pred=model.predict_proba(xvl)[:,1]

#Decision Tree
from sklearn import tree
i=1
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
	print('\n{} of kfold{}'.format(i,kf.n_splits))
	xtr,xvl=X.loc[train_index],X.loc[test_index]
	ytr,yvl=y[train_index],y[test_index]
	
	model=tree.DecisionTreeClassifier(random_state=1)
	model.fit(xtr,ytr)
	pred_test=model.predict(xvl)
	score=accuracy_score(yvl,pred_test)
	print('Decision Tree accuracy_score',score)
	i+=1
	

#Random Forest
from sklearn.ensemble import RandomForestClassifier
i=1
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
	print('\n{} of kfold{}'.format(i,kf.n_splits))
	xtr,xvl=X.loc[train_index],X.loc[test_index]
	ytr,yvl=y[train_index],y[test_index]
	
	model=RandomForestClassifier(random_state=1)
	model.fit(xtr,ytr)
	pred_test=model.predict(xvl)
	score=accuracy_score(yvl,pred_test)
	print('RF accuracy_score',score)
	i+=1
	
#XGBoost
from xgboost import XGBClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = XGBClassifier(n_estimators=20, max_depth=4)
     model.fit(xtr, ytr);
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('XGBoost accuracy_score',score)
     i+=1
	 
#SVM
from sklearn.svm import SVC
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = SVC(gamma=0.001,random_state=1)
     model.fit(xtr, ytr);
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('SVM accuracy_score',score)
     i+=1
	 
#Hence here the accuracy_score for SVM and XGBoost is better.