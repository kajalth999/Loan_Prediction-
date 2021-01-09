#importing the required modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder

#Read the Datasets
loan = pd.read_csv('train.csv')
loan_test = pd.read_csv('test.csv')
loan.describe()
loan.info()
loan.isnull().sum()

def fill_dataset(dataset):
    dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
    dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean())
    dataset['Credit_History'] = dataset['Credit_History'].fillna(dataset['Credit_History'].mean())
    dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
    dataset['Married'] = dataset['Married'].fillna(dataset['Married'].mode()[0])
    dataset['Dependents'] = dataset['Dependents'].fillna(dataset['Dependents'].mode()[0])
    dataset['Self_Employed'] = dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0])
    dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']

def stan_datasets(dataset):
    dataset['ApplicantIncomeLog'] = np.log(dataset['ApplicantIncome'])
    dataset['CoapplicantIncomeLog'] = np.log(1+dataset['CoapplicantIncome'])
    dataset['LoanAmountLog'] = np.log(dataset['LoanAmount'])
    dataset['Loan_Amount_Term_Log'] = np.log(dataset['Loan_Amount_Term'])
    dataset['TotalIncomeLog'] = np.log(dataset['TotalIncome'])

def plot_dataset(dataset):
    sns.distplot(dataset['ApplicantIncomeLog'])
    sns.distplot(dataset['CoapplicantIncomeLog'])
    sns.distplot(dataset['LoanAmountLog'])
    sns.distplot(dataset['Loan_Amount_Term_Log'])
    sns.distplot(dataset['Credit_History'])
    

#correlation matrix
corr = loan.corr()
sns.heatmap(corr,annot= True ,cmap = "BuPu" )


#drop columns
def clean_dataset(dataset):
    dataset = dataset.drop(['ApplicantIncome' , 'CoapplicantIncome' ,'LoanAmount' ,'TotalIncome' ,'Loan_ID','CoapplicantIncomeLog'],axis=1,inplace=True)


#Encoding the categorical data
def encoding_trainset(dataset):
    cols = ['Gender' , 'Married' , 'Education' , 'Self_Employed' , 'Property_Area' ,'Loan_Status','Dependents']
    le=LabelEncoder()
    for col in cols:
        dataset[col]=le.fit_transform(dataset[col])

def encoding_testset(dataset):
    cols = ['Gender' , 'Married' , 'Education' , 'Self_Employed' , 'Property_Area' ,'Dependents']
    le=LabelEncoder()
    for col in cols:
        dataset[col]=le.fit_transform(dataset[col])


fill_dataset(loan_test)
stan_datasets(loan_test)
clean_dataset(loan_test)
encoding_testset(loan_test)
fill_dataset(loan)
stan_datasets(loan)
clean_dataset(loan)
encoding_trainset(loan)

#training and testing model

X=loan.drop(columns='Loan_Status',axis=1)
y=loan['Loan_Status']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.model_selection import cross_val_score
#from xgboost import XGBClassifier
#rgr = XGBClassifier(n_estimators = 300)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
#rgr.fit(X_train,y_train)
#ypred=rgr.predict(X_test)
#print("Accuracyc is",rgr.score(X_test,y_test)*100)
#score = cross_val_score(rgr,X,y,cv=5)
#print("cross validation",np.mean(score)*100)

from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression(solver='liblinear')
classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,ypred)


#predicting loan status on given test set
y_pred = rgr.predict(loan_test)




