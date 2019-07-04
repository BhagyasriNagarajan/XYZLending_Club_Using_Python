# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:02:50 2019

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 01:02:58 2019

@author: dell

Lending Club:
Our business problem is that investors require a more comprehensive assessment 
of these borrowers than what is presented by Lending Club in order to make a smart
business decision, by identifying new borrowers that would likely default on their 
loans.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%

#Reading the file
loan_credit = pd.read_csv(r'C:\Users\dell\Desktop\project\Python/XYZCorp_LendingData.txt',
                          sep = '\t',na_values = 'NaN',low_memory = False);
                
#%%
#printing first few rows                      
print(loan_credit.head(4))
#trying to know number of rows and columns
print("Number of rows and columns:");
(loan_credit.shape)
#%%
loan_credit.info()

#%%
#Creating the column period so that we can split the data according to the given 
#problem statement that is test and train
loan_1 = loan_credit
# trying to split the data as ['Jan','2010']
loan_1['str_split'] = loan_1.issue_d.str.split('-')
#trying to fetch only th month part i.e. jan,feb,march...etc
loan_1['issue'] = loan_1.str_split.str.get(0)
#trying to fetch only the year part
loan_1['d']=loan_1.str_split.str.get(1)
#trying to assign values to months
loan_1['issue'] = loan_1['issue'].replace({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                                              'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
                                  ,regex = True)
#trying to concatenate year+month
loan_1["period"] = loan_1["d"].map(str) + loan_1["issue"]
loan_1 = loan_1.sort_values('period')
loan_2 = loan_1
#%%
loan_2.to_csv(r'C:\Users\dell\Desktop\python\loan_data_original.csv',
          index=False,header=True)
#%%
#trying to set period as index column for splitting purpose
loan_2 = loan_2.set_index('period')
#%%

#Bad rate
pos = loan_2[loan_2["default_ind"] == 1].shape[0]
neg = loan_2[loan_2["default_ind"] == 0].shape[0]
print(f"Class 1 = {pos}")
print(f"Class 0 = {neg}")
print(f"Bad rate = {(pos / (pos+neg)) * 100:.2f}%")
plt.figure(figsize=(8, 6))
sns.countplot(loan_2["default_ind"])
plt.xticks((0, 1), ["Paid fully", "Not paid fully"])
plt.xlabel("")
plt.ylabel("Count")
plt.title("Class counts", y=1, fontdict={"fontsize": 20});

#%%

#trying to find the columns having missing values
loan_2.isnull().sum()
#%%
# trying to replace ?,.,/ any with nan so that missing value can be computed
loan_2=loan_2.replace(['?','.','/'], np.nan)
#again trying to find total number of missing values after replacement
loan_2.isnull().sum()



#%%
# trying to list columns having misssing values more than 50%
#get_ipython is used for plotting bar chart
get_ipython().magic('matplotlib inline')
#trying to assign missing value column to NA_col
NA_col = loan_2.isnull().sum()
#trying to find columns having greater than 50% of missing values
NA_col = NA_col[NA_col.values >=(0.50*len(loan_2))]
#trying plot bar chart
plt.figure(figsize=(20,4))
plt.title('List of Columns & NA counts where NA values are more than 50%')
NA_col.plot(kind='bar')
plt.show()
print(NA_col)
#%%

print("\n Number of rows,columns before dropping columns",loan_2.shape)
fifty=0.50*len(loan_2)
#dropping the columns having missing values greater than 50 %
loan_2=loan_2.dropna(how='all',axis=1,thresh=fifty);
print("\n Number of rows,columns after dropping missing value columns",loan_2.shape)
#%%
loan_2.info()
loan_2.isnull().sum()
#inq_last_6mths
#%%
print("\n Number of rows,columns before dropping irrelevent columns",loan_2.shape)
#dropping off irrelevant columns from data frame
loan_2=loan_2.drop(['issue_d','str_split','issue','d','id','member_id','title','zip_code','pymnt_plan','emp_title',
                    'addr_state','collection_recovery_fee','collections_12_mths_ex_med','funded_amnt_inv','sub_grade',
                    'initial_list_status','out_prncp_inv','policy_code','title','total_pymnt_inv',
                    'tot_coll_amt','next_pymnt_d','funded_amnt' ],axis=1)


print("\n Number of rows,columns after dropping irrelevent columns",loan_2.shape)
#checking data type and columns left after dropping

#%%
loan_2.isnull().sum()
#%%



#replacing the missing value with mean and mode

for x in loan_2.columns[:]:
    if loan_2[x].dtype=='object':
        loan_2[x].fillna(loan_2[x].mode()[0],inplace=True)
    elif loan_2[x].dtype=='int64':
        loan_2[x].fillna(round(loan_2[x].mean()),inplace=True)
    elif loan_2[x].dtype=='float64':
        loan_2[x].fillna(loan_2[x].mean(),inplace=True)

loan_2.isnull().sum()
#%%
loan_2.info()

#%%

loan_3=loan_2
#%%
#trying to assign dummy values using label encoder for nominal varibales
colname=['earliest_cr_line','last_credit_pull_d','last_pymnt_d','term','application_type',
         'verification_status','home_ownership','purpose']

from sklearn import preprocessing

le={}

le=preprocessing.LabelEncoder()

for x in colname:
     loan_2[x]=le.fit_transform(loan_2[x])
     
    

#%%                                      

#trying to assign user defined values to ordinal varibales   

loan_2['emp_length']=loan_2['emp_length'].replace({
        '10+ years': 10,
        '9 years': 9,
        '8 years': 8,
        '7 years': 7,
        '6 years': 6,
        '5 years': 5,
        '4 years': 4,
        '3 years': 3,
        '2 years': 2,
        '1 year': 1,
        '< 1 year': 0,
        'n/a': 0
    },
regex=False)

loan_2['grade']=loan_2['grade'].replace({"A": 1,
"B": 2,
"C": 3,
"D": 4,
"E": 5,
"F": 6,
"G": 7},
regex=False)

#%%
loan_2.info()
#%%

#finding correlation between variables


corr = loan_2.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#%%

cor = loan_2.corr()
cor.loc[:,:] = np.tril(cor, k=-1) # below main lower triangle of an array
cor = cor.stack()
cor[(cor > 0.55) | (cor < -0.55)]        

            
#%%

#dropping highly correlated columns

loan_2=loan_2.drop(['total_rec_prncp','total_rec_int','total_rev_hi_lim'], axis=1)


print("Data types and their frequency\n{}".format(loan_2.dtypes.value_counts()))  

#%%

loan_2.to_csv(r'C:\Users\dell\Desktop\python\loan_data_final.csv',
          index=False,header=True)          
            
#%%
#splitting data into train and test
Train_data = loan_2.loc['200706':'201505',:]
Test_data = loan_2.loc['201506':'201512',:]





#%%

# splitting the data into x_train and y_train
X_train = Train_data.values[:,:-1]
Y_train = Train_data.values[:,-1]

X_test = Test_data.values[:,:-1]
Y_test = Test_data.values[:,-1]


#%%


#standardizing the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




#%%

#converting y into int type
Y_train=Y_train.astype(int)
Y_test=Y_test.astype(int)


#%%
print("Data types and their frequency\n{}".format(loan_2.dtypes.value_counts()))            


#%%
#Creating logistic Regression model
from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

#print(classifier.coef_)
#print(classifier.intercept_)

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("Logistic Regression :")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%
from sklearn.tree import DecisionTreeClassifier
model_DecisionTree=DecisionTreeClassifier(random_state=10)

#fit the model on the data and predict the values
model_DecisionTree.fit(X_train,Y_train)


#%%
Y_pred=model_DecisionTree.predict(X_test)
#print(Y_pred)
#print(list(zip(Y_test,Y_pred)))
#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("Decision Tree:")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%
from sklearn import tree
with open("model_DecisionTree.txt", "w") as f:
    f = tree.export_graphviz(model_DecisionTree, feature_names= colname[:-1],out_file=f)
    
##generate the file and upload the code in webgraphviz.com to plot the decision tree


#%%
#Accuracy of decision tree is  around 97.63% and type 1 error is more therefore we try out forrandom forest
#random forest classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(100,random_state=10)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)
#%%
Y_pred=model_RandomForest.predict(X_test)

 #%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

print("Random Forest:")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%
#Accuracy of random forest is also 82.23% and type 1 error is more therefore we try gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)
#%%
Y_pred=model_GradientBoosting.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

print("Gradient Boosting:")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%%
Y_pred_col=list(Y_pred)
#%%

Test_data.to_csv(r'C:\Users\dell\Desktop\python\loan_data_test.csv',
          index=False,header=True)


#%%

for x in range(0,len(Y_pred_col)):

    if Y_pred_col[x]==0:
        Y_pred_col[x]= "N"
    else:
        Y_pred_col[x]="Y"
    
print(Y_pred_col)
#%%
test_data=pd.read_csv(r'C:\Users\dell\Desktop\python\loan_data_test.csv',
                      header=0)
test_data["Y_predictions"]=Y_pred_col
test_data.head()
#%%
test_data.to_csv('test_data.csv')
#%%
