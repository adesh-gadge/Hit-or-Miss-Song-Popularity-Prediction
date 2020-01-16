# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:39:55 2019

@author: Adesh
"""


# Reading all the 44 files
import pandas as pd
import glob
path = r'D:\ADS\SEM 3\CIS 787\Project' # use your path
all_files = glob.glob(path + "/*.csv")
li = [] 

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=1)
    li.append(df)
dframe = pd.concat(li, axis=0, ignore_index=True)

# Visualizations
dframe.columns
dframe['Track Name'].nunique()
x=dframe['Track Name'].value_counts()
dframe['Track Name'].value_counts()["One Kiss (with Dua Lipa)"]
dframe=dframe.drop('Position',axis=1)
dframe

######
# Summerizing streams
#####
dframe.columns
sum_ls= dframe.groupby(['Track Name'], as_index=True).sum()
streams_df = pd.DataFrame.from_dict(sum_ls,orient='columns')
streams_df.info()
streams_df.columns
streams_df['Track Name']=streams_df.index

dframe=dframe.drop('Streams',axis=1)
dframe = pd.merge(dframe,streams_df,on='Track Name',how='inner')
f=pd.merge(dframe,streams_df,on='Track Name',how='inner')
f=f.drop_duplicates()



######################################################################
######
# Extracting Track ID
#####

# Extracting Track ID and appending to column in our dataframe
import re
ls=[]
for i in range(len(dframe)):
    line=dframe['URL'].iloc[i]
    if(re.search("https://open.spotify.com/track/(.*)",line) is not None):
                ls.append(re.search("https://open.spotify.com/track/(.*)",line).group(1))
dframe['track_id']= ls

###########################################3
######
# API verification
#####

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

cid ="Enter_Your_ID" 
secret = "Enter_Your_Secret"

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


######
# Extracting Features
#####
# again measuring the time
import timeit
start = timeit.default_timer()
# empty list, batchsize and the counter for None results
rows = []
batchsize = 100
None_counter = 0

for i in range(0,len(dframe['track_id']),batchsize):
    batch = dframe['track_id'][i:i+batchsize]
    feature_results = sp.audio_features(batch)
    for i, t in enumerate(feature_results):
        if t == None:
            None_counter = None_counter + 1
        else:
            rows.append(t)
            
print('Number of tracks where no audio features were available:',None_counter)

stop = timeit.default_timer()
print ('Time to run this code (in seconds):',stop - start)

df_audio_features = pd.DataFrame.from_dict(rows,orient='columns')
df_audio_features.info()
df_audio_features.drop(['analysis_url','track_href','type','uri'], axis=1,inplace=True)
df_audio_features.rename(columns={'id': 'track_id'}, inplace=True)
df = pd.merge(dframe,df_audio_features,on='track_id',how='inner')
df.shape
df = df.drop_duplicates()
df.shape
df.head()



####
# Extracting 10,000 songs
###

# timeit library to measure the time needed to run this code
import timeit
start = timeit.default_timer()

# create empty lists where the results are going to be stored
artist_name = []
track_name = []
popularity = []
track_id = []

for i in range(10000,20000,50):
    track_results = sp.search(q='year:2018', type='track', limit=50,offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        popularity.append(t['popularity'])
      

stop = timeit.default_timer()
print ('Time to run this code (in seconds):', stop - start)

print('number of elements in the track_id list:', len(track_id))

import pandas as pd
df_tracks = pd.DataFrame({'artist_name':artist_name,'track_name':track_name,'track_id':track_id,'popularity':popularity})
print(df_tracks.shape)
df_tracks.head()
#x=df_tracks['popularity'].value_counts()
#x=x.to_frame()
#x['Pop_Value']= x.index
#x.drop(len(x),axis=0)

####
# making isPopular column 
####
df_tracks=df_tracks.assign(isPopular=df_tracks.popularity > 69)
df_tracks['isPopular'].value_counts()
###


#####
# Extracting their features
#####
# again measuring the time
import timeit
start = timeit.default_timer()
# empty list, batchsize and the counter for None results
rows = []
batchsize = 100
None_counter = 0

for i in range(0,len(df_tracks['track_id']),batchsize):
    batch = df_tracks['track_id'][i:i+batchsize]
    feature_results = sp.audio_features(batch)
    for i, t in enumerate(feature_results):
        if t == None:
            None_counter = None_counter + 1
        else:
            rows.append(t)
            
print('Number of tracks where no audio features were available:',None_counter)

stop = timeit.default_timer()
print ('Time to run this code (in seconds):',stop - start)

df_tracks_audio_features = pd.DataFrame.from_dict(rows,orient='columns')
df_tracks_audio_features.info()
df_tracks_audio_features.drop(['analysis_url','track_href','type','uri'], axis=1,inplace=True)
df_tracks_audio_features.rename(columns={'id': 'track_id'}, inplace=True)
data_new = pd.merge(df_tracks,df_tracks_audio_features,on='track_id',how='inner')
data_new.shape
data_new=data_new.drop('track_id',axis=1)
data_new = data_new.drop_duplicates()
# Around 10000 songs and their feataures

data_new.to_csv('12_11_data.csv')


import matplotlib.pyplot as plt
# the histogram of the data
num_bins=20
n, bins, patches = plt.hist(data_new.popularity, num_bins, normed=1, facecolor='blue', alpha=0.5)
import numpy as np
np.percentile(streams_df.Streams, 100)/np.percentile(streams_df.Streams, 3)




###
# Reading the data
###
data = pd.read_csv('D:\ADS\SEM 3\CIS 787\Project\SpotifyAudioFeaturesApril2019.csv', index_col=None, header=0)

#histogram for popularity
n, bins, patches = plt.hist(data.popularity, num_bins, normed=1, facecolor='blue', alpha=0.5)


data=data.assign(isPopular=data.popularity > 69)
data['isPopular'].value_counts()


#distribution of isPopular
val=data['isPopular'].value_counts()
nam=['Not popular','Popular']
y_pos = np.arange(len(nam))
plt.bar(y_pos, val)
plt.xticks(y_pos, nam)
plt.show()


data.info()
data['key'].unique()

a=data.groupby('isPopular').mean()

table=pd.crosstab(data.key,data.isPopular)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')

# No null values
data.isnull().sum()


###
#  Pre-Processing
###

# desired column order
Oncols=['artist_name',
 'track_name',
 'acousticness',
 'danceability',
 'duration_ms',
 'energy',
 'instrumentalness',
 'liveness',
 'loudness',
 'mode',
 'speechiness',
 'tempo',
 'valence',
 'key',
 'time_signature',
 'popularity', 
 'isPopular']

# New Dataframes
data_unseen = data_new[Oncols]
data_seen= data[Oncols]

# splitting seen data into training and testing 

X = data_seen.iloc[:,2:-2]
y = data_seen.iloc[:,-1]



X_unseen = data_unseen.iloc[:,2:-2]
y_unseen = data_unseen.iloc[:,-1]



# Data is already label encoded for categorical variables: key, time signature
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features= [11,12])
#X = onehotencoder.fit_transform(X)

##Encoding the categortical variables
cat_vars=['key','time_signature']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(X[var], prefix=var,drop_first= True)
    data1=X.join(cat_list)
    X=data1
#cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=X.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
X_final = X[to_keep]

##Encoding the categortical variables UNSEEN
cat_vars=['key','time_signature']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(X_unseen[var], prefix=var,drop_first= True)
    data1=X_unseen.join(cat_list)
    X_unseen=data1
#cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=X_unseen.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
X_final_unseen = X_unseen[to_keep]


#Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final,y, test_size=0.3, random_state=0)
columns = X_train.columns
##########
# StandardScaler
#############
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_test= pd.DataFrame(data=X_test,columns=columns)
X_final_unseen = sc_X.transform(X_final_unseen)
X_final_unseen = pd.DataFrame(data=X_final_unseen,columns=columns )
X_all = sc_X.transform(X_final)
X_all =pd.DataFrame(data=X_all,columns=columns)


# Over-sampling using SMOTE
# Implementation of SMOTE

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['isPopular'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['isPopular']==0]))
print("Number of subscription",len(os_data_y[os_data_y['isPopular']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['isPopular']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['isPopular']==1])/len(os_data_X))

# Logistic Model
import statsmodels.api as sm
logit_model=sm.Logit(os_data_y,os_data_X)
result=logit_model.fit()
print(result.summary2())
# removing variables with low p-values (p-value<0.05)
rem_col = ['key_6']
X=os_data_X.drop(rem_col,axis=1)
y=os_data_y


y = pd.DataFrame(data=y,columns=['isPopular'])
#distribution of isPopular
val=y['isPopular'].value_counts()
nam=['Not popular','Popular']
y_pos = np.arange(len(nam))
plt.bar(y_pos, val)
plt.xticks(y_pos, nam)
plt.show()


#FOR UNSEEN
X_unseen=X_final_unseen.drop(rem_col,axis=1)
y_unseen= pd.DataFrame(data=y_unseen,columns=['isPopular'])


logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X, y)

X_test = X_test.drop(rem_col,axis=1)
y_pred = logreg.predict(X_test)

### training accuracy
y_pred_train = logreg.predict(X)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X, y)))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print('Confusion_matrix')
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


######## UNSEEN ########
y_pred_unseen = logreg.predict(X_unseen)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_unseen, y_unseen)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_unseen,y_pred_unseen)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_unseen, y_pred_unseen))
##########################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression Test data (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_unseen, logreg.predict(X_unseen))
fpr, tpr, thresholds = roc_curve(y_unseen, logreg.predict_proba(X_unseen)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression Generalization data (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X,y)
# predictions
rfc_predict = rfc.predict(X_test)
print('Accuracy of RF classifier on Training set: {:.2f}'.format(rfc.score(X,y)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(rfc.score(X_test, y_test)))
rfc_predict= pd.DataFrame(data=rfc_predict,columns=['isPopular'])
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, rfc_predict)
print('ConfusionMatrix')
print(CM)

from sklearn.metrics import classification_report
print(classification_report(y_test, rfc_predict))

######## UNSEEN ########3
y_rfcpred_unseen = rfc.predict(X_unseen)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(rfc.score(X_unseen, y_unseen)))
    
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_unseen,y_rfcpred_unseen)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_unseen, y_rfcpred_unseen))


logit_roc_auc = roc_auc_score(y_test, rfc.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest  (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



logit_roc_auc = roc_auc_score(y_unseen, rfc.predict(X_unseen))
fpr, tpr, thresholds = roc_curve(y_unseen, rfc.predict_proba(X_unseen)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RandomForest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_unseen, rfc.predict(X_unseen))
fpr, tpr, thresholds = roc_curve(y_unseen, rfc.predict_proba(X_unseen)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RandomForest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


## TUNING RANDOM FOREST


## Cross validation score

from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=5)
print(all_accuracies)
print(all_accuracies.mean())
print(all_accuracies.std())


## Grid search
grid_param = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

grid_param = {
    'n_estimators': [90, 100, 110],
    'criterion': ['gini'],
    'bootstrap': [True],
}

grid_param = {
    'n_estimators': [ 10,20,5],
   'criterion': ['gini'],
    'bootstrap': [True],
    'min_samples_split':[2,3]
}

from sklearn.model_selection import GridSearchCV
gd_sr = GridSearchCV(estimator=rfc,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)

gd_sr.fit(X, y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_
print(best_result)

best_model = gd_sr.best_estimator_

len(X_final.columns)
len(best_model.feature_importances_)
rfc.feature_importances_

best_model_predict = best_model.predict(X_test)
print('Accuracy of RF classifier on Training set: {:.2f}'.format(best_model.score(X,y)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(best_model.score(X_test, y_test)))
best_model_predict= pd.DataFrame(data=best_model_predict,columns=['isPopular'])
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, best_model_predict)
print('ConfusionMatrix')
print(CM)

from sklearn.metrics import classification_report
print(classification_report(y_test,best_model_predict))

######## UNSEEN ########3
y_rfcpred_unseen = rfc.predict(X_unseen)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(best_model.score(X_unseen, y_unseen)))
    
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_unseen,y_rfcpred_unseen)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_unseen, y_rfcpred_unseen))





# test
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, best_model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RandomForest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()







# unseen
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_unseen, best_model.predict(X_unseen))
fpr, tpr, thresholds = roc_curve(y_unseen, best_model.predict_proba(X_unseen)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RandomForest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()





# Training on all of the data

# random forest model creation
rfc_final = RandomForestClassifier()
X_all = X_all.drop(rem_col,axis=1)
rfc_final.fit(X_all,y)
best_model = gd_sr.best_estimator_


grid_param = {
    'n_estimators': [90, 100, 110],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True]
}

gd_sr2 = GridSearchCV(estimator=rfc,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)

gd_sr2.fit(X_all, y)
best_model2 = gd_sr2.best_estimator_


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_unseen, best_model2.predict(X_unseen))
fpr, tpr, thresholds = roc_curve(y_unseen, best_model2.predict_proba(X_unseen)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RandomForest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# feature importance

df = pd.DataFrame(list(zip(X.columns,best_model.feature_importances_ )), 
               columns =['Variable', 'Importance']) 
df.sort_values(by='Importance',ascending =False) 