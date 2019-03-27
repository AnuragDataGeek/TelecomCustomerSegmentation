from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import warnings
from sklearn import preprocessing
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

features_to_consider1=['Churn','MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','OverageMinutes','DroppedCalls','BlockedCalls','ReceivedCalls','PeakCallsInOut','OffPeakCallsInOut','MonthsInService']
features_to_consider=['Churn','MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','OverageMinutes','ReceivedCalls','PeakCallsInOut','OffPeakCallsInOut','MonthsInService']
random.seed(1025)
def read_data():
    df=pd.read_csv(os.getcwd()+'/cell2celltrain.csv')
    df=df.head(10000)
    df=df.iloc[:,:25]
    df=df.dropna()
    df.to_csv('test.csv',index=False)

'''    
def prep():
    df=pd.read_csv('test.csv')
    df=df.iloc[:,:25]
    #print(df.isnull().sum())
    df=df.dropna()
    #print(df.shape)
    #print('Features')
    #print(df.columns)
    elbow(df[features_to_consider])
'''

 
def elbow():
    data=pd.read_csv('test.csv')
    data=data[features_to_consider]
    x=data.iloc[:,1:].values
    x=preprocessing.scale(x)
    wcss=[]

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, random_state = 0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    
    #Plotting the results onto a line graph, allowing us to observe 'The elbow'
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') #within cluster sum of squares
    plt.show()

def clustering(): 
    data=pd.read_csv('test.csv')
    data=data[features_to_consider]
    x=data.iloc[:,1:].values
    x_scaled=preprocessing.scale(x)
    kmeans_model=KMeans(n_clusters=4,random_state=258).fit(x_scaled)
    pred=kmeans_model.predict(x_scaled)           
    df2=pd.DataFrame(pred)
    df1=pd.DataFrame(x_scaled)
    print(df1.head())
    print(df2.head())
    data1=pd.concat([df1,df2],axis=1,ignore_index=True)
    data1.columns=['MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','OverageMinutes','ReceivedCalls','PeakCallsInOut','OffPeakCallsInOut','MonthsInService','Cluster label']
    print(data1[data1['Cluster label']==2].describe().iloc[:,1:6])
    #print(data1.head())
    data1.to_csv('clustered_telecom.csv',index=False)
    for i in range(4):
        print("For cluster:",i)
        print(kmeans_model.cluster_centers_[i])
    #liy=list(kmeans_model.cluster_centers_[3])
    lix=['MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','OverageMinutes','ReceivedCalls','PeakCallsInOut','OffPeakCallsInOut','MonthsInService']
    titl=['Basic users','Least revenue generators','Hotspot for targetted marketing','Lucrative customers']
    fig,axes=plt.subplots(nrows=2,ncols=2)
    #for i in range(4):
    i=0
    for row in axes:
        for col in row:
        
            col.plot(lix,list(kmeans_model.cluster_centers_[i]))
            col.set_title(titl[i])
            col.set_ylabel("Cluster centroids")
            col.tick_params(labelrotation=30)
            i+=1
    plt.tight_layout() 
    #plt.plot(lix,liy)
    #plt.title("Lucrative customers")
    #plt.ylabel("Cluster Centroids")
    #plt.xticks(rotation=30)
    plt.show()

if __name__ == '__main__':
    #read_data()
    #elbow()
    clustering()
