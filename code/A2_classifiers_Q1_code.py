# -*- coding: utf-8 -*-
"""
Created On : Fri Oct 15 2021
Last Modified : Fri Oct 22 2021
Course : MSBD5002 
Assignment : Assignment 02 Question 01

Remarks:
    - This Code will print out the following tables:
        - Table for Decision Tree (Entropy)
        - Table for Decision Tree (Gini)
        - Table for K-NearestNeighbors (weights = 'uniform')
        - Table for K-NearestNeighbors (weights = 'distance')
        - Table for Random Forest (Entropy)
        - Table for Random Forest (Gini)
    - This Code will also generate txt files containing the main classification metrics of each Classifier
        - DecisionTree_Metrics_Report.txt
        - KNN_Metrics_Report.txt
        - RandomForest_Metrics_Report.txt
    - The generated txt files report is to illustrate which averages was selected for the Comparison Summary

"""
## For all Models
import time
import pandas as pd
import numpy as np
## For Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
## For KNN Model
from sklearn.neighbors import KNeighborsClassifier
## For Random Forests
from sklearn.ensemble import RandomForestClassifier


####################################################
##### Extract the Training and Testing Dataset #####
df_wine_train = pd.read_csv("winequality_train.csv", sep=';')
df_wine_test = pd.read_csv("winequality_test.csv", sep=';')

## For Training ##
# extract all features columns excluding quality for attributes
features_list_train = [x for x in df_wine_train.columns.tolist() if x != 'quality']
x_wine_features_train = df_wine_train[features_list_train]
y_wine_label_train = df_wine_train['quality']

## For Testing ##
# extract all features columns excluding quality for attributes
features_list_test = [x for x in df_wine_test.columns.tolist() if x != 'quality']
x_wine_features_test = df_wine_test[features_list_test]
y_wine_label_test = df_wine_test['quality']
####################################################




####################################################
##### Classifiers Model = Decision Tree 
####################################################
print("\n\n<< Decision Tree (Entropy and Gini) >>")

# Parameter for Decision Tree Classifier
# Maximum depth of the tree
depth_tree = [5,10,15,20]
# parameter 'average' for each classification metrics
averagedf = 'weighted' 
data_entropy  = [] # Store the Metrics information for Decision Tree with Entropy to be printed as DataFrame table
data_gini = []  # Store the Metrics information for Decision Tree with Entropy to be printed as DataFrame table
for i in depth_tree:
    max_depth = i # Set the max depth for each iteration

    ### Build Decision Tree with Entropy
    # Train the Decision Tree
    start = time.time() # Start of the training time
    clf_decisiontree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
    clf_decisiontree_entropy = clf_decisiontree_entropy.fit(x_wine_features_train.values,y_wine_label_train.values)
    stop = time.time() # End of the training time
    # Predict the class for test dataset 
    y_wine_label_predict_tree_e = clf_decisiontree_entropy.predict(x_wine_features_test.values)
    # Convert y_label_testing into array
    y_wine_label_test_df = np.array(y_wine_label_test)
    
    ### Evaluate Models Accuracy and other Metrics
    # accuracy_score
    accuracy_score_e = metrics.accuracy_score(y_wine_label_test_df, y_wine_label_predict_tree_e )
    # precision
    precision_e = metrics.precision_score(y_wine_label_test_df, y_wine_label_predict_tree_e, average = averagedf, zero_division=0)
    # recall
    recall_e = metrics.recall_score(y_wine_label_test_df, y_wine_label_predict_tree_e, average = averagedf, zero_division=0)
    # f1 score
    f1_score_e  = metrics.f1_score(y_wine_label_test_df, y_wine_label_predict_tree_e, average = averagedf, zero_division=0)
    # training time
    time_seconds_e = stop-start
    
    ### Store the Information for DataFrame Display
    data_entropy.append([i, round(accuracy_score_e,4),round(precision_e,4),round(recall_e,4),round(f1_score_e,4),round(time_seconds_e,4)])

    
    ### Build Decision Tree with Gini
    # Train the Decision Tree
    start = time.time() # Start of the training time
    clf_decisiontree_gini = DecisionTreeClassifier(criterion="gini", max_depth=max_depth)
    clf_decisiontree_gini = clf_decisiontree_gini.fit(x_wine_features_train.values,y_wine_label_train.values)
    stop = time.time()  # End of the training time
    # Predict the class for test dataset 
    y_wine_label_predict_tree_g = clf_decisiontree_gini.predict(x_wine_features_test.values)
    # Convert y_label_testing into array
    y_wine_label_test_df = np.array(y_wine_label_test)
    
    ### Evaluate Models Accuracy and other Metrics
    # accuracy_score
    accuracy_score_g = metrics.accuracy_score(y_wine_label_test_df, y_wine_label_predict_tree_g )
    # precision
    precision_g = metrics.precision_score(y_wine_label_test_df, y_wine_label_predict_tree_g, average = averagedf, zero_division=0)
    # recall
    recall_g = metrics.recall_score(y_wine_label_test_df, y_wine_label_predict_tree_g, average = averagedf, zero_division=0)
    # f1 score
    f1_score_g  = metrics.f1_score(y_wine_label_test_df, y_wine_label_predict_tree_g, average = averagedf, zero_division=0)
    # training time
    time_seconds_g = stop-start
    
    ### Store the Information for DataFrame Display
    data_gini.append([i, round(accuracy_score_g,4),round(precision_g,4),round(recall_g,4),round(f1_score_g,4),round(time_seconds_g,4)])    


#### DataFrame Display 
#### Classifier Model = Decision Tree 
col = ['Max Depth','Accuracy Score','Precision','Recall','f1_score','Training Time']
df_table_entropy = pd.DataFrame(data_entropy,columns = col)
df_table_entropy.set_index('Max Depth',inplace=True)
print("\nTable for Decision Tree (Entropy)" )
print("-"*70 )
print(df_table_entropy )

df_table_gini = pd.DataFrame(data_gini,columns = col)
df_table_gini.set_index('Max Depth', inplace=True)
print("\nTable for Decision Tree (Gini)" )
print("-"*70 )
print(df_table_gini)


#### Generate a metrics report (.txt file) of the main classification metrics of Decision Tree Entropy and Gini
f = open("DecisionTree_Metrics_Report.txt", "w") 
print("\nTable for Decision Tree (Entropy)" , file=f)
DF_report_tree_e = metrics.classification_report(y_wine_label_test_df, y_wine_label_predict_tree_g ,zero_division = 0)
print(DF_report_tree_e, file=f)
print("\nTable for Decision Tree (Gini)" , file=f)
DF_report_tree_g = metrics.classification_report(y_wine_label_test_df, y_wine_label_predict_tree_g  ,zero_division = 0)
print(DF_report_tree_g, file=f)
f.close()




####################################################
##### Classifiers Model = K-NearestNeighbors (KNN) #
####################################################
print("\n\n<< K-Nearest Neighbors >>")

# Parameter for K-Nearest Neighbor Classifier
k_neighbour = [5,7,9,11,13,15]
# parameter 'average' for each classification metrics
averageK = 'weighted'
data_KNN_uniform = [] # Store the Metrics information for KNN with weights as 'uniform' to be printed as DataFrame table
data_KNN_distance = [] # Store the Metrics information for KNN with weights as 'distance' to be printed as DataFrame table
for k in k_neighbour:
    
    ### Model KNN with weights parameter set to default 'uniform' 
    ### (All points in each neighborhood are weighted equally)
    weights_val = 'uniform' # This is the default 
    start = time.time()
    clf_KNN_u = KNeighborsClassifier(n_neighbors = k, weights= weights_val)
    clf_KNN_u.fit(x_wine_features_train.values, y_wine_label_train.values)
    stop = time.time()
    # Predict the class for test dataset 
    y_wine_label_predict_knn_u = clf_KNN_u.predict(x_wine_features_test.values)
    # Convert y_label_testing into array
    y_wine_label_test_K = np.array(y_wine_label_test)
    
    ### Evaluate Models Accuracy and other Metrics
    # accuracy_score
    accuracy_score_knn_u = metrics.accuracy_score(y_wine_label_test_K, y_wine_label_predict_knn_u)
    # precision
    precision_knn_u = metrics.precision_score(y_wine_label_test_K, y_wine_label_predict_knn_u, average = averageK, zero_division=0)
    # recall
    recall_knn_u = metrics.recall_score(y_wine_label_test_K, y_wine_label_predict_knn_u, average = averageK, zero_division=0)
    # f1 score
    f1_score_knn_u  = metrics.f1_score(y_wine_label_test_K, y_wine_label_predict_knn_u, average = averageK, zero_division=0)
    # training time
    time_seconds_knn_u = stop-start
    
    ### Store the Information for DataFrame Display
    data_KNN_uniform.append([k, round(accuracy_score_knn_u,4),round(precision_knn_u,4),round(recall_knn_u,4),round(f1_score_knn_u,4),round(time_seconds_knn_u,4)])


    ### Model KNN with weights parameter set to default 'distance' 
    ### (Closer neighbors of a query point will have a greater influence)
    weights_val = 'distance' # This is the default 
    start = time.time()
    clf_KNN_d = KNeighborsClassifier(n_neighbors = k, weights= weights_val)
    clf_KNN_d.fit(x_wine_features_train.values, y_wine_label_train.values)
    stop = time.time()
    # Predict the class for test dataset 
    y_wine_label_predict_knn_d = clf_KNN_d.predict(x_wine_features_test.values)
    # Convert y_label_testing into array
    y_wine_label_test_K = np.array(y_wine_label_test)
    
    ### Evaluate Models Accuracy and other Metrics
    # accuracy_score
    accuracy_score_knn_d = metrics.accuracy_score(y_wine_label_test_K, y_wine_label_predict_knn_d)
    # precision
    precision_knn_d = metrics.precision_score(y_wine_label_test_K, y_wine_label_predict_knn_d, average = averageK, zero_division=0)
    # recall
    recall_knn_d = metrics.recall_score(y_wine_label_test_K, y_wine_label_predict_knn_d, average = averageK, zero_division=0)
    # f1 score
    f1_score_knn_d  = metrics.f1_score(y_wine_label_test_K, y_wine_label_predict_knn_d, average = averageK, zero_division=0)
    # training time
    time_seconds_knn_d = stop-start
    
    ### Store the Information for DataFrame Display
    data_KNN_distance.append([k, round(accuracy_score_knn_d,4),round(precision_knn_d,4),round(recall_knn_d,4),round(f1_score_knn_d,4),round(time_seconds_knn_d,4)])


#### DataFrame Display 
#### Classifier Model = K-NearestNeighbors
col_KNN = ['K value','Accuracy Score','Precision','Recall','f1_score','Training Time']
df_table_KNN_uniform = pd.DataFrame(data_KNN_uniform,columns = col_KNN)
df_table_KNN_uniform.set_index('K value',inplace=True)
print("\nTable for K-NearestNeighbors (weights function = 'uniform')" )
print("-"*70 )
print(df_table_KNN_uniform )

df_table_KNN_distance = pd.DataFrame(data_KNN_distance,columns = col_KNN)
df_table_KNN_distance.set_index('K value',inplace=True)
print("\nTable for K-NearestNeighbors (weights function = 'distance')" )
print("-"*70 )
print(df_table_KNN_distance )


#### Generate a metrics report (.txt file) of the main classification metrics of KNN with weights parameter set to 'uniform' or 'distance'
f = open("KNN_Metrics_Report.txt", "w") 
print("\nTable for K Nearest Neighbor (weights function = 'uniform')" , file=f)
DF_report_knn_u = metrics.classification_report(y_wine_label_test_K, y_wine_label_predict_knn_u, zero_division = 0)
print(DF_report_knn_u, file=f)

print("\nTable for K Nearest Neighbor (weights function = 'distance')" , file=f)
DF_report_knn_d = metrics.classification_report(y_wine_label_test_K, y_wine_label_predict_knn_d, zero_division = 0)
print(DF_report_knn_d, file=f)
f.close()




####################################################
##### Classifiers Model = Random Forest ############
####################################################
print("\n\n<< Random Forest (Entropy and Gini) >>" )

# Parameter for Random Forest Classifier
num_estimators = [10,100,200,250]
# parameter 'average' for each classification metrics
average_rf = 'weighted'
data_rf_entropy = [] # Store the Metrics information for Random Forest with Entropy to be printed as DataFrame table
data_rf_gini = [] # Store the Metrics information for Random Forest with Gini to be printed as DataFrame table
for num in num_estimators:

    ### Build a Random Forest with Entropy
    # Train the Random Forest
    start = time.time()
    clf_randomforest_e = RandomForestClassifier(criterion = 'entropy',n_estimators=num)
    clf_randomforest_e.fit(x_wine_features_train.values, y_wine_label_train.values)
    stop = time.time()
    # Predict the class for test dataset 
    y_wine_label_predict_RF_e = clf_randomforest_e.predict(x_wine_features_test.values)
    # Convert y_label_testing into array
    y_wine_label_test_rf = np.array(y_wine_label_test)
    
    # Evaluate Models Accuracy and other Metrics
    # accuracy_score
    accuracy_score_rf_e = metrics.accuracy_score(y_wine_label_test_rf, y_wine_label_predict_RF_e)
    # precision
    precision_rf_e = metrics.precision_score(y_wine_label_test_rf, y_wine_label_predict_RF_e, average = average_rf, zero_division=0)
    # recall
    recall_rf_e = metrics.recall_score(y_wine_label_test_rf, y_wine_label_predict_RF_e, average = average_rf, zero_division=0)
    # f1 score
    f1_score_rf_e  = metrics.f1_score(y_wine_label_test_rf, y_wine_label_predict_RF_e, average = average_rf, zero_division=0)
    # training time
    time_seconds_rf_e = (stop-start)
    
    # For DataFrame Display
    data_rf_entropy.append([num, round(accuracy_score_rf_e,4),round(precision_rf_e,4),round(recall_rf_e,4),round(f1_score_rf_e,4),round(time_seconds_rf_e,4)])

    
    ### Build a Random Forest with Gini
    # Train the Random Forest
    start = time.time()
    clf_randomforest_g = RandomForestClassifier(criterion = 'gini',n_estimators=num)
    clf_randomforest_g.fit(x_wine_features_train.values, y_wine_label_train.values)
    stop = time.time()
    # Predict the class for test dataset 
    y_wine_label_predict_RF_g = clf_randomforest_g.predict(x_wine_features_test.values)
    # Convert y_label_testing into array
    y_wine_label_test_rf = np.array(y_wine_label_test)
    
    # Evaluate Models Accuracy and other Metrics
    # accuracy_score
    accuracy_score_rf_g = metrics.accuracy_score(y_wine_label_test_rf, y_wine_label_predict_RF_g)
    # precision
    precision_rf_g = metrics.precision_score(y_wine_label_test_rf, y_wine_label_predict_RF_g, average = average_rf, zero_division=0)
    # recall
    recall_rf_g = metrics.recall_score(y_wine_label_test_rf, y_wine_label_predict_RF_g, average = average_rf, zero_division=0)
    # f1 score
    f1_score_rf_g  = metrics.f1_score(y_wine_label_test_rf, y_wine_label_predict_RF_g, average = average_rf, zero_division=0)
    # training time
    time_seconds_rf_g = (stop-start)

    # For DataFrame Display
    data_rf_gini.append([num, round(accuracy_score_rf_g,4),round(precision_rf_g,4),round(recall_rf_g,4),round(f1_score_rf_g,4),round(time_seconds_rf_g,4)])


#### DataFrame Display 
#### Classifiers Model = Random Forest 
col_rf = ['Number of Trees','Accuracy Score','Precision','Recall','f1_score','Training Time']
df_table_RF_entropy = pd.DataFrame(data_rf_entropy,columns = col_rf)
df_table_RF_entropy.set_index('Number of Trees',inplace=True)
print("\nTable for Random Forest (Entropy)" )
print("-"*80 )
print(df_table_RF_entropy )

df_table_RF_gini = pd.DataFrame(data_rf_gini,columns = col_rf)
df_table_RF_gini.set_index('Number of Trees',inplace=True)
print("\nTable for Random Forest (Gini)" )
print("-"*80 )
print(df_table_RF_gini )


#### Generate a metrics report (.txt file) of the main classification metrics of Random Forest with Entropy and Gini
f = open("RandomForest_Metrics_Report.txt", "w") 
print("\nTable for Random Forest (Entropy)" , file=f)
DF_report_RF_e = metrics.classification_report(y_wine_label_test_rf, y_wine_label_predict_RF_e, zero_division = 0)
print(DF_report_RF_e, file=f)

print("\nTable for Random Forest (Gini)" , file=f)
DF_report_RF_g = metrics.classification_report(y_wine_label_test_rf, y_wine_label_predict_RF_g, zero_division = 0)
print(DF_report_RF_g, file=f)
f.close()











