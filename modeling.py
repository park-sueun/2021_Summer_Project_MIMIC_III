import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np

missing_value = -999
estimators_rf = 100
estimators_xgb = 50
k_value = 7
random_seed = 7777

datasets = pd.read_csv('timeseries_feature_all.csv')
#print(datasets.dtypes)

id_list = []

cnt = 0

for row in datasets.iterrows():
    for i in row[1:166]:
        if type(i) == float and (pd.isna(i)):
            cnt = cnt+1
    if ((len(row[1:166])-cnt)/len(row[1:166])*100) < 60.0: 
        id_list.append(row[0])
    cnt = 0

for id in id_list:
    datasets = datasets[(datasets['SUBJECT_ID'] != id)]

data = datasets

li = data.drop(['SUBJECT_ID', 'Chronic', 'HEP', 'HCC'], axis=1)
for col in li.iteritems():
    if type(li[col[0]].mean()) == float and (pd.isna(li[col[0]].median())):
        li[col[0]] = li[col[0]].fillna(0) # avg = 0인 경우
    else:
        li[col[0]] = li[col[0]].fillna(li[col[0]].mean()) 

x = li.to_numpy()  
y_hep = data['HEP'].to_numpy()
y_hcc = data['HCC']

#StratifiedKFold()
skf = StratifiedKFold(n_splits=k_value, random_state=random_seed, shuffle=True)
rf_accuracy = []
rf_auroc = []
rf_auprc = []
rf_precision = []
rf_recall = []
rf_f1 = []
xgb_accuracy = []
xgb_auroc = []
xgb_auprc = []
xgb_precision = []
xgb_recall = []
xgb_f1 = []
iter = 0

#model
randomforest = RandomForestClassifier(n_estimators=estimators_rf)
#xgboost = XGBClassifier(n_estimators=estimators_xgb)

for train_index, test_index in skf.split(x, y_hep): #hep
    #train, test
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y_hep[train_index], y_hep[test_index]

    #model_fit
    randomforest.fit(x_train, y_train)
    #xgboost.fit(x_train, y_train)
    # model_predict
    rf_pred = randomforest.predict(x_test)
    rf_prob = randomforest.predict_proba(x_test)

    iter += 1
    #evaluate(Random Forest)
    #accuracy
    rf_accuracy_cross = metrics.accuracy_score(y_test, rf_pred)
    #auroc
    rf_auroc_cross = metrics.roc_auc_score(y_test, rf_pred)
    #auprc
    pre_cross, rec_cross, _ = metrics.precision_recall_curve(y_test, rf_prob[:, 1])
    f1_cross, rf_auprc_cross = metrics.f1_score(y_test, rf_pred), metrics.auc(rec_cross, pre_cross)
    #precision
    rf_precision_cross = metrics.precision_score(y_test, rf_pred)
    #recall
    rf_recall_cross = metrics.recall_score(y_test, rf_pred)
    #f1
    rf_f1_cross = metrics.f1_score(y_test, rf_pred)
    #train_size = x_train.shape[0]
    #test_size = x_test.shape[0]

    #cross validation
    #print(iter, accuracy_cross, train_size, test_size)
    rf_accuracy.append(rf_accuracy_cross)
    rf_auroc.append(rf_auroc_cross)
    rf_auprc.append(rf_auprc_cross)
    rf_precision.append(rf_precision_cross)
    rf_recall.append(rf_recall_cross)
    rf_f1.append(rf_f1_cross)

# print(randomforest.feature_importances_)
df = pd.DataFrame(x, columns = li.columns)
n_features = len(df.columns)
idx = li.columns
# print(x.columns)
# print(df)

ft_importance_values = randomforest.feature_importances_
ft_series = pd.Series(ft_importance_values, index = idx)
ft_top20 = ft_series.sort_values(ascending=False)[:20]

print("------------ Top 20 Features --------------")
plt.figure(figsize=(8,6))
sns.barplot(x=ft_top20, y=ft_top20.index)
plt.show()

#print result(Random Forest)
print("\n[Random Forest]")
#print("accuracy: ", rf_accuracy)
#print("accuracy_avg: ", np.mean(rf_accuracy))
print("auroc_avg: ", np.mean(rf_auroc))
print("auprc_avg: ", np.mean(rf_auprc))
#print("precision: ", rf_precision)
print("precision_avg: ", np.mean(rf_precision))
#print("recall: ", rf_recall)
print("recall_avg: ", np.mean(rf_recall))
#print("f1: ", rf_f1)
print("f1_avg: ", np.mean(rf_f1))
