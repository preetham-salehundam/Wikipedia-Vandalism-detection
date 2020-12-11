"""
@author: Preetham Salehundam
@email: salehundam.2@wright.edu

script to run classification
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report
# from sklearn.tree import export_graphviz
# from IPython.display import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    data = pd.read_csv("combined_old.csv")
    data=data.drop(["editor", "newrevisionid", "oldrevisionid", "editid", "bad_words", "character_distribution"], axis=1) #"articleid","articletitle",
    print(data.columns)
    #data.loc[data["character_distribution"]==np.inf] = 0
    data = data.fillna(0)
    corr = data.corr()
    sns.heatmap(corr)
    plt.show()
    y = data["class"]
    data.drop(["class"], axis=1, inplace=True)
    data_x = StandardScaler().fit(data).transform(data)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(data_x, y)

    X_train, X_test = train_test_split(np.array(data_x), test_size=0.2, random_state=123)
    Y_train, Y_test = train_test_split(np.array(y), test_size=0.2, random_state=123)

    rf=RandomForestClassifier(class_weight="balanced", max_depth=5, n_estimators=1)
    lr = LogisticRegression(class_weight="balanced", penalty="l2")
    print(cross_val_score(rf, data_x,y, scoring="f1_macro", cv=5))
    print(cross_val_score(lr, data_x, y, scoring="f1_macro", cv=5))
    i = 0
    for train_index, test_index in skf.split(data_x, y):
        print("iteration ", i)
        i += 1
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        importances = {col: imp for col, imp in zip(data.columns, rf.feature_importances_)}
        print(sorted(importances.items(), key=lambda x: -x[1]))
        importances = {col: imp for col, imp in zip(data.columns, lr.coef_[0])}
        print(sorted(importances.items(), key=lambda x: -x[1]))
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        print("Randomforest-train,\n", classification_report(y_true=y_train, y_pred=y_pred_train))
        print("Randomforest-test,\n", classification_report(y_true=y_test, y_pred=y_pred_test))
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        print("Logistic Regression-train,\n", classification_report(y_true=y_train, y_pred=y_pred_train))
        print("Logistic Regression-test,\n", classification_report(y_true=y_test, y_pred=y_pred_test))
