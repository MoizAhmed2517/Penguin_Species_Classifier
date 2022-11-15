import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, plot_confusion_matrix

df = pd.read_csv('penguins_size_clean_data.csv')

# Using decision tree classifiers, as feature are multi-categorical need to do One Hot Encoding.
X = pd.get_dummies(df.drop('species', axis=1))
X = X.drop('sex_FEMALE', axis=1)
X.rename(columns={'sex_MALE': 'Gender'}, inplace=True)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# default model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

base_preds = model.predict(X_test)
# plot_confusion_matrix(model, X_test, y_test)
# print(classification_report(y_test, base_preds))
# plt.show()

# Checking the feature importance in this decision tree.
impFeature = pd.DataFrame(index=X.columns, data=model.feature_importances_,
                          columns=['Feature Importance Val']).sort_values('Feature Importance Val', ascending=False)
# print(impFeature)
# plt.figure(figsize=(12, 8), dpi=200)
# plot_tree(model, feature_names=X.columns, filled=True)
# plt.show()


def report_models(model):
    model_pred = model.predict(X_test)
    print(classification_report(y_test, model_pred))
    print('\n')
    plt.figure(figsize=(12, 8), dpi=200)
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.show()

# Changing hyper-parameter

pruned_tree = DecisionTreeClassifier(max_depth=2)
pruned_tree.fit(X_train, y_train)
#

max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=3)
max_leaf_tree.fit(X_train, y_train)
# report_models(max_leaf_tree)

entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train, y_train)
report_models(entropy_tree)

