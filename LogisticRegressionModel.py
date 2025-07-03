import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Detect type based on filename
def detect_type(filename, label):
    if 'audio' in filename and 'video' in filename:
        return 'video+audio'
    elif 'audio' in filename:
        return 'audio'
    elif 'video' in filename:
        return 'video'
    else:
        return 'real'


df = pd.read_csv("final_feature_dataset.csv")
df['type'] = df.apply(lambda row: detect_type(row['filename'], row['label']), axis=1)


X = df.drop(columns=["filename", "label", "type"])
y = df["label"]


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__penalty': ['l2'],
    'logreg__solver': ['lbfgs']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X, y)




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


best_model = grid.best_estimator_
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)


print("\nFinal Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["real", "deepfake"]))



# print(" Accuracy:", accuracy_score(y_test, y_pred))
# print("\n Classification Report:\n")
# print(classification_report(y_test, y_pred, target_names=["real", "deepfake"]))


test_df = X_test.copy()
test_df['true_label'] = y_test.values
test_df['pred_label'] = y_pred
test_df['type'] = df.loc[y_test.index, 'type'].values
sns.barplot(
    data=test_df,
    x='type',
    y=(test_df['true_label'] == test_df['pred_label']),
    estimator=np.mean
)
plt.ylabel("Accuracy")
plt.title("Model Accuracy by Deepfake Type")
plt.show()



#Confusion matrix
# cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
#
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "deepfake"])
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix - Logistic Regression")
# plt.show()