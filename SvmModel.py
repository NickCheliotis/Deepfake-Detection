
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



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

X = df.drop(columns=["filename", "label","type"])
y = df["label"]


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])


param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100, 1000],
    'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
    'svm__kernel': ['rbf', 'poly', 'sigmoid', 'linear']
}


grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
grid.fit(X, y)


# print("Best Score: {:.2f}%".format(grid.best_score_ * 100))
# print("Best Params:", grid.best_params_)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
grid_best_model = grid.best_estimator_
grid_best_model.fit(X_train, y_train)
y_pred = grid_best_model.predict(X_test)


print("\n Final Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["real", "deepfake"]))



#Accuracy per type
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
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "deepfake"])
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix - SVM")
# plt.show()




