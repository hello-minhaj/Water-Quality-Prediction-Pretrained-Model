import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve,f1_score
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import pickle

from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# load Dataset 
df=pd.read_csv('water_potability.csv')

#define target and features
target_col = "Potability"
numeric_cols = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# outlier handeling
for col in numeric_cols:
  q1=df[col].quantile(0.25)
  q3=df[col].quantile(0.75)
  iqr=q3-q1
  upper_limit=q3+(1.5*iqr)
  lower_limit=q1-(1.5*iqr)

  df[col]=df[col].clip(lower_limit,upper_limit)

# there is no categorical value so we only work on numerical value impute them and scaling

#pipeline
#for numerical features
num_transformer = Pipeline (
    steps = [
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# -----------------------------------------------
# base leaner

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=2)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced',random_state=42)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, random_state=42)

# Support Vector Classifier
svc = SVC(kernel='rbf', C=0.1, gamma='scale', random_state=42)
# ---------------------------------------------------

models={
    'Logistic Regression':lr,
    'K-Nearest Neighbors':knn,
    'Decision Tree':dt,
    'Random Forest':rf,
    'Gradient Boosting':gb,
    'Support Vector Classifier':svc
}

# model trainig and testing
X=df.drop('Potability',axis=1)
y=df['Potability']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


#testing models accuracy
acc_score=[]
for model_name,model in models.items():
  pipe=Pipeline(
      [
          ('num_transformer',num_transformer),
          ('model',model)
      ]
  )
  pipe.fit(X_train,y_train)
  y_pred=pipe.predict(X_test)
  acc=accuracy_score(y_test,y_pred)
  f1=f1_score(y_test,y_pred)

  acc_score.append([model_name,acc,f1])

acc_df=pd.DataFrame(acc_score,columns=['Model','Accuracy','F1 Score'])
acc_df=acc_df.sort_values(by='F1 Score',ascending=False)
acc_df

best_model=acc_df.iloc[0,0]
best_model

# cross validation
best_model = models[best_model]

pipe = Pipeline([
    ('num_transformer', num_transformer),
    ('model', best_model)
])

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=10,
    scoring='f1'
)

param_grid = {
    "n_estimators": [100, 150, 200, 250, 300],
    "max_depth": [5, 10, 15, 20, 25],
    "class_weight": ['balanced','balanced_subsample']
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=10,
    scoring='f1',
    n_jobs=-1
)

grid.fit(X_train, y_train)


best_rf = grid.best_estimator_

model=best_rf.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Accuracy:", acc)
print("F1 Score:", f1)

with open('water_quality.pkl','wb') as file:
  pickle.dump(model,file)

print("Rnadom forest pipeline saved as water_quality.pkl")