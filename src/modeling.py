import pandas as pd
df = pd.read_csv("data\german.csv",sep=';')
df.info()
df['target'] = 1 - df['Creditability']
df.drop('Creditability',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X = df.drop('target',axis=1)
y = df['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
numeric_features = [
    "Duration_of_Credit_monthly",
    "Credit_Amount",
    "Age_years",
    "Instalment_per_cent",
    "No_of_Credits_at_this_Bank",
    "No_of_dependents"
]

categorical_features = [cat for cat in X.columns if cat not in numeric_features]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numeric_features),
        ('cat',OneHotEncoder(drop='first'),categorical_features)
    ]
)
model = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',LogisticRegression(max_iter=1000))
])
from sklearn.model_selection import cross_val_score

cv = cross_val_score(
    model,
    X_train,
    y_train,
    cv = 5,
    scoring='roc_auc'
)
model.fit(X_train,y_train)
train_probs = model.predict_proba(X_train)[:,1]
test_probs = model.predict_proba(X_test)[:,1]
from sklearn.metrics import confusion_matrix
import numpy as np

preds = (test_probs>=0.18).astype(int)
tn,fp,fn,tp = confusion_matrix(y_test,preds).ravel()

cm = confusion_matrix(y_test,preds)
print('confusion_matrix:\n',cm)
# Expected cost ( 10:1)
expected_cost = (10*fn) + (1*fp)
# Expected profit (70000:20000) or (3.5:1)
profit = 20000*tn
loss = 70000*fn
total_profit = profit - loss