import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("Iris.csv)

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", PetalLengthCm"]]
y = df["Species"]

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_stage=42)

model= DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precison = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("Precison:", precison)
print("Recall:", recall)
                 
