# =============================
# ✅ SkillCraft Task 03 - Final Code with Graphs
# =============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# =============================
# ✅ Step 1: Load Dataset
# =============================
df = pd.read_csv("bank-additional-full.csv", sep=';')
print("Dataset Loaded Successfully!")
print(df.head())

# =============================
# ✅ Step 2: Encode Categorical Variables
# =============================
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# =============================
# ✅ Step 3: Split Data
# =============================
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# ✅ Step 4: Train Model
# =============================
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# =============================
# ✅ Step 5: Predictions & Evaluation
# =============================
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =============================
# ✅ Step 6: Feature Importance Graph
# =============================
plt.figure(figsize=(10,5))
sns.barplot(x=model.feature_importances_, y=X_train.columns, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# =============================
# ✅ Step 7: Confusion Matrix Graph
# =============================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# =============================
# ✅ Step 8: Decision Tree Visualization
# =============================
plt.figure(figsize=(25,15))
tree.plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['No', 'Yes'])
plt.title("Decision Tree Visualization")
plt.show()