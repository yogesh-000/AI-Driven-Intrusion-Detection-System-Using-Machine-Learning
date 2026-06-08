import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
data_path = r"D:\AI-Driven-Intusion-Detection-Using-Machine-Learning--main\datasets\boit_processed.csv"

df = pd.read_csv(data_path)

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)


# --------------------------------------------------
# 2. Features and Target
# --------------------------------------------------
y = df["attack"]
X = df.drop("attack", axis=1)

# Fix LightGBM feature name issue
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])


# --------------------------------------------------
# 4. Algorithms
# --------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "LightGBM": lgb.LGBMClassifier(),
    "MLP": MLPClassifier(max_iter=300)
}


# --------------------------------------------------
# 5. Training and Evaluation
# --------------------------------------------------
performance_results = []

for name, model in models.items():

    print("\n==========================================")
    print("Algorithm :", name)
    print("==========================================")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    performance_results.append([name, acc, prec, rec, f1])

    print(f"\nAccuracy : {acc*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 6. Performance Summary Table
# --------------------------------------------------
results_df = pd.DataFrame(
    performance_results,
    columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score"]
)

results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n==========================================")
print("ALGORITHM PERFORMANCE TABLE")
print("==========================================")

print(results_df.to_string(index=False))


# --------------------------------------------------
# 7. Best Model
# --------------------------------------------------
best_model = results_df.iloc[0]

print("\nBest Performing Algorithm :", best_model["Algorithm"])
print("Highest Accuracy :", f"{best_model['Accuracy']*100:.2f}%")

print("\nEvaluation Completed Successfully")
