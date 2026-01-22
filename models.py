import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


# 1. Load dataset

dataset_path = 'datasets/robot_fault_dataset.csv'  # Correct path & filename
data = pd.read_csv(dataset_path)
print("✅ Dataset loaded. Shape:", data.shape)
print("Columns:", data.columns)


# 2. Split features and target

X = data.drop('Fault_Label', axis=1)  # Your target column is 'Fault_Label'
y = data['Fault_Label']


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split: Train={X_train.shape}, Test={X_test.shape}")

# 4. Function to train, evaluate, and plot


def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {acc}")
    print(f"F1-score: {f1}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot confusion heatmap

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    return acc, f1

# 5. Random Forest

rf_model = RandomForestClassifier(random_state=42)
rf_acc, rf_f1 = evaluate_model(rf_model, "Random Forest")

# 6. SVM


svm_model = SVC(random_state=42)
svm_acc, svm_f1 = evaluate_model(svm_model, "SVM")

# 7. Average Accuracy & F1

average_acc = (rf_acc + svm_acc) / 2
average_f1 = (rf_f1 + svm_f1) / 2
print(f"\nAverage Accuracy: {average_acc:.4f}")
print(f"Average F1-score: {average_f1:.4f}")

# 8. Optional: Plot temperature distributions

plt.figure(figsize=(10,5))
sns.histplot(data['Air temperature [K]'], color='orange', kde=True, label='Air Temp [K]')
sns.histplot(data['Process temperature [K]'], color='blue', kde=True, label='Process Temp [K]')
plt.title("Temperature Distributions")
plt.xlabel("Temperature (K)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
