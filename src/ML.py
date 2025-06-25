import pickle
from typing import Tuple
import numpy as np
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.utils import resample

# Load the PCA transformed data and labels
data = np.load('./data/pca_data_50_target.npz', allow_pickle=True)
X = data['X']
y_gender = data['gender']
y_identity = data['identity']

# Define the parameter grid for SVM hyperparameter tuning
param_grid = {
    'C': [0.5, 1, 10, 20, 30, 50],
    'kernel': ['rbf', 'poly'],
    'gamma': [0.1, 0.05, 0.01, 0.001, 0.002, 0.005],
    'coef0': [0.1]
}

# Function to train the SVM model
def train_model(X: np.ndarray, y: np.ndarray, label_name: str) -> Tuple[SVC, np.ndarray, np.ndarray]:
    print(f"\nTraining SVM model for {label_name} classification...")
    # Intialize the SVM classifier
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Train the SVM model with hyperparameter tuning
    model = SVC(probability=True)
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=1)
    grid.fit(x_train, y_train)

    # Evaluate the model
    print(f"Best parameters for {label_name}: {grid.best_params_}")
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy for {label_name}: {accuracy:.4f}")

    # Save the best model
    with open(f'./model/svm_{label_name}.pickle', 'wb') as f:
        pickle.dump(best_model, f)

    return best_model, x_test, y_test

def balance_classes(X, y):
    # Combine X and y for easier resampling
    data = np.hstack((X, y.reshape(-1, 1)))
    df = pd.DataFrame(data)
    class_counts = df.iloc[:, -1].value_counts()
    min_count = class_counts.min()
    balanced = []
    for cls in class_counts.index:
        cls_samples = df[df.iloc[:, -1] == cls]
        balanced.append(resample(cls_samples, replace=False, n_samples=min_count, random_state=42))
    balanced_df = pd.concat(balanced)
    X_bal = balanced_df.iloc[:, :-1].values
    y_bal = balanced_df.iloc[:, -1].values
    return X_bal, y_bal

# Function to evaluate the trained model
def evaluate_model(model: SVC, x_test: np.ndarray, y_test: np.ndarray, label_name: str) -> None:
    y_pred = model.predict(x_test)
    # Print evaluation metrics
    print(f"\nEvaluation for {label_name} classification:")
    cr = metrics.classification_report(y_test, y_pred, output_dict=True) # Generate classification report
    print("Classification Report:\n", pd.DataFrame(cr).T)
    kappa = metrics.cohen_kappa_score(y_test, y_pred) # Calculate Cohen's Kappa score
    print(f"Kappa score:", kappa)
    unique_classes = np.unique(y_test) # Get unique classes in the test set
    if len(unique_classes) > 2: # If more than two classes, calculate AUC for each class
        auc = metrics.roc_auc_score(np.where(y_test == unique_classes[1], 1, 0),
                                  np.where(y_pred == unique_classes[1], 1, 0))
        print(f"AUC score:", auc)
    else: # If only two classes, calculate macro AUC
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        y_pred_bin = label_binarize(y_pred, classes=unique_classes)
        auc = metrics.roc_auc_score(y_test_bin, y_pred_bin, average='macro', multi_class='ovr')
        print(f"AUC score (macro):", auc)

def main() -> None:
    # Train and evaluate the models
    model_gender, x_test_gender, y_test_gender = train_model(X, y_gender, "gender")
    evaluate_model(model_gender, x_test_gender, y_test_gender, label_name="gender")

    X_bal, y_identity_bal = balance_classes(X, y_identity)
    model_identity, x_test_identity, y_test_identity = train_model(X_bal, y_identity_bal, "identity")
    evaluate_model(model_identity, x_test_identity, y_test_identity, label_name="identity")


if __name__ == "__main__":
    main()  # Run the main function to train and evaluate the models
    print("Training and evaluation completed.")