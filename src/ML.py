import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import pickle
from sklearn.preprocessing import label_binarize

# Load the PCA-transformed data and labels
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
def train_model(X, y, label_name):
    print(f"\n Training SVM for {label_name} classification...")
    # Initialize the SVM classifier
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Train the SVM model with hyperparameter tuning
    model = SVC(probability=True)
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=1)
    grid.fit(x_train, y_train)

    # Evaluate the model
    print(f"Best parameters for {label_name}:", grid.best_params_)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy for {label_name}: {accuracy:.4f}")

    # Save the best model
    with open(f'./model/svm_{label_name}.pickle', 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model, x_test, y_test

def evaluate_model(model, x_test, y_test, label_name):
    y_pred = model.predict(x_test)
    print(f"\nEvaluation for {label_name} classification:")
    # Classification report
    cr = metrics.classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n", pd.DataFrame(cr).T)
    # Kappa score
    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    print(f"Kappa score:", kappa)
    # AUC score
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 2:
        auc = metrics.roc_auc_score(np.where(y_test == unique_classes[1], 1, 0),
                                    np.where(y_pred == unique_classes[1], 1, 0))
        print(f"AUC score:", auc)
    else:
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        y_pred_bin = label_binarize(y_pred, classes=unique_classes)
        auc = metrics.roc_auc_score(y_test_bin, y_pred_bin, average='macro', multi_class='ovr')
        print(f"AUC score (macro):", auc)

# Train and evaluate for gender
model_gender, x_test_gender, y_test_gender = train_model(X, y_gender, 'gender')
evaluate_model(model_gender, x_test_gender, y_test_gender, label_name='gender')

# Train and evaluate for identity
model_identity, x_test_identity, y_test_identity = train_model(X, y_identity, 'identity')
evaluate_model(model_identity, x_test_identity, y_test_identity, label_name='identity')