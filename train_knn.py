"""
Loan Approval Prediction using K-Nearest Neighbors (KNN)
Includes:
- Data Preprocessing (Imputation & Encoding)
- Train/Test Split (80/20) & Feature Scaling
- Default KNN Classifier training & evaluation
- Experimentation with different k values (3, 5, 7, etc.)
- Hyperparameter Tuning with GridSearchCV for n_neighbors
- Evaluation: Accuracy, Confusion Matrix, Classification Report
- Confusion Matrix Visualization (Saved as confusion_matrices.png)
- Model Export for FastAPI / Next.js Web App Deployment
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    print("=" * 70)
    print("LOAN PREDICTION: K-NEAREST NEIGHBORS (KNN) MODEL PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------
    csv_file = "train.csv" if os.path.exists("train.csv") else "Loan_Prediction_Problem.csv"
    print(f"\n[1] Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)

    print("\n--- First 5 rows of the Dataset ---")
    print(df.head())
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # -------------------------------------------------------------
    # 2. Data Preprocessing
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[2] DATA PREPROCESSING")
    print("-" * 50)

    # Check for missing values
    print("\nMissing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Drop Loan_ID as it is an identifier, not a predictive feature
    if 'Loan_ID' in df.columns:
        df_clean = df.drop(columns=['Loan_ID']).copy()
    else:
        df_clean = df.copy()

    # Identify categorical and numerical columns
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    target_col = 'Loan_Status'

    # Fill missing values:
    # Numerical columns with median
    # Categorical columns with mode
    imputation_values = {}
    for col in numerical_cols:
        med = df_clean[col].median()
        imputation_values[col] = float(med)
        df_clean[col] = df_clean[col].fillna(med)

    for col in categorical_cols:
        mode_val = df_clean[col].mode()[0]
        imputation_values[col] = mode_val
        df_clean[col] = df_clean[col].fillna(mode_val)

    print("\nMissing values after imputation:")
    missing_after = df_clean.isnull().sum()
    print("Total missing values across all features:", missing_after.sum())

    # Encode categorical variables into numerical values
    # We define explicit mapping for reproducible inference in FastAPI backend
    category_mappings = {
        'Gender': {'Female': 0, 'Male': 1},
        'Married': {'No': 0, 'Yes': 1},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Not Graduate': 0, 'Graduate': 1},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    }
    target_mapping = {'N': 0, 'Y': 1}

    for col, mapping in category_mappings.items():
        df_clean[col] = df_clean[col].map(mapping).astype(int)

    df_clean[target_col] = df_clean[target_col].map(target_mapping).astype(int)

    print("\nSample Encoded Data (first 5 rows):")
    feature_cols = categorical_cols + numerical_cols
    print(df_clean[feature_cols + [target_col]].head())

    # Separate Features and Target
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # Split dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size:  {X_test.shape[0]} samples")

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # 3. Train Default K-Nearest Neighbors (KNN) Model
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[3] TRAIN K-NEAREST NEIGHBORS (KNN) MODEL")
    print("-" * 50)

    # Train default KNeighborsClassifier (default n_neighbors=5)
    default_knn = KNeighborsClassifier()
    default_knn.fit(X_train_scaled, y_train)

    default_y_pred = default_knn.predict(X_test_scaled)
    default_acc = accuracy_score(y_test, default_y_pred)
    print(f"Default KNN Classifier (k={default_knn.n_neighbors}) Test Accuracy: {default_acc:.4f} ({default_acc*100:.2f}%)")

    # Experiment with different values of k (e.g. 3, 5, 7, 9, 11, 15)
    print("\nExperimenting with different values of k:")
    k_experiments = [3, 5, 7, 9, 11, 13, 15]
    k_results = {}
    for k in k_experiments:
        knn_exp = KNeighborsClassifier(n_neighbors=k)
        knn_exp.fit(X_train_scaled, y_train)
        y_exp_pred = knn_exp.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_exp_pred)
        k_results[k] = acc
        print(f"  k = {k:2d} -> Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # -------------------------------------------------------------
    # 4. Hyperparameter Tuning for KNN (n_neighbors)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[4] HYPERPARAMETER TUNING FOR KNN")
    print("-" * 50)

    param_grid = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_knn = grid_search.best_estimator_
    print(f"Best Parameters from GridSearch: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # Retrain/evaluate tuned model on test set
    tuned_y_pred = best_knn.predict(X_test_scaled)
    tuned_acc = accuracy_score(y_test, tuned_y_pred)
    print(f"Tuned KNN Classifier Test Accuracy: {tuned_acc:.4f} ({tuned_acc*100:.2f}%)")

    # -------------------------------------------------------------
    # 5. Model Evaluation (Default vs Tuned KNN)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[5] MODEL EVALUATION METRICS")
    print("-" * 50)

    cm_default = confusion_matrix(y_test, default_y_pred)
    cm_tuned = confusion_matrix(y_test, tuned_y_pred)

    print("\n=== Model 1: Default KNN (k=5) ===")
    print(f"Accuracy: {default_acc:.4f} ({default_acc*100:.2f}%)")
    print("Confusion Matrix:")
    print(cm_default)
    print("\nClassification Report:")
    print(classification_report(y_test, default_y_pred, target_names=['Rejected (0)', 'Approved (1)']))

    print("\n=== Model 2: Tuned KNN ===")
    print(f"Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {tuned_acc:.4f} ({tuned_acc*100:.2f}%)")
    print("Confusion Matrix:")
    print(cm_tuned)
    print("\nClassification Report:")
    print(classification_report(y_test, tuned_y_pred, target_names=['Rejected (0)', 'Approved (1)']))

    # -------------------------------------------------------------
    # 6. Visualization: Plot Confusion Matrices
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[6] VISUALIZATION OF CONFUSION MATRICES")
    print("-" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')

    # Color customization
    cmap = plt.cm.Blues

    # Default KNN Matrix
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_default, display_labels=['Rejected (N)', 'Approved (Y)'])
    disp1.plot(ax=axes[0], cmap=cmap, colorbar=False)
    axes[0].set_title(f"Default KNN (k={default_knn.n_neighbors})\nAccuracy: {default_acc*100:.2f}%", 
                      color='white', fontsize=14, pad=12, fontweight='bold')
    axes[0].tick_params(colors='white', labelsize=11)
    axes[0].xaxis.label.set_color('white')
    axes[0].yaxis.label.set_color('white')
    for text in disp1.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight('bold')

    # Tuned KNN Matrix
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=['Rejected (N)', 'Approved (Y)'])
    disp2.plot(ax=axes[1], cmap=plt.cm.Greens, colorbar=False)
    axes[1].set_title(f"Tuned KNN (k={best_knn.n_neighbors}, {best_knn.weights})\nAccuracy: {tuned_acc*100:.2f}%", 
                      color='white', fontsize=14, pad=12, fontweight='bold')
    axes[1].tick_params(colors='white', labelsize=11)
    axes[1].xaxis.label.set_color('white')
    axes[1].yaxis.label.set_color('white')
    for text in disp2.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight('bold')

    plt.suptitle("Loan Approval Prediction: Confusion Matrix Comparison", color='white', fontsize=16, y=1.03, fontweight='bold')
    plt.tight_layout()
    output_fig = "confusion_matrices.png"
    plt.savefig(output_fig, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Confusion matrices visualization saved successfully to: {output_fig}")

    # Plot k accuracy curve
    fig_k, ax_k = plt.subplots(figsize=(8, 4.5))
    fig_k.patch.set_facecolor('#0f172a')
    ax_k.set_facecolor('#1e293b')
    ax_k.plot(list(k_results.keys()), [v * 100 for v in k_results.values()], marker='o', linewidth=2.5, color='#38bdf8', markersize=8)
    ax_k.set_title("KNN Accuracy vs. k Value", color='white', fontsize=14, fontweight='bold', pad=10)
    ax_k.set_xlabel("Number of Neighbors (k)", color='white', fontsize=12)
    ax_k.set_ylabel("Test Accuracy (%)", color='white', fontsize=12)
    ax_k.tick_params(colors='white')
    ax_k.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("k_value_accuracy.png", dpi=300, bbox_inches='tight', facecolor=fig_k.get_facecolor())
    plt.close()
    print("k vs. accuracy curve saved to: k_value_accuracy.png")

    # -------------------------------------------------------------
    # 7. Save Model & Preprocessing Artifacts for Web App
    # -------------------------------------------------------------
    os.makedirs("backend", exist_ok=True)
    artifacts = {
        'model': best_knn,
        'default_model': default_knn,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'category_mappings': category_mappings,
        'target_mapping': target_mapping,
        'imputation_values': imputation_values,
        'metrics': {
            'default_accuracy': float(default_acc),
            'tuned_accuracy': float(tuned_acc),
            'best_params': grid_search.best_params_,
            'k_experiments': {int(k): float(v) for k, v in k_results.items()}
        }
    }

    joblib.dump(artifacts, "backend/loan_knn_model.joblib")
    joblib.dump(artifacts, "loan_knn_model.joblib")
    
    # Save a JSON summary for frontend or API consumers
    summary_path = "backend/model_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            'default_accuracy': float(default_acc),
            'tuned_accuracy': float(tuned_acc),
            'best_params': grid_search.best_params_,
            'features': feature_cols,
            'k_experiments': {str(k): float(v) for k, v in k_results.items()}
        }, f, indent=2)

    print("\nTrained model and preprocessing artifacts successfully exported to:")
    print("  - backend/loan_knn_model.joblib")
    print("  - backend/model_summary.json")
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
