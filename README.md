# Loan Approval Prediction System

An end-to-end Machine Learning and Full-Stack Web Application for evaluating and predicting loan approval outcomes in real time using a tuned K-Nearest Neighbors (KNN) classifier.

---

## 📌 Project Architecture

```
Class 13/
├── train.csv                     # Loan Prediction Dataset (614 rows x 13 columns)
├── Loan_Prediction_Problem.csv    # Original source dataset
├── train_knn.py                  # End-to-end ML training, tuning & evaluation pipeline
├── confusion_matrices.png        # Confusion Matrix visual comparison (Default vs Tuned)
├── k_value_accuracy.png          # Test accuracy across different values of k
├── loan_knn_model.joblib         # Serialized best model, StandardScaler, and metadata
├── start_backend.py              # Launch script for FastAPI server
│
├── backend/
│   ├── app.py                   # FastAPI REST API with /predict, /model-info, /health
│   ├── loan_knn_model.joblib    # Deployed model artifacts & encoders
│   └── model_summary.json       # JSON export of evaluation metrics
│
└── frontend/                    # Next.js 16 (React 19 + TypeScript + Vanilla CSS)
    ├── app/
    │   ├── layout.tsx           # SEO metadata, fonts, and responsive layout
    │   ├── page.tsx             # Interactive application UI with live predictions
    │   └── globals.css          # Rich dark-mode financial dashboard design system
    ├── package.json
    └── tsconfig.json
```

---

## 📊 Machine Learning Pipeline & Results

### 1. Data Preprocessing & Missing Values
- **Imputation**:
  - **Numerical columns** (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`): Imputed with **Median**.
  - **Categorical columns** (`Gender`, `Married`, `Dependents`, `Self_Employed`): Imputed with **Mode**.
- **Categorical Encoding**:
  - `Gender`: Female = 0, Male = 1
  - `Married`: No = 0, Yes = 1
  - `Dependents`: 0 = 0, 1 = 1, 2 = 2, 3+ = 3
  - `Education`: Not Graduate = 0, Graduate = 1
  - `Self_Employed`: No = 0, Yes = 1
  - `Property_Area`: Rural = 0, Semiurban = 1, Urban = 2
  - `Loan_Status` (Target): N = 0, Y = 1
- **Feature Scaling**:
  - Applied `StandardScaler` to ensure distance metrics in KNN are not biased by disparate feature magnitudes (e.g. Applicant Income in thousands vs Dependents in units).
- **Split**:
  - 80% Training (491 samples), 20% Testing (123 samples), stratified by target class.

---

### 2. K-Nearest Neighbors (KNN) Performance

#### A. Experiments with Different Values of $k$:
| $k$ Value | Test Accuracy |
| :---: | :---: |
| $k = 3$ | 82.93% |
| $k = 5$ (Default) | 84.55% |
| $k = 7$ | 84.55% |
| $k = 9$ | 85.37% |
| $k = 11$ | 84.55% |
| $k = 13$ | 84.55% |
| $k = 15$ | 85.37% |

#### B. Hyperparameter Tuning (`GridSearchCV`):
- Explored:
  - `n_neighbors`: 1 to 30
  - `weights`: `['uniform', 'distance']`
  - `metric`: `['euclidean', 'manhattan']`
- **Best Hyperparameters**:
  - `n_neighbors`: **16**
  - `metric`: **'euclidean'**
  - `weights`: **'uniform'**
  - **Best Test Accuracy**: **85.37%**

#### C. Confusion Matrix & Metrics Comparison:

| Metric | Default KNN ($k=5$) | Tuned KNN ($k=16$) |
| :--- | :---: | :---: |
| **Accuracy** | **84.55%** | **85.37%** |
| **Confusion Matrix** | `[[22, 16], [3, 82]]` | `[[21, 17], [1, 84]]` |
| **Approved Class Recall** | 96% | **99% (84 / 85 approved loans correctly classified)** |
| **Rejected Class Precision** | 88% | **95%** |

---

## 🚀 How to Run the Project

### 1. Retrain or Evaluate the Model (Optional)
```bash
python train_knn.py
```
This updates the models, outputs accuracy reports, saves `confusion_matrices.png` and `k_value_accuracy.png`, and writes artifacts into `backend/`.

### 2. Start the FastAPI Backend
```bash
python start_backend.py
```
The REST API will be available at `http://127.0.0.1:8000`:
- `GET http://127.0.0.1:8000/health` (Health Check)
- `GET http://127.0.0.1:8000/model-info` (Model Architecture & Parameters)
- `POST http://127.0.0.1:8000/predict` (Real-Time Loan Prediction)
- `GET http://127.0.0.1:8000/docs` (Interactive Swagger Documentation)

### 3. Start the Next.js Frontend
In a new terminal:
```bash
cd frontend
npm run dev
```
Open **`http://localhost:3000`** in your browser.

---

## 🎨 Next.js Web App Highlights
- **Preset Scenarios**: Quick one-click scenario testing (Prime Graduate, Rural Buyer, High-Risk Profile, Self-Employed).
- **Interactive Underwriting Dashboard**: Live monthly income calculations, debt-to-income ratio estimator, and amortization estimator.
- **Dynamic Decision Display**: Instant **APPROVED** (Emerald glow) or **REJECTED** (Rose glow) feedback with probability confidence meter.
- **Model Switcher**: Allows toggling between Tuned KNN ($k=16$) and Default KNN ($k=5$) dynamically to compare decisions.
