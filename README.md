# 🩺 Heart Disease Prediction – Machine Learning Full Pipeline

This project implements a complete **Machine Learning pipeline** on the [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease).  
The goal is to analyze, predict, and visualize heart disease risks using **supervised and unsupervised learning models**, with a bonus **Streamlit UI** deployed via **Ngrok**.

---

## 📌 Project Overview
The pipeline includes:
- Data preprocessing & cleaning
- Dimensionality reduction (PCA)
- Feature selection (RFE, Chi-Square, Feature Importance)
- Supervised learning models: Logistic Regression, Decision Tree, Random Forest, SVM
- Unsupervised learning models: K-Means, Hierarchical Clustering
- Model evaluation and hyperparameter tuning
- Model export for deployment
- Streamlit-based UI for user interaction
- Deployment with Ngrok (bonus)

---

## 🗂️ Project Structure
Heart_Disease_Project/
│── data/
│ ├── heart_disease.csv
| ├── heart_disease_cleaned.csv
| ├── heart_disease_pca.csv
| ├── heart_disease_reduced.csv
| ├── heart_disease_test.csv
| ├── heart_disease_train.csv
│
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│
│── models/
│ ├── final_model.pkl
│
│── ui/
│ ├── app.py # Streamlit UI
│
│── deployment/
│ ├── ngrok_setup.txt
│
│── results/
│ ├── evaluation_metrics.txt
│ ├── best_model_details.txt
│ ├── cluster_visualization.png
│ ├── cumulative_variance_plot.png
│ ├── elbow_method.png
│ ├── dendrogram.png
│ ├── feature_importance_plot.png
│ ├── pca_scatter_plot.png
│ ├── roc_curves.png
│
│── README.md
│── requirements.txt
│── .gitignore

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yasmin798/Heart_Disease_Project.git
   cd Heart_Disease_Project
2. Create a virtual environment and install dependencies:
   pip install -r requirements.txt
3. Run Jupyter notebooks (optional, for analysis):
   jupyter notebook

## ▶️ Running the Streamlit App
1. Navigate to the project root and run:
    streamlit run ui/app.py 
2. Open http://localhost:8501 in your browser.

## 🌍 Deployment with Ngrok
1.In a new terminal, start Ngrok:
   ngrok http 8501
2.Copy the generated public link (e.g., [https://abc123.ngrok-free.app](https://be8f27523a21.ngrok-free.app/)).
3.Share the link for live access to the app.
📌 Note: Ngrok free links are temporary and remain active only while the app is running locally.

## 📊 Results
1.Cleaned dataset with selected features
2.PCA results and variance plots
3.Model performance metrics (accuracy, precision, recall, F1, ROC/AUC)
4.Optimized final model exported as .pkl
5.Streamlit UI for real-time prediction
6.Live app deployment via Ngrok



