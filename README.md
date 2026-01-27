Heart Disease Prediction – Machine Learning Project

Predict the presence of heart disease based on patient data using machine learning, with an interactive Flask web application for real-time predictions.
🎯 Project Objectives

Build a robust predictive model for early heart disease diagnosis.

Compare performance of multiple machine learning algorithms.

Deploy the model using Flask for interactive, real-time predictions.

🗂 Dataset

Dataset: heart.csv (Cleveland Heart Disease dataset)
Contains 13 features and 1 target:

🧠 Machine Learning Models Implemented

Model	Description
K-Nearest Neighbors (KNN)	Distance-based classification algorithm
Logistic Regression	Linear model for binary classification
Naive Bayes	Probabilistic classifier using Bayes’ theorem
Decision Tree Classifier	Tree-based model splitting data by feature importance
Random Forest Classifier	Ensemble of decision trees for higher accuracy
AdaBoost Classifier	Boosting algorithm combining weak learners
Gradient Boosting Classifier	Iterative ensemble optimizing errors for better performance
XGBoost Classifier	Optimized gradient boosting for high accuracy and speed
Support Vector Machine (SVC)	Margin-based classifier suitable for high-dimensional data
📊 Model Evaluation (Without AUC-ROC)

Evaluation is done using:
Accuracy
Confusion Matrix
Precision, Recall, F1-Score
⚙️ Model Deployment

Flask Web App allows real-time predictions based on user input.
Predictive objects:
heart_model.pkl → trained ML model
scaler.pkl → numeric feature scaling
target_encode.pkl → target encoder for categorical features
📁 Project Structure
Heart-Disease-Prediction/
│
├─ app.py                 # Flask web application
├─ train_model.py         # Script to train and save model
├─ heart.csv              # Dataset
├─ requirements.txt       # Python dependencies
├─ heart_model.pkl        # Saved trained model
├─ scaler.pkl             # Saved scaler
└─ target_encode.pkl      # Saved target encoder

📌 License

📁 Project Structure
Heart-Disease-Prediction/
│
├─ app.py                 # Flask web application
├─ train_model.py         # Script to train and save model
├─ heart.csv              # Dataset
├─ requirements.txt       # Python dependencies
├─ heart_model.pkl        # Saved trained model
├─ scaler.pkl             # Saved scaler
└─ target_encode.pkl      # Saved target encoder

📌 License

MIT License – free to use and modify.
