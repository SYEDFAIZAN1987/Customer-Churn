# Customer Churn Analysis

![Customer Churn](https://github.com/SYEDFAIZAN1987/Customer-Churn/blob/main/Customer-Churn.png)

## 📖 About the Project

Customer churn prediction is a critical task in business analytics aimed at identifying customers likely to discontinue a service or product. This project leverages machine learning models to predict customer churn using the `Churn Modelling.csv` dataset, encompassing demographic, financial, and behavioral features. The project is divided into six key parts:

1. **Data Cleansing**: Preprocessing, missing value handling, encoding categorical variables, and handling outliers.
2. **KNN Model**: Using the k-Nearest Neighbors algorithm for churn prediction.
3. **Decision Tree, Random Forest, and Gradient Boosting Models**: Comparing the performance of these models.
4. **Support Vector Machine (SVM)**: Using different kernels to improve prediction.
5. **Neural Networks**: Implementing artificial neural networks for classification.
6. **Model Comparison**: Evaluating and comparing all models for the best performance.

---

## 🚀 Key Highlights

### **Data Preprocessing**
- Handled missing values and irrelevant columns.
- Encoded categorical variables using one-hot encoding.
- Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.

### **Machine Learning Models**
- **k-Nearest Neighbors (KNN):**
  - Achieved optimal results with `k = 9` using cross-validation.
- **Decision Tree, Random Forest, and Gradient Boosting:**
  - Hyperparameter tuning with `GridSearchCV` for maximum efficiency.
- **Support Vector Machines (SVM):**
  - Explored linear, RBF, polynomial, and sigmoid kernels.
- **Neural Networks:**
  - Compared solvers for weight optimization and activation functions.

### **Model Comparison**
- Conducted a detailed comparison using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC Curve and AUC values

---

## 📊 Tools & Technologies
- **Programming Languages**: Python
- **Libraries**:
  - **Preprocessing**: pandas, numpy
  - **Visualization**: matplotlib, seaborn
  - **Machine Learning**: scikit-learn, imbalanced-learn
- **Techniques**: Hyperparameter tuning, SMOTE, cross-validation, feature importance analysis

---

## 📈 Key Results
- **Best Performing Model**: Random Forest with an AUC of 0.86 and strong overall predictive accuracy.
- **Feature Importance**:
  - Age and NumOfProducts were consistently the most important predictors across models.
- **Insights**:
  - Addressed challenges like class imbalance and overfitting with techniques such as SMOTE and pruning.

---

## 📂 Project Structure
 ├── Data/ │ ├── Churn_Modelling.csv ├── Analysis/ │ ├── scripts/ │ │ ├── data_cleansing.py │ │ ├── knn_model.py │ │ ├── decision_tree_model.py │ │ ├── svm_model.py │ │ ├── neural_network_model.py ├── Visualizations/ │ ├── roc_curves/ ├── Reports/ │ ├── Customer_Churn_Analysis.pdf ├── README.md
 
---

## 📜 Detailed Report

For a comprehensive understanding of the project, including detailed methodologies, visualizations, and results, refer to the full report: [Customer Churn Analysis Report](https://github.com/SYEDFAIZAN1987/Customer-Churn/blob/main/Customer%20Churn%20Analysis.pdf).

---

## 🤝 Contributions & Feedback
If you'd like to contribute, suggest improvements, or have any questions, feel free to open an issue or reach out!

---

**Author**: Syed Faizan  
**Master’s Student in Data Analytics and Machine Learning**  
