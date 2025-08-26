# 📌 Fraud Detection using Machine Learning  

This project focuses on **predicting fraudulent financial transactions** using machine learning.  
The dataset contains **6.3 million+ transactions** with details such as transaction type, amount, balances, and fraud labels.  

---

## 📊 Dataset  

- Transactions: ~6.3 million  
- Features: 10 columns (step, type, amount, balances, etc.)  
- Target: `isFraud` (1 = fraud, 0 = genuine)  
- Source: Provided in the case study (Accredian)  
- ⚠️ Dataset is not included here due to size/confidentiality.  

---

## 🎯 Objective  

- Build a model to **detect fraud proactively**.  
- Identify **key factors** that signal fraudulent behavior.  
- Suggest **business strategies** to reduce fraud.  

---

## ⚙️ Approach  

1. **Data Cleaning** – removed irrelevant ID columns (`nameOrig`, `nameDest`), checked for missing values (none).  
2. **Feature Engineering** – created `errorBalanceOrg` & `errorBalanceDest` to capture suspicious balance mismatches.  
3. **Encoding** – converted categorical variable `type` into one-hot encoded features.  
4. **Model Training** –  
   - Logistic Regression (baseline)  
   - Random Forest (main model, with `class_weight=balanced`)  
   - XGBoost (optimized model for imbalanced data)  
5. **Evaluation** – Precision, Recall, F1, ROC-AUC.  
6. **Interpretation** – Used feature importance to extract business insights.  

---

## 🏆 Results  

- **Random Forest** achieved ROC-AUC > **0.90**.  
- Fraud is most associated with:  
  - Large `amount`  
  - Transaction type: **TRANSFER** & **CASH_OUT**  
  - Balance mismatches: `errorBalanceOrg`, `errorBalanceDest`  
- PAYMENT and DEBIT transactions are rarely fraudulent.  

---

## 💡 Business Insights  

- Fraudsters mostly use **TRANSFER** or **CASH_OUT** to steal funds.  
- Large sudden amounts are strong fraud signals.  
- Balance mismatches (origin vs destination) are key red flags.  
- **Recommended Prevention:**  
  - Real-time flagging of high-value transfers.  
  - Multi-factor authentication for risky transactions.  
  - Continuous retraining of fraud detection models.  

---

## 🛠 Tech Stack  

- **Python** – pandas, numpy, scikit-learn, matplotlib, seaborn  
- **imbalanced-learn** – SMOTE (for experiments, not final full dataset)  
- **XGBoost**  
- **Jupyter Notebook**  

---

## 🚀 How to Run  

1. Clone this repo:  
   ```bash
   git clone https://github.com/your-username/fraud-detection-ml.git
   cd fraud-detection-ml
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Open Jupyter Notebook and run:  
   ```bash
   jupyter notebook fraud_detection.ipynb
   ```  

---

## 🔮 Future Improvements  

- Try deep learning models (LSTM for transaction sequences).  
- Deploy as API (Flask/Django).  
- Real-time fraud detection using Kafka/Spark.  

---

## 📄 License  

This project is licensed under the **MIT License**.  
