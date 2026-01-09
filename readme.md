# 🏭 ML_PLC_Industry  
**Industrial Machine Learning with PLC–MATLAB OPC UA Communication**

## 📌 Project Overview
This project demonstrates how **industrial process data** can be used to train a **machine learning model** and how such models can be connected to **PLC systems** using **OPC UA**.

The main goal is to **predict product quality** in an industrial mining process and show how ML can support **automation and process optimization**.

---

## 🎯 Project Goals
- Communicate with PLC systems using **OPC UA (MATLAB side)**
- Train an **industrial ML model in Python**
- Predict **Silica percentage (Silica %)** in a flotation process
- Use real industrial data from **Kaggle**
- Demonstrate ML usage for **Industry 4.0**

---

## 📊 Dataset
**Source:** Kaggle  
**Dataset:** Quality Prediction in a Mining Process  
🔗 https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process/data

### Dataset Description
- Real data from an **iron ore flotation plant**
- Time-series industrial sensor data
- The target variable is **Silica % in iron concentrate**
- Large dataset (millions of rows)

### Used Features (Inputs)
Selected process parameters:
- % Iron Feed  
- % Silica Feed  
- Starch Flow  
- Amina Flow  
- Ore Pulp pH  
- Ore Pulp Density  
- Flotation Columns Air Flow  

### Target (Output)
- **% Silica Concentrate**

---

## ⚙️ Data Preparation
Steps used in the notebook:
- Load CSV data
- Select relevant industrial features
- Remove unnecessary columns
- Random sampling (100,000 rows) to speed up training
- Train/Test split
- Basic data visualization and inspection

---

## 🤖 Machine Learning Approach
- **Problem type:** Regression
- **Goal:** Predict Silica % from process parameters
- **Language:** Python
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

### Models Used
- Linear Regression  
- Random Forest Regressor  

Random Forest showed better performance for nonlinear industrial data.

---

## 📈 Model Evaluation
Evaluation metrics used:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

The trained model is suitable for **process monitoring** and **decision support**, not for direct closed-loop control.

---

## 🔌 PLC & OPC UA Concept
This project is designed to work together with:
- **PLC systems**
- **OPC UA communication**
- **MATLAB as OPC UA client**

### Concept Flow
```text
PLC Sensors → OPC UA → MATLAB → Python ML Model → Prediction → Operator / SCADA

---

### 📁 Project Structure
ML_PLC_industry/
├── ML_PLC_Jupyter.ipynb   # Main ML notebook
├── README.md              # Project documentation
└── data/                  # Dataset (not included)
