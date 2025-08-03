# 🏡 House Price Prediction Pipeline using Random Forest & Scikit-learn 🚀

This project demonstrates a complete **end-to-end Machine Learning pipeline** to predict house prices based on housing data. It covers data preprocessing, stratified sampling, model training using Random Forest, saving/loading the model, and performing inference on new input data.

---

## 📌 Project Objective
- Build a machine learning model to predict **median house values**.
- Perform **Stratified Sampling** based on income categories.
- Use a **Pipeline** to preprocess numerical and categorical attributes.
- Save the trained model and pipeline for later inference.
- Automate inference on unseen data and save predictions to a CSV file.

---

## 🗂️ Project Structure

---

## 🛠️ Technologies & Libraries Used
- Python 🐍
- Jupyter Notebook 📓
- Pandas
- NumPy
- Scikit-learn (Pipeline, RandomForestRegressor, StratifiedShuffleSplit)
- Joblib (for saving/loading models)
- Matplotlib & Seaborn (for EDA & Visualization)

---

## 🧱 Workflow Breakdown
### 1. Data Loading & Stratified Sampling
- Load `housing.csv` dataset.
- Create **income categories** for stratified sampling.
- Perform **Stratified Shuffle Split** to ensure test data is representative.

### 2. Preprocessing Pipeline
- Separate **numerical attributes** and **categorical attributes**.
- Build a **data preprocessing pipeline**:
  - Numerical: Imputer, StandardScaler.
  - Categorical: OneHotEncoder.

### 3. Model Training
- Train a **Random Forest Regressor** on the preprocessed training data.
- Save the trained model and preprocessing pipeline using **Joblib**.

### 4. Inference Phase
- Load the saved model and pipeline.
- Read new input data (`input.csv`).
- Transform input data using the pipeline.
- Predict house prices.
- Save the results into `output.csv`.

---

## ⚙️ How to Run This Project
1. Clone the repository or download the files.
2. Ensure you have Python and required libraries installed:
    ```bash
    pip install pandas numpy scikit-learn joblib matplotlib seaborn
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook final_project.ipynb
    ```
4. Run all the cells to:
    - Train the model (if not already trained).
    - Generate predictions on the test input (`input.csv`).
5. Check the **output.csv** file for prediction results.

---

## 📝 Important Files Explanation
- **housing.csv** – Original dataset with features and target (`median_house_value`).
- **input.csv** – Test data after stratified split, used for inference.
- **output.csv** – The predicted house values after running inference.
- **model.pkl** – Saved Random Forest model.
- **pipeline.pkl** – Saved preprocessing pipeline for transforming input data.
  
---

## ✨ Features
- Fully automated ML pipeline (Train & Inference modes).
- Efficient handling of numerical & categorical data using Scikit-learn Pipelines.
- Stratified sampling ensures accurate evaluation.
- Model & pipeline serialization using Joblib for fast inference.
  
---

## 📈 Potential Improvements
- Hyperparameter tuning using GridSearchCV.
- Deploy the model using a Flask/Django web app.
- Add interactive visualizations for feature importance.
- Extend the pipeline with more advanced preprocessing techniques.

---

## 📧 Contact
- **Name**: Himanshu Balodi
- **Email**: himanshu.balodi.ds@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/himanshu-balodi2001/

---

## 📚 References
- Dataset: California Housing Dataset
- Scikit-learn Documentation
- Inspired by “Hands-On Machine Learning” by Aurélien Géron
