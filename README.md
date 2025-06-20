# 🧠 ML Regression Project with EDA, Pipelines, and Residual Analysis

This project is a complete machine learning pipeline built for a **regression task**. It includes **Exploratory Data Analysis (EDA)**, **data preprocessing** using `sklearn` Pipelines, **model training**, and **evaluation** using industry-standard regression metrics like **MAE**, **RMSE**, and **R² Score**, along with **residual visualization**.

---

## 📂 Project Structure

``` ├── venv/ 
├── app.py
 ```

---

## 🔍 Features

- 📊 **EDA**: Automated analysis of missing values, categorical feature distributions, correlation heatmaps.
- 🛠️ **Preprocessing Pipeline**: Uses `ColumnTransformer` to handle numerical and categorical columns separately.
- 🧪 **Model Training**: Train regression models like `LinearRegression`, `RandomForestRegressor`, etc.
- 📈 **Metrics Evaluation**: MAE, RMSE, R² Score for model accuracy.
- 📉 **Residual Plot**: Visual tool to evaluate model fit.
- 🌐 Optional **Streamlit UI** to make the workflow interactive.

---

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
---
## Streamlit App (Optional)
bash
Copy
Edit
streamlit run app.py
---
