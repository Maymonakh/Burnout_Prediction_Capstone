# 🧠 Predicting Remote Work Burnout & Social Isolation
## Using Machine Learning to Flag At-Risk Employees Early

---

## 🚀 Live Demo

Try the interactive burnout prediction app:

👉 [Open Streamlit App](https://maymonakh-burnout-prediction-capstone-streamlitapp-cgoyy0.streamlit.app/)

<img width="1361" height="603" alt="image" src="https://github.com/user-attachments/assets/849452d9-cd53-48be-acc9-9e6851ebb202" />

---

## 👥 Team Members

| Name | Role & Responsibilities |
|------|------------------------|
| Eman Sarhan | Project Manager — coordinates tasks, tracks timeline, ensures deliverables are on time |
| Yasmeen Nofal | Data Cleaning — handles missing values, encodes categories, normalizes features |
| Joud Hijaz | EDA — creates visualizations, detects outliers, analyzes feature correlations |
| Nada Tabib | ML Modeling — trains and tunes Logistic Regression, Decision Tree, Random Forest |
| Maymona Khanfar | Evaluation & Insights — evaluates models with metrics, interprets results, writes conclusions |

---

## 📌 Problem and Goal

Remote work has been steadily increasing in popularity. By 2026, hybrid and fully remote arrangements have become the new norm across industries worldwide, fundamentally changing how employees work and interact. However, despite this shift, most organizations still lack effective tools to detect burnout and social isolation at an early stage.

Remote work has introduced new and often unnoticed challenges, including longer working hours, reduced human interaction, and unclear boundaries between work and personal life. These factors can gradually lead to burnout — a state of ongoing stress causing emotional and physical exhaustion — as well as social isolation, where employees feel increasingly disconnected from their teams and workplace. One of the main difficulties is that these issues develop silently over time and are not easy to detect early. In many cases, by the time managers notice them, the negative impact on productivity, mental health, and employee retention has already occurred.

In this project, we aim to solve this problem by building a complete machine learning pipeline that predicts burnout risk for remote employees. The model classifies employees into three categories: low, medium, and high risk. It relies on several behavioral and work-related features such as daily working hours, fatigue score, sleep duration, isolation level, screen time, and task completion rate.

To achieve this, we train and compare three classification models: Decision Tree, Random Forest, and Logistic Regression. Our goal is to identify the most important factors influencing burnout risk and provide HR teams and managers with a reliable early warning tool that helps them detect at-risk employees before burnout escalates into a serious organizational problem.

---

## ❓ Why does it matter?

Burnout drives employee turnover, reduces productivity, and negatively impacts mental health. Beyond individual employees, these effects create measurable organizational costs, from increased absenteeism to higher recruitment expenses when experienced staff leave. Early detection allows organizations to take preventive action before the situation escalates, turning data-driven insights into timely, targeted responses.

- **HR Departments** can design targeted well-being programs, monitor at-risk groups based on fatigue scores and isolation index, and intervene before burnout escalates.
- **Team Managers** can adjust workloads by tracking work hours, after-hours work, and task completion rates among their remote teams.
- **Company Leadership** can shape remote work policies backed by data, using burnout risk predictions to evaluate the effectiveness of existing well-being initiatives.

---

## 📊 Data

The Remote Work Burnout and Social Isolation dataset is an open dataset focused on employee well-being in remote work environments. It contains 2,000 records covering key behavioral and work-related indicators such as daily working hours, screen time, virtual meetings, sleep duration, social isolation, task completion, and fatigue levels. This dataset provides a solid foundation for building a burnout risk classifier. To view the dataset click the link below:

[View Dataset](https://www.kaggle.com/datasets/aryanmdev/remote-work-burnout-and-social-isolation-2026/data)

### What one row represents:
A single remote employee's recorded work metrics and self-reported well-being indicators.

### Columns

| Column | Description |
|--------|------------|
| day_type | Type of day (Weekday or Weekend) |
| work_hours | Total hours worked that day |
| screen_time_hours | Hours spent in front of screens |
| meetings_count | Number of meetings attended that day |
| breaks_taken | Number of breaks taken during the day |
| after_hours_work | Whether the employee worked after hours (0 = No, 1 = Yes) |
| app_switches | Number of times the user switched between applications |
| sleep_hours | Hours of sleep the previous night |
| task_completion | Proportion of tasks completed that day |
| isolation_index | Score measuring degree of social isolation |
| fatigue_score | Self-reported fatigue level score |
| burnout_score | Composite burnout score |
| burnout_risk | 🎯 Target variable — Low / Medium / High |

The `burnout_score` is a composite metric that directly reflects the burnout level. Therefore, it was excluded from the modeling process to prevent data leakage.

---

## 📊 Sample Data Preview
The following table shows a sample of the dataset used in this project:

| user_id | day_type | work_hours | screen_time_hours | meetings_count | breaks_taken | after_hours_work | app_switches | sleep_hours | task_completion | isolation_index | fatigue_score | burnout_score | burnout_risk |
|--------|----------|-----------|-------------------|----------------|--------------|------------------|--------------|-------------|-----------------|-----------------|---------------|---------------|--------------|
| 129 | Weekday | 7.74 | 5.99 | 3 | 7 | 0 | 50 | 7.5 | 74.8 | 4 | 3.51 | 17.23 | Low |
| 24  | Weekday | 8.48 | 7.23 | 3 | 5 | 0 | 63 | 7.31 | 86.36 | 4 | 5.15 | 24.01 | Low |
| 147 | Weekend | 8.39 | 7.14 | 1 | 5 | 0 | 27 | 6.73 | 89.23 | 8 | 7.85 | 51.46 | Medium |
| 42  | Weekday | 12.48 | 12.23 | 4 | 1 | 1 | 89 | 6.55 | 64.47 | 5 | 10 | 55.5 | Medium |
| 99  | Weekend | 8.85 | 7.35 | 4 | 6 | 0 | 40 | 8.03 | 98.62 | 3 | 3.45 | 11.95 | Low |


---

## ⚠️ Limitations and Missing Data Notes

- Synthetically generated dataset — may not fully capture real-world employee behavior complexity.
- Limited feature set — cannot account for all psychological, cultural, or organizational factors contributing to burnout.
- No missing data — the dataset contained no missing values, so no imputation was required.
- Class imbalance — the dataset contains:
  - 1,019 low-risk  
  - 843 medium-risk  
  - 138 high-risk  

This was monitored and addressed using class weighting during modeling.


---

## 🧹 Data Cleaning

- Handle Missing Values:
```python
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")
```
Output:

<img width="215" height="294" alt="image" src="https://github.com/user-attachments/assets/7d9b9225-824d-4b4e-ab32-26a3fd9c29ab" />

- Check for Duplicate Rows:
```python
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed. New shape: {df.shape}")
else:
    print("No duplicates found. Dataset is clean.")
```
Output:

<img width="341" height="44" alt="image" src="https://github.com/user-attachments/assets/79628f47-29db-4b32-9417-358a75cfdede" />

---

## 🔍 Exploratory Data Analysis (EDA)

For a full step-by-step analysis, detailed code, and interactive visualizations, please refer to the Google Colab notebook:

👉 [View Full EDA on Google Colab](https://colab.research.google.com/drive/1YpwI6m2I0H21bEem1xc-veKaBcs6vhWT?usp=sharing)

EDA helps us understand data distributions, feature relationships, and patterns relevant to predicting burnout risk.

### 📊 Key Visualization

- **Burnout Risk Distribution**

Understanding class distribution is important, as imbalance can impact model performance and evaluation.

<img width="684" height="384" alt="image" src="https://github.com/user-attachments/assets/1fb50ce8-5820-4ce0-9ed7-d001cd3d7cc9" />

---

### 📊 Key Insights

- `burnout_score` perfectly separates risk categories → identified as a leaky feature and excluded.
- Higher burnout risk is associated with:
  - Lower `sleep_hours`
  - Higher `work_hours` and `screen_time_hours`
- `fatigue_score` and `isolation_index` strongly differentiate between risk groups.
- `task_completion` decreases as burnout risk increases.
- Most features show low-to-moderate correlation → each contributes unique information.

📎 For detailed visualizations and full analysis, refer to the Colab notebook above.

---

## 🧪 Prepare Data for Modeling

Before training the machine learning models, the following preprocessing steps were performed:

- **Encoding categorical variables**: Categorical features were converted into numerical format using `LabelEncoder`.
- **Removing data leakage features**: The `burnout_score` column was excluded because it directly encodes the target variable.
- **Feature-target split**: The dataset was divided into features (`X`) and target (`y`), where `burnout_risk` is the target variable.
- **Feature scaling**: Numerical features were standardized using `StandardScaler` to ensure all features are on the same scale, which is especially important for Logistic Regression to converge properly.
- **Train-test split**: The data was split into 80% training and 20% testing sets, using stratification to maintain class distribution consistency.

## 🌳 Baseline Model: Decision Tree

We started with a Decision Tree Classifier as a baseline model. It is simple, interpretable, and does not require feature scaling.

This model provides a performance benchmark (baseline) to compare against more advanced models such as Logistic Regression and Random Forest.

<img width="468" height="202" alt="image" src="https://github.com/user-attachments/assets/e3ed0d87-17ed-49bf-a850-3512ced4f547" />

Decision Tree – Confusion Matrix :

<img width="563" height="384" alt="image" src="https://github.com/user-attachments/assets/2ae111cd-8ae2-495f-8f8b-918cd3fea91d" />

## 🌲 Improved Model: Random Forest

Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs through averaging (for classification, majority voting). This approach helps reduce overfitting and generally improves both accuracy and model stability compared to a single Decision Tree.

In this project, we used a Random Forest model with the following configuration:

- **Number of trees**: 200 estimators
- **Random state**: 42 (for reproducibility)

This setup ensures more robust predictions by leveraging multiple trees trained on different subsets of the data, leading to better generalization on unseen samples.

<img width="475" height="201" alt="image" src="https://github.com/user-attachments/assets/2a21108d-f2da-40a2-a234-a02c1ba758bd" />

Random Forest – Confusion Matrix

<img width="563" height="384" alt="image" src="https://github.com/user-attachments/assets/90516938-2fbc-4595-ad6d-7a965c9472cc" />

### 🌟 Feature Importance (Random Forest)

The Random Forest model provides an inherent measure of feature importance, which indicates how much each variable contributes to the prediction process.

Features with higher importance scores have a stronger influence on predicting burnout risk, while lower-scoring features contribute less to the model’s decision-making.

This helps us understand which behavioral and work-related factors are most influential in identifying employees at risk of burnout.

<img width="884" height="484" alt="image" src="https://github.com/user-attachments/assets/892db436-468a-4683-8fc9-b4ca0a2c60e6" />

## 📈 Improved Model: Logistic Regression

Logistic Regression is a strong and highly interpretable linear classification model. It is used as one of the main models in this project due to its simplicity and ability to provide clear decision boundaries.

Since Logistic Regression is sensitive to feature magnitudes, the input features were previously standardized using `StandardScaler` to ensure proper model convergence.

In this implementation, we increased `max_iter` to guarantee full convergence during training and improve model stability.

<img width="422" height="189" alt="image" src="https://github.com/user-attachments/assets/0431057a-48b0-4df0-8331-fff7ac1f8c02" />

Logistic Regression – Confusion Matrix

<img width="563" height="384" alt="image" src="https://github.com/user-attachments/assets/34012354-cfbc-4e25-8832-9e016636912c" />

## 📊 Model Comparison and Final Results

We compared all three models using the same test set and evaluated them with accuracy as the main metric. This allows a fair and consistent comparison of model performance.

<img width="684" height="384" alt="image" src="https://github.com/user-attachments/assets/b3dffc10-d38d-4ebb-8174-03fe7ef143dc" />

## 📌 16. Conclusions and Recommendations

### Summary of Findings

We developed and evaluated three classification models to predict burnout risk among remote employees. The results show clear differences in performance across models:

| Model | Accuracy | Observation |
|------|----------|-------------|
| Decision Tree | ~93% | Strong baseline, but tends to overfit |
| Random Forest | ~97% | Significant improvement, more robust due to ensemble learning |
| Logistic Regression | ~97.5% | Best overall performance with strong interpretability and stability |

---

### 🌟 Most Important Predictors

Based on the Random Forest feature importance analysis, the most influential factors in predicting burnout risk are:

- `fatigue_score` — the most powerful indicator of burnout risk
- `isolation_index` — highlights the impact of social disconnection
- `sleep_hours` — lower sleep is strongly associated with higher risk
- `work_hours` and `screen_time_hours` — indicators of overwork and prolonged exposure

---

### 💡 Recommendations for Stakeholders (HR / Management)

- Regularly monitor **fatigue levels** and **isolation index** as early warning indicators.
- Encourage healthy work habits, especially maintaining sufficient **sleep** and taking regular breaks.
- Pay attention to employees with consistently high **work hours** or **screen time**.
- Use this model as a **supportive decision-making tool** for early intervention, rather than a standalone or definitive assessment.

---

## 🚀 How to Run

### 📌 Option A — Google Colab (Recommended)

1. Open the **Google Colab Notebook Link**
2. Upload `wfh_burnout_dataset.csv` when prompted by the first cell
3. Run all cells from top to bottom to reproduce the results

---

### 💻 Option B — Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/Maymonakh/Burnout_Prediction_Capstone.git
cd burnout-prediction

# 2. Install required packages
pip install -r requirements.txt

# 3. Place dataset in the data folder
# data/wfh_burnout_dataset.csv

# 4. Launch Jupyter Notebook
jupyter notebook
```
## 📦 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## ⚠️ Limitations

- The dataset is synthetically generated, so real-world performance may differ.
- The `burnout_score` feature was excluded due to data leakage; without it, the prediction task becomes more challenging but more realistic.
- The model does not account for psychological, personal, or external life factors that may influence burnout.

---

## 🚀 Possible Next Steps

- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Experiment with advanced models such as XGBoost or Support Vector Machines (SVM).
- Validate the model using real-world employee data.
