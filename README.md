# Loan Approval Prediction

## Project Overview
This project aims to build a machine learning model to predict whether a loan application will be approved or not based on applicant data. The model leverages multiple classification algorithms, including Random Forest, K-Nearest Neighbors, SVM, and Logistic Regression, and selects Random Forest as the final model due to its strong performance.

---

## Dataset
The dataset contains various features related to the applicant and the loan, such as:
- Gender
- Married status
- Dependents
- Education
- Self_Employed status
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (target variable)

---

## Data Preprocessing
- Dropped the `Loan_ID` column as it is not relevant.
- Used OneHotEncoding for categorical variables: Gender, Married, Education, Self_Employed, Property_Area.
- Filled missing values in numerical columns with the mean value.
- Encoded the target variable `Loan_Status` using LabelEncoder.
- Balanced the dataset using SMOTE to handle class imbalance.
- Standardized numerical features using StandardScaler.

---

## Model Training and Evaluation
- Split the data into training (60%) and testing (40%) sets.
- Trained multiple classifiers: Random Forest, KNN, SVM, Logistic Regression.
- Evaluated models based on accuracy on training and testing data.
- Selected Random Forest Classifier (with 7 estimators and entropy criterion) as the final model.

---

## Saving Artifacts
- Saved the OneHotEncoder, StandardScaler, and the trained Random Forest model using `pickle` for later use.

---

## How to Use the Model
You can load the saved model and preprocessing objects to make predictions with new data inputs. An example function `predict_loan()` is provided which takes applicant details as input and returns whether the loan will be approved or not.

```python
status = predict_loan('Male', 'Yes', 2, 'Graduate', 'No', 4583, 1508, 128, 360, 1, 'Urban')
print("Loan Approved") if status == 1 else print("Loan Not Approved")
