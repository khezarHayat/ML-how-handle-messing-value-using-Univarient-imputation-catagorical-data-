# Handling Missing Values Using Univariate Imputation (catagorical data)

This repository contains my practical work on **handling missing values using Univariate Imputation techniques in Machine Learning**.

Missing values are one of the most common problems found in real-world datasets. Before training a Machine Learning model, missing values need to be handled properly. One of the simplest approaches is **Univariate Imputation**, where missing values in a feature are estimated using information from that same feature.

## What is Univariate Imputation?

Univariate imputation means filling missing values by looking at **only one feature at a time**.

For example, if an `Age` column contains missing values, we can calculate the mean or median of the available `Age` values and use it to fill the missing values.

Example:

```text
Age
20
25
NaN
30
35
```

Using the median:

```text
Age
20
25
25
30
35
```

The missing value is estimated using information from the `Age` column itself.

---

# Types of Univariate Imputation

In this repository, I practiced several common techniques.

## 1. Mean Imputation

Mean imputation replaces missing numerical values with the **mean (average)** of the available values.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X_imputed = imputer.fit_transform(X)
```

Mean imputation can be useful when the numerical feature does not contain significant outliers.

---

## 2. Median Imputation

Median imputation replaces missing values with the **median** of the feature.

```python
imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)
```

Median imputation is often useful when a feature is skewed or contains outliers because the median is less affected by extreme values than the mean.

---

## 3. Mode Imputation

Mode imputation replaces missing values with the **most frequently occurring value**.

It is commonly used for categorical features.

Example:

```text
City
Peshawar
Lahore
Peshawar
NaN
Peshawar
```

The mode is `Peshawar`, so the missing value can be replaced with `Peshawar`.

Using Scikit-learn:

```python
imputer = SimpleImputer(strategy="most_frequent")

X_imputed = imputer.fit_transform(X)
```

---

## 4. Constant Imputation

Constant imputation replaces missing values with a value that we specify.

For example:

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value=0
)

X_imputed = imputer.fit_transform(X)
```

For categorical data, we can use a value such as:

```python
fill_value="Missing"
```

This makes the missing values an explicit category.

---

## 5. Random Sample Imputation

Random sample imputation replaces a missing value with a randomly selected value from the **same feature's existing values**.

For example:

```text
Age
20
25
NaN
30
35
```

A missing value could be replaced by one of the existing values such as `20`, `25`, `30`, or `35`.

This approach can help preserve the original distribution of the feature.

---

# Univariate vs Multivariate Imputation

The main difference is the information used to estimate the missing value.

### Univariate Imputation

```text
Feature A
   ↓
Calculate value
   ↓
Fill missing value
```

It uses information from the **same feature**.

### Multivariate Imputation

```text
Feature A ─┐
Feature B ─┼──> Imputation Model
Feature C ─┘
                 ↓
          Missing Value
```

It uses information from **multiple features**.

---

# Important Practice

The imputer should generally be fitted on the **training data only**.

```python
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

This prevents information from the test dataset from influencing the imputation process.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

# Learning Goal

The goal of this repository is to build a strong practical understanding of **Univariate Imputation** and learn how different techniques can be used to handle missing values before training Machine Learning models.

Through this practice, I learned how to choose between mean, median, mode, constant, and other univariate approaches depending on the type and distribution of the data.

This repository is part of my ongoing **Machine Learning and AI learning journey**.
