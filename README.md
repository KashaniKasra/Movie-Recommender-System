# Movie Recommendation System (Trust-Aware)

This project implements a trust-aware movie recommender system using traditional machine learning techniques (non-deep learning) to predict user ratings for unseen movies. The goal is to combine user-movie interaction data with trust relationships between users for more accurate predictions.

---

## Objective

Predict ratings in the test set for given (user_id, item_id) pairs based on:
- Historical user-movie ratings
- Trust relationships between users

---

## Dataset Description

### `train_data_movie_rate.csv`
- `user_id`: Unique identifier for the user
- `item_id`: Unique identifier for the movie
- `label`: Rating given by the user to the movie

### `train_data_movie_trust.csv`
- `user_id_trustor`: The user who trusts another
- `user_id_trustee`: The user being trusted
- `trust_value`: Trust score between users

### `test_data.csv`
- Contains `user_id` and `item_id` pairs for which predicted ratings must be generated

---

## Implementation Pipeline

### 1. Data Preprocessing
- Merged trust data with user ratings to introduce social awareness
- Handled missing values and normalized ratings (if needed)
- Converted user and movie IDs into encoded indices for modeling

### 2. Feature Engineering
- Constructed collaborative filtering features
- Integrated trust propagation features between users

### 3. Model Development
- Trained regression models (e.g., Ridge, XGBoost, kNN) to predict ratings
- Cross-validation to tune parameters
- Final model selected based on RMSE and MAE

### 4. Prediction and Submission
- Loaded `test_data.csv` and predicted ratings for each user-movie pair
- Submission format: `submission.csv` with `id`, `label` (predicted rating)

---

## Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared Score (R²)

---

## Tools & Libraries
- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Jupyter Notebook

---

## Notes
- No deep learning models were used
- Trust-based recommendation enhanced cold-start performance for sparse users