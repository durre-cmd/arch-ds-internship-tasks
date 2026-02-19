# ==========================================
# Task 4: Movie Rating Prediction (Advanced)
# Collaborative Filtering using SVD
# ==========================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# -----------------------------
# 1. Create Structured Dataset
# -----------------------------

n_users = 50
n_movies = 30

# Create latent factors (hidden preferences)
user_preferences = np.random.rand(n_users, 5)
movie_features = np.random.rand(n_movies, 5)

# Generate ratings matrix (users x movies)
ratings_matrix = np.dot(user_preferences, movie_features.T)

# Scale ratings to 1–5
ratings_matrix = 1 + 4 * (ratings_matrix - ratings_matrix.min()) / (ratings_matrix.max() - ratings_matrix.min())

ratings_df = pd.DataFrame(ratings_matrix)

# Convert matrix into long format
ratings_long = ratings_df.stack().reset_index()
ratings_long.columns = ["UserID", "MovieID", "Rating"]

# -----------------------------
# 2. Train-Test Split
# -----------------------------

train_data, test_data = train_test_split(
    ratings_long, test_size=0.2, random_state=42
)

# Create pivot table from training data
train_matrix = train_data.pivot(index="UserID", columns="MovieID", values="Rating").fillna(0)

# -----------------------------
# 3. Matrix Factorization (SVD)
# -----------------------------

U, sigma, Vt = np.linalg.svd(train_matrix, full_matrices=False)

# Keep top k latent factors
k = 5
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# Reconstruct predicted ratings
predicted_matrix = np.dot(np.dot(U_k, sigma_k), Vt_k)

predicted_df = pd.DataFrame(
    predicted_matrix,
    index=train_matrix.index,
    columns=train_matrix.columns
)

# -----------------------------
# 4. Evaluate Model
# -----------------------------

predictions = []
actuals = []

for row in test_data.itertuples():
    user = row.UserID
    movie = row.MovieID
    
    if user in predicted_df.index and movie in predicted_df.columns:
        predictions.append(predicted_df.loc[user, movie])
        actuals.append(row.Rating)

rmse = np.sqrt(mean_squared_error(actuals, predictions))

print("Model Performance:")
print("RMSE:", rmse)

# -----------------------------
# 5. Predict Unseen Rating
# -----------------------------

sample_user = 10
sample_movie = 5

predicted_rating = predicted_df.loc[sample_user, sample_movie]

print("\nPredicted Rating for User", sample_user,
      "on Movie", sample_movie, ":", round(predicted_rating, 2))

print("\nProgram Finished Successfully")
