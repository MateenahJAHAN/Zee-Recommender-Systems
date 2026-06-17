# Zee — Movie Recommender System

A personalized movie **Recommender System** built using **Collaborative Filtering** on the MovieLens-style Zee dataset. The goal is to improve user experience and engagement by recommending movies based on ratings given by a user and by other users similar to them.

**Author:** Mateenah Jahan

Woolf Universilty Malta Europe

Ms Computer Science ML/AI
---

## Problem Statement

Zee (an OTT / streaming platform) wants to move away from generic, one-size-fits-all suggestions and instead show each user personalized movie recommendations. With thousands of movies available, users struggle to discover content they will enjoy, leading to poor engagement and higher churn. Using historical ratings, user demographics and movie metadata, this project predicts what each user is most likely to enjoy and recommends those titles — a classic collaborative-filtering recommender problem.

## Dataset

Three `::`-separated data files (included in this repo):

| File | Format | Description |
|------|--------|-------------|
| `zee-ratings.dat` | UserID::MovieID::Rating::Timestamp | ~1M ratings, 5-star scale, each user has at least 20 ratings |
| `zee-users.dat` | UserID::Gender::Age::Occupation::Zip-code | Demographic info (UserIDs 1–6040) |
| `zee-movies.dat` | MovieID::Title::Genres | Movie titles (with year) and pipe-separated genres |

`Age` and `Occupation` are coded values that are decoded into readable labels during preprocessing.

## Repository Structure

```
Zee-Recommender-Systems/
├── zee.ipynb          # Complete solution notebook
├── zee-ratings.dat    # Ratings data
├── zee-users.dat      # Users data
├── zee-movies.dat     # Movies data
└── README.md
```

## Approach

The notebook follows an end-to-end workflow:

1. **Data loading & merging** — read the three `::`-separated files with proper column names and merge ratings + users + movies into a single consolidated dataframe.
2. **EDA, cleaning & feature engineering** — review structure, check missing values and duplicates, perform type conversions, decode the `Age` and `Occupation` columns into readable labels, and derive new features `ReleaseYear` and `Decade` from the movie title. Data is also grouped by average rating and number of ratings.
3. **Item-based Pearson Correlation** — build a user × movie-title pivot table, then recommend movies whose rating columns correlate most strongly with a chosen movie (with a minimum-ratings filter to avoid unreliable correlations).
4. **Cosine Similarity + KNN** — build user-user and item-item similarity matrices, convert the pivot table to a sparse **CSR** matrix, and use **NearestNeighbors** (cosine metric) to return the top 5 similar movies for a given title.
5. **Matrix Factorization (SVD)** — use the **Surprise** library's SVD with `n_factors = 4` (d = 4), train on a train/test split, and evaluate with RMSE and MAPE.
6. **MF embeddings for similarity** — redesign the item-item and user-user similarity functions to use the learned 4-dim latent embeddings instead of raw rating vectors; d=2 embeddings are also plotted and analysed.
7. **User-based Pearson (optional)** — for a new user who rates a few movies, find overlapping old users, take the top 100 by movies in common, score similarity via Pearson, take the top 10, and produce weighted recommendations.

## Models & Results

| Model | Technique | Output |
|-------|-----------|--------|
| Item-based | Pearson Correlation | Top-N similar movies to a given title |
| Item-based | Cosine Similarity + KNN | Top-5 nearest movies |
| Matrix Factorization | SVD (Surprise, d=4) | Predicted ratings |
| User-based (optional) | Pearson Correlation | Top-10 movies for a new user |

**Matrix Factorization (SVD, d=4) performance on the held-out test set:**

- **RMSE: 0.8830**
- **MAPE: 26.95%**

The latent-factor model clearly outperforms a naive popularity baseline, showing that the embeddings capture genuine user-taste patterns.

## Key EDA Insights

- The most active raters belong to the **25–34** age group, and **college/grad students** are the most active profession — the core audience is young and educated.
- The majority of raters are **Male**, indicating a gender imbalance worth keeping in mind for content strategy.
- Most movies in the catalogue were released in the **1990s**, and **American Beauty (1999)** has the highest number of ratings, making it a strong anchor for popularity-based fallbacks.

## Questionnaire Answers

| # | Question | Answer |
|---|----------|--------|
| Q1 | Age group that rated the most movies | **25–34** |
| Q2 | Profession that rated the most movies | **college/grad student** |
| Q3 | Most raters are Male (T/F) | **True** |
| Q4 | Decade with most movie releases | **1990s** |
| Q5 | Movie with the maximum number of ratings | **American Beauty (1999)** |
| Q6 | Top 3 movies similar to 'Liar Liar' (item-based Pearson) | **Life (1999), East-West (Est-ouest) (1999), Oliver & Company (1988)** |
| Q7 | Collaborative Filtering is classified into ___ and ___ | **User-based** and **Item-based** |
| Q8 | Pearson range / Cosine range | Pearson: **-1 to +1**; Cosine: **0 to 1** (non-negative ratings; -1 to 1 in general) |
| Q10 | MF model RMSE / MAPE | **RMSE 0.8830, MAPE 26.95%** |
| Q11 | CSR 'row' representation of [[1,0],[3,7]] | values **[1, 3, 7]**, column indices **[0, 0, 1]**, row pointer **[0, 1, 3]** |

## Tech Stack

Python, pandas, NumPy, scikit-learn (NearestNeighbors), SciPy (CSR sparse matrices), Surprise (SVD matrix factorization), Matplotlib / Seaborn.

## How to Run

1. Open `zee.ipynb` in Jupyter or Google Colab.
2. Ensure the three `.dat` files are accessible (update the file paths if needed).
3. Install dependencies: `pip install pandas numpy scikit-learn scipy scikit-surprise matplotlib seaborn`.
4. Run the cells top to bottom.
