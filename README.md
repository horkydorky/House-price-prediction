# Kaggle House Prices: Advanced Regression Techniques
This repository contains my solution for the Kaggle "House Prices: Advanced Regression Techniques" competition. The goal is to predict the final sale price of homes in Ames, Iowa using advanced machine learning models.

## Final Result
*   **Kaggle Score (RMSE):** **0.12028**
*   **Leaderboard Ranking:** **Top 7%**

## Methodology
The project follows a structured machine learning pipeline:

### 1. Data Cleaning & Preprocessing
*   Handled missing values using context-aware strategies (e.g., imputing `LotFrontage` with the median of its neighborhood).
*   Identified and removed known outliers from the training data based on `GrLivArea` to improve model stability.
*   Applied a log transformation (`np.log1p`) to the target variable `SalePrice` to normalize its distribution.

### 2. Feature Engineering
*   Created new, impactful features like `TotalSF` (total square footage) and `HouseAge`.
*   Converted key ordinal features (e.g., `ExterQual`, `KitchenQual`) into numerical rankings to preserve their inherent order.
*   One-hot encoded the remaining nominal categorical features.

### 3. Modeling & Optimization
*   Established a strong baseline using a tuned **XGBoost** model, optimized with Bayesian hyperparameter search (`hyperopt`).
*   Implemented feature selection using `SelectFromModel` to reduce noise and improve the XGBoost model's performance.
*   Developed a **stacking ensemble** combining several diverse models (XGBoost, LightGBM, RandomForest, and Ridge regression) to leverage their individual strengths and achieve the final score.

### 4. Validation
*   Used 5-fold cross-validation (`cross_val_score`) to robustly evaluate model performance and prevent overfitting during the tuning and selection phases.

## Repository Structure
*   **/data**: Contains the original `train.csv` and `test.csv` files from the competition.
*   **/notebooks**: Includes the main Jupyter Notebook with the full analysis and modeling code.
*   **/submission**: Contains the final generated `submission.csv` file.

## Technologies Used
*   Python
*   Pandas & NumPy
*   Scikit-learn
*   XGBoost & LightGBM
*   Hyperopt
*   Matplotlib & Seaborn
