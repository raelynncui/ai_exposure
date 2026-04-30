# CIS 2450 Final Project: Predicting AI Exposure from Census Demographic Data
Teresa Shang & Raelynn Cui

Can we predict a census tract's AI exposure score using only its demographic characteristics? This project builds and evaluates models that estimate AI exposure at the census-tract level using ACS data, with the goal of enabling more timely and granular estimates for policymakers, economists, and workforce planners.


## Data Sources

| File | Description |
|---|---|
| `data/ai.csv` | Social Explorer AI Exposure Index — `AI Exposure Score (0-0.29)` by FIPS (census tract level) |
| `data/acs.csv` | ACS demographic data: 1,000+ columns spanning education, employment, occupation, income, commute, and technology access |
| `data/joined_cleaned.csv` | Inner join of the above two files on FIPS code |
| `data/features.csv` | 32 engineered features + target, ready for modeling |
| `data/model_df.csv` | Final feature-selected dataset used as input to all models |
| `data/rf_feature_importance.csv` | Random Forest feature importances |
| `data/model_results.csv` | Test RMSE for each model |
| `feature_groups.json` | Feature-to-group mapping used for group-level MLP analysis |



## Pipeline

### Step 1–2: Join and Clean (`join_clean.py`)

Joins `ai.csv` and `acs.csv` on FIPS code via an inner join, then cleans the result:

- Replaces Census sentinel values (`-666666666`, `-888888888`, `-999999999`, etc.) with `null`
- Removes tracts where the AI Exposure Score is null or zero
- Removes tracts where the total civilian employed population (C24010) is null or zero
- Renames variable codes to human-readable labels using the two-row CSV header structure

Output: `data/joined_cleaned.csv`

### Step 3: Feature Engineering (`feature_engineering.py`)

Reduces 1,035 raw columns to 32 clean, interpretable features across 8 categories. All features are expressed as proportions of the relevant denominator to ensure comparability across tracts of different sizes. Standard error columns and pre-computed percentage columns are dropped throughout.

| Category | ACS Table | Raw Columns | Features | Approach |
|---|---|---|---|---|
| Occupation shares | C24010 | 218 | 4 | Sum male + female; keep 4 of 5 top-level group shares (drop production/transport as reference) |
| Industry shares | C24030 | 164 | 12 | Sum male + female; keep 12 of 13 industry shares (drop other services as reference) |
| Industry × Occupation | C24050 | 251 | 0 | Dropped entirely — redundant with C24010 + C24030 |
| Class of worker | B24080 | 62 | 5 | Sum male + female; combine local/state/federal government into one feature |
| Education | B15003 | 111 | 3 | Collapse 24 grade levels into 3 tiers; drop less-than-HS as reference |
| Income | B19013 + B19301 | 4 | 2 | Keep median household income and per capita income as-is |
| Commute mode + travel time | B08301 + B08303 | 75 | 4 | Keep drove alone, public transit, worked from home; collapse 12 time bins into one 45+ min share |
| Technology access | B28002 + B28001 | 57 | 2 | One broadband share and one computer ownership share |

Output: `data/features.csv`



## Exploratory Data Analysis (`eda.ipynb`)

- **Target distribution**: AI Exposure Score is approximately unimodal and bell-shaped for positive values, centered around ~0.175. A significant point mass at zero (tracts with no measured exposure) was removed prior to modeling.
- **Feature distributions**: Many features show strong right skewness, particularly industry and commuting variables which are concentrated near zero in many regions. Private for-profit employment, broadband access, and driving alone are left-skewed and concentrated near higher values.
- **Correlation structure**: Confirmed strong multicollinearity within the raw joined dataset (many features are strict subsets of others), motivating the feature engineering reduction.



## Feature Selection (`data_analysis.ipynb`)

After feature engineering, a three-step selection process reduces 32 features to the final model input:

**1. Correlation filtering** — Drop one feature from each pair with |r| > 0.80. Dropped: `pct_mgmt_business_arts`, `pct_hs_or_ged`, `median_household_income`.

**2. Mutual information** — Compute MI scores between each feature and the target. Feature groups are ranked by mean MI; the top-k features are identified for interpretability analysis.

**3. PCA** — Visualize variance explained by principal components and inspect PC1/PC2 loadings to understand the primary axes of variation in the feature space.

Features are organized into 6 groups for group-level analysis:

| Group | Features |
|---|---|
| `occupation` | `pct_service`, `pct_sales_office`, `pct_natural_resources_construction` |
| `industry` | 12 industry share features |
| `employment_type` | `pct_private_forprofit`, `pct_self_employed_incorporated`, `pct_nonprofit`, `pct_government`, `pct_self_employed_unincorporated` |
| `education_income` | `pct_some_college_assoc`, `pct_bachelors_plus`, `per_capita_income` |
| `commute` | `pct_drove_alone`, `pct_public_transit`, `pct_worked_from_home`, `pct_long_commute` |
| `digital_access` | `pct_broadband`, `pct_has_computer` |

Output: `data/model_df.csv`, `feature_groups.json`



## Modeling (`models.ipynb`)

All models are evaluated using RMSE on a held-out test set. Data is split 60/20/20 into train/validation/test sets (seed = 42).

### Baseline
A mean predictor (predicting the training mean for all observations) establishes the minimum performance bar.

### Linear Regression
Ordinary least squares, trained without regularization, provides a linear baseline and measures how much predictive signal exists in a linear combination of the demographic features.

### Lasso Regression
L1-regularized linear regression trained across multiple alpha values (`1e-5`, `1e-4`, `1e-3`). The sparse weight vector is inspected to identify which features contribute most to prediction.

### Random Forest
An ensemble of decision trees (`n_estimators=10`, `max_depth=5`). Feature importances are extracted and saved to `data/rf_feature_importance.csv` to understand which demographic variables drive predictions.

### MLP (Multi-Layer Perceptron)
A PyTorch neural network trained with Adam optimizer and L2 weight decay (`1e-4`). Architecture and learning rate are tuned via 10 random trials over:
- Learning rate: `{1e-3, 1e-4, 1e-5}`
- Hidden dimension: `{32, 64, 128, 256}`
- Number of layers: `{2, 3, 4}`

An MLP is also trained separately on each feature group to quantify each group's individual predictive contribution.

Results are saved to `data/model_results.csv`.


## Hypothesis testing
We use a Mann-Whitney U Test to answer the question: Do census tracts with high bachelor's degree attainment have significantly higher AI exposure than tracts with low attainment? We reject the null hypothesis, finding that high-education tracts have significantly higher AI exposure (p < 0.05).



## Setup

```bash
pip install polars duckdb pandas numpy scikit-learn torch pytorch-lightning matplotlib seaborn
```

Run the pipeline in order:

```bash
python join_clean.py
python feature_engineering.py
jupyter notebook eda.ipynb
jupyter notebook data_analysis.ipynb
jupyter notebook models.ipynb
```

## Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```