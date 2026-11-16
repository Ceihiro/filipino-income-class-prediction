# Filipino Household Income Class Prediction Using Machine Learning

A machine learning classification project that predicts Filipino household income classes (Low, Middle, High) using the Family Income and Expenditure Survey (FIES) dataset. The trained classifier is then used for scenario-based projection to explore whether Low, Middle, and High income households would still persist from 2027 to 2050 under assumed future household conditions.

---

## Methodology Note

This project uses a **historical cross-sectional dataset** (FIES, approximately 2015–2020) to train a multiclass classifier. Because the dataset captures household conditions at a single point in time, this is **not a time-series forecast**. Instead, the trained model is applied to **scenario-based simulated data** representing hypothetical future household profiles (2027–2050). The projection answers the question: *"If future Filipino households have characteristics X, Y, and Z, what income class would the model predict?"* This is a scenario simulation, not a prediction of actual future income trends.

## Dataset

- **Source:** [OpenML](https://www.openml.org/search?type=data&sort=runs&id=43443&status=active) (Dataset ID 43443)
- **Original source:** Philippine Statistics Authority (PSA) — Family Income and Expenditure Survey
- **License:** CC0 (Public Domain)
- **Retrieval date:** 2026-11-02
- **File format:** Original `dataset_` (ARFF), exported to `dataset.csv` and `cleaned_data.csv`

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total Records | 41,544 households |
| Total Features (including target) | 60 original + 3 derived |
| Target Variable | `Income_Class` (Low, Middle, High) |
| Duplicate Records | 0 |
| Missing Values — `Household_Head_Occupation` | 7,536 |
| Missing Values — `Household_Head_Class_of_Worker` | 7,536 |
| Missing Values — `Toilet_Facilities` | 1,580 |

---

## Team Members and Roles

| Name | Role | Responsibilities |
|------|------|----------------|
| Kim | Data Engineer | Data loading, cleaning, validation, feature engineering |
| Eyro | Analyst | EDA, visualization, statistical analysis |
| Ceihiro | ML Engineer | Model training, tuning, evaluation, deployment |
| Kenn | Documentation Lead | Overleaf paper, README, GitHub, logs |

*All members commit under their own accounts. Seed 42 is used for reproducibility.*

---

## Repository Structure

```
├── dataset.txt                    # Raw ARFF file (do not edit)
├── dataset.csv                    # Initial CSV export from ARFF
├── dataset_shuffled.csv           # Pre-shuffled with seed 42
├── cleaned_data.csv               # Fully preprocessed dataset
├── cleaning_log.txt               # Step-by-step preprocessing decisions
├── 01_cleaning.ipynb              # Google Colab notebook (ARFF parsing, shuffle)
├── Orange_workflow.ows            # Orange Data Mining workflow file
├── README.md                      # This file
├── figures/                       # EDA charts and screenshots
│   ├── fig_income_distribution.png
│   ├── fig_income_class_distribution.png
│   ├── fig_education_mosaic.png
│   ├── fig_regional_boxplot.png
│   └── fig_correlations.png
└── paper/
    └── conference_101719.tex      # IEEE Conference Template (Overleaf)
```

---

## Data Preprocessing Summary

All preprocessing was performed in **Orange Data Mining** with reproducible steps:

1. **ARFF Parsing:** `dataset.txt` was parsed in Python using `liac-arff` and exported to CSV.
2. **Column Renaming:** Fixed 5 misspelled column names (e.g., `Enterpreneurial` → `Entrepreneurial`).
3. **Categorical Cleanup:** Fixed Region values (` ARMM` → `ARMM`, `Zasmboanga` → `Zamboanga`) and wall type (`NOt applicable` → `Not applicable`).
4. **Missing Values:** 7,536 missing occupation records were labeled as `"No Job/Business"` instead of dropped. Toilet_Facilities missing values labeled as `"Unknown"`.
5. **Shuffling:** Dataset was pre-shuffled with `random_state=42` in pandas to prevent regional bias in train-test splits.
6. **Feature Engineering:** Created `Employed_Ratio` (employed members / total family members) and `Income_Copy`.
7. **Target Creation:** `Income_Copy` was discretized into 3 equal-frequency bins and renamed to `Income_Class` (Low, Middle, High).
8. **Age Check:** 43 household heads under age 18 were retained after review (no clear data entry error).

---

## Key EDA Findings

- **Income is heavily right-skewed:** Most households cluster below 250,000 pesos, with extreme outliers reaching ~11.8 million pesos.
- **Balanced target classes:** Equal-frequency discretization produced roughly 13,848 households per class.
- **Education matters:** Mosaic Display shows higher education strongly associated with High income class.
- **Regional inequality:** Box Plot reveals ARMM has the lowest median income; northern and central regions are higher.
- **Strong correlations:** Expenditure variables (housing, food, communication) are highly correlated with total income.

---

## How to Use

### Option 1: Orange Data Mining (Recommended for EDA and modeling)
1. Open `Orange_workflow.ows` or manually load `dataset_shuffled.csv` in the **File** widget.
2. Follow the documented preprocessing chain: Edit Domain → Impute → Formula → Discretize → Select Columns.
3. Connect modeling widgets (Test & Score, Logistic Regression, Random Forest, etc.) to the **Data Sampler** branch.

### Option 2: Python
```python
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Target variable
y = df["Income_Class"]

# Features (all columns except target)
X = df.drop(columns=["Income_Class"])
```

---

## Random Seed

- **Agreed group seed:** `42`
- **Purpose:** Ensures reproducible train-test splits and consistent results across all team members.
- **Implementation:** Dataset is pre-shuffled in Python with `random_state=42` before loading into Orange. The Data Sampler widget is set to 100% with replicable deterministic sampling.

---

## Citation

If you use this dataset, please cite:

> Philippine Statistics Authority. *Family Income and Expenditure Survey (FIES)*. Available on OpenML: https://www.openml.org/d/43443 (CC0 License).

---

## License

This project uses the **CC0-licensed** FIES dataset from OpenML. All code and documentation in this repository are provided for academic purposes.
