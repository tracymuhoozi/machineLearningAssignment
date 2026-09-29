# Diabetes Prediction: Exploratory Data Analysis

Exploratory data analysis (EDA) of a diabetes dataset, looking at which clinical
factors are linked to a diabetes diagnosis and what this means for modelling.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Selected Charts](#selected-charts)
- [How to Run](#how-to-run)
- [Tools](#tools)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

## Project Structure

```
machineLearningAssignment/
├── data/              # dataset (CSV)
├── notebooks/         # eda.ipynb
├── images/            # charts exported from the notebook
├── reports/           # written report
├── src/               # Python scripts
├── requirements.txt   # Python dependencies
└── README.md
```

## Dataset

**Source:** [Diabetes Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

| Feature | Description |
|---|---|
| `gender` | Patient gender |
| `age` | Age in years |
| `hypertension` | 0 = no, 1 = yes |
| `heart_disease` | 0 = no, 1 = yes |
| `smoking_history` | Smoking status category |
| `bmi` | Body mass index |
| `HbA1c_level` | Average blood sugar level over about 3 months (%) |
| `blood_glucose_level` | Blood glucose (mg/dL) |
| `diabetes` | Target: 0 = no diabetes, 1 = diabetes |

The notebook uses a transformed version of this data with extra engineered
columns: category bands (HbA1c, BMI, blood glucose), interaction terms and
binary risk flags.

To reproduce the analysis, download the CSV from Kaggle and place it in the
`data/` folder.

## Key Findings

- **Class imbalance:** about 8.8% of records are diabetic, so techniques such as
  SMOTE or class weights are needed when modelling.
- **Strongest predictors:** HbA1c level and blood glucose level, followed by age
  and BMI.
- **Age:** the diabetes rate rises sharply after age 50, and the 60+ group has
  the highest prevalence.
- **Cardiovascular conditions:** patients with both hypertension and heart
  disease show an elevated diabetes rate.
- **BMI:** the obesity category has the highest diabetes rate.
- **Clinical thresholds:** a high HbA1c combined with a high blood glucose level
  points strongly to diabetes.
- **Smoking:** former and ever smokers show slightly higher rates than never
  smokers.
- **Engineered features:** the glucose x HbA1c interaction separates the two
  classes clearly and should help tree-based models.
- **Gender:** little difference overall, with males slightly higher in older age
  groups.

## Selected Charts

### Class distribution
![Class distribution](images/01_class_distribution.png)

### Continuous feature distributions
![Continuous distributions](images/02_continuous_distributions.png)

### Diabetes rate by risk profile
![Risk profile](images/04_risk_profile.png)

### Correlation heatmap
![Correlation heatmap](images/05_correlation_heatmap.png)

### BMI category analysis
![BMI analysis](images/11_bmi_category_analysis.png)

## How to Run

**1. Clone the repository**

```bash
git clone git@github.com:tracymuhoozi/machineLearningAssignment.git
cd machineLearningAssignment
```

**2. Create a virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Add the dataset**

Download the CSV from the Kaggle link above and place it in the `data/` folder.

**4. Open the notebook**

Open `notebooks/eda.ipynb` in VS Code (with the Python and Jupyter extensions)
or in Jupyter, select the `venv` kernel, then run all cells.

## Tools

- Python 3
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Jupyter

## Acknowledgements

Dataset provided on Kaggle by Mohammed Mustafa (iammustafatz). Please check the
Kaggle page for the author name and license terms.