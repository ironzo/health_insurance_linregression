# Health Insurance Cost Prediction using Linear Regression

This project aims to predict health insurance charges based on several individual factors using Linear Regression. It includes a custom implementation of the Linear Regression algorithm from scratch, as well as a comparison with the `scikit-learn` implementation.

## Project Overview

The goal is to understand the factors that influence health insurance premiums and build a predictive model. The project follows a standard data science workflow:
1.  **Data Exploration & Analysis**: Understanding the dataset and relationships between features.
2.  **Data Preprocessing**: Handling outliers, encoding categorical variables, and normalizing features.
3.  **Model Implementation**: Building a Linear Regression model from scratch using Gradient Descent.
4.  **Evaluation**: Comparing the custom model with a standard library implementation and analyzing performance metrics.

## Dataset

The model uses the `insurance.csv` dataset, which includes the following features:
- `age`: Age of the primary beneficiary.
- `sex`: Insurance contractor gender (female, male).
- `bmi`: Body Mass Index (kg/m²).
- `children`: Number of children covered by health insurance / Number of dependents.
- `smoker`: Smoking status (yes, no).
- `region`: The beneficiary's residential area in the US.
- `charges`: Individual medical costs billed by health insurance (Target variable).

## Data Source

Data was taken from Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance/data

## Key Findings from EDA

- **Outliers**: Outliers were identified in the `bmi` (above 47) and `charges` columns and were removed to improve model performance.
- **Correlations**: Smoking showed the strongest positive correlation (0.79) with insurance charges.
- **Smoking & BMI**: Analysis revealed that individuals with obesity (Class 1 and 2) and those who are overweight are more prevalent among smokers compared to non-smokers.

## Implementation Details

### Custom Linear Regression
The notebook implements the following mathematical components:
- **Cost Function**: Mean Squared Error (MSE) calculation.
- **Gradient Function**: Partial derivatives of the cost function with respect to weights ($w$) and bias ($b$).
- **Gradient Descent**: An iterative optimization algorithm to minimize the cost function.

**Parameters used:**
- Learning Rate ($\alpha$): 0.109
- Iterations: 10,000

### Normalization
Features like `age` and `bmi` were normalized by dividing them by their respective maximum values to help gradient descent converge faster.

## Results

Both the custom implementation and the `scikit-learn` model achieved nearly identical results on the test set:

- **Mean Absolute Error (MAE)**: ~4,099.66
- **Mean Squared Error (MSE)**: ~34,513,463.07
- **R-squared Score**: 0.77

The model explains approximately 77% of the variability in insurance charges. While the predictive capability is strong, the model's accuracy tends to decrease as the target charges increase, suggesting that higher premiums might be influenced by more complex factors not fully captured by a simple linear model.
