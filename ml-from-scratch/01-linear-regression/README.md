# Project 1: Linear Regression from Scratch

Multivariate linear regression with gradient descent, written with NumPy only.
scikit-learn is used only to load the data and to check my result.

## Data
California Housing (`sklearn.datasets.fetch_california_housing`): 20,640 districts,
8 features, target = median house value in $100k.

## What it does
1. `z_score`: scales each feature to mean 0, std 1 (population std, NumPy default).
2. `compute_fx`: prediction `f(x) = w·x + b`.
3. `cost_function_MSE`: `J(w,b) = (1/2m) Σ (f(x_i) - y_i)²`.
4. `compute_gradient`: vectorized, `dJ/dw = (1/m) Xᵀ(f - y)`, `dJ/db = (1/m) Σ(f - y)`.
5. `gradient_descent`: updates `w` and `b` together, `w = w - α·dJ/dw`, `b = b - α·dJ/db`,
   and records J at every step.

Settings: α = 0.1, 5000 iterations, w and b start at 0.

## Results
- Cost at w = 0, b = 0: **2.8052**
- Cost after gradient descent: **0.2622** (it stops improving after about 500 iterations)
- My `w` and `b` match `sklearn.linear_model.LinearRegression` on the same scaled data
  (all digits shown):

| | b | w[0] (MedInc) | w[6] (Latitude) | w[7] (Longitude) |
|---|---|---|---|---|
| Mine | 2.0686 | 0.8296 | -0.8999 | -0.8705 |
| sklearn | 2.0686 | 0.8296 | -0.8999 | -0.8705 |

With 1000 iterations the weights were close but not equal (w[0] = 0.8308), so gradient
descent had not fully converged yet.

## Plots
- `J` vs iterations: smooth decrease, then flat.
- Predicted vs actual: points follow the diagonal. The flat row at the top is a data
  artifact: house values are capped at 5.0 ($500k) in this dataset.

## Notes
- Without scaling, one learning rate does not work for all features (their ranges
  differ by orders of magnitude).
- The intercept b equals the mean of y (2.0686) because the features are centered.

## Run
Open `linear_regression.ipynb` and run all cells. Needs `numpy`, `matplotlib`, `scikit-learn`.
