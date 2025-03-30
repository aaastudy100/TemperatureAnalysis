#The python script analyze the monthly temperature anomaly data in the past 30 years, and generate the Polynomial Curve. The output can be seen as the screenshot image PolynomialCurve.jpg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('30years_monthlydata.txt', sep=r'\s+', header=0, names=["YearMonth", "Anomaly"])
data['Date'] = pd.to_datetime(data['YearMonth'], format='%Y%m')
data['MonthsSinceStart'] = ((data['Date'].dt.year - data['Date'].min().year) * 12 +
                            (data['Date'].dt.month - data['Date'].min().month))

# Step 2: Extract x and y
x = data['MonthsSinceStart'].values
y = data['Anomaly'].values

# Step 3: Fit a polynomial curve (e.g., degree 3)
degree = 20
coefficients = np.polyfit(x, y, deg=degree)  # Fit polynomial
polynomial = np.poly1d(coefficients)  # Create polynomial equation

# Step 4: Calculate predicted y values
y_pred = polynomial(x)

# Step 5: Calculate R-squared (coefficient of determination)
ss_total = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
ss_residual = np.sum((y - y_pred) ** 2)  # Residual sum of squares
r_squared = 1 - (ss_residual / ss_total)

# Generate the polynomial equation as a string
equation_terms = [
    f"{coeff:.4e}x^{degree - i}" if degree - i > 0 else f"{coeff:.4e}"
    for i, coeff in enumerate(coefficients)
]
equation = " + ".join(equation_terms)

# Print the polynomial equation and R-squared value
print(f"Polynomial Equation (Degree {degree}):")
print(polynomial)
print(f"R-squared: {r_squared:.4f}")

# Step 6: Plot the original data and polynomial curve
x_smooth = np.linspace(x.min(), x.max(), 500)  # Generate smooth x values
y_smooth = polynomial(x_smooth)  # Generate smooth y values

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data', color='blue', s=10)
plt.plot(x_smooth, y_smooth, label=f'Polynomial Fit (Degree {degree})', color='red', linewidth=2)
plt.title('Polynomial Curve Fit')
plt.xlabel('Months Since Start (1994-Jan)')
plt.ylabel('Temperature Anomaly (°C)')

# Add equation and R-squared to the graph
plt.text(0.01 * x.max(), 0.9 * y.max(), f"Equation:\n{equation}", fontsize=10, color='black')
plt.text(0.05 * x.max(), 0.8 * y.max(), f"$R^2$: {r_squared:.4f}", fontsize=10, color='black')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
