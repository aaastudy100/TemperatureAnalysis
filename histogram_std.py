#The python script analyze the monthly temperature anomaly data in the past 30 years, and generate the Normal Distribution Graph. The output can be seen as the screenshot image histogram_std.jpg

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
file_path = '30years_monthlydata.txt'  # Update this path if needed
data = pd.read_csv(file_path, sep=r'\s+', header=0, names=["YearMonth", "Anomaly"])

# Convert 'YearMonth' to datetime format
data['YearMonth'] = pd.to_datetime(data['YearMonth'], format='%Y%m')

# Calculate the mean
mean_anomaly = data['Anomaly'].mean()

# Calculate the standard deviation
std_anomaly = data['Anomaly'].std()

# Calculate boundaries for μ ± σ
lower_bound = mean_anomaly - std_anomaly
upper_bound = mean_anomaly + std_anomaly

# Calculate boundaries for μ ± 2σ
lower_bound_2std = mean_anomaly - 2 * std_anomaly
upper_bound_2std = mean_anomaly + 2 * std_anomaly

# Calculate boundaries for μ ± 3σ
lower_bound_3std = mean_anomaly - 3 * std_anomaly
upper_bound_3std = mean_anomaly + 3 * std_anomaly

# Print results
print(f"Mean (μ): {mean_anomaly:.2f}")
print(f"μ ± σ: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"μ ± 2σ: [{lower_bound_2std:.2f}, {upper_bound_2std:.2f}]")
print(f"μ ± 3σ: [{lower_bound_3std:.2f}, {upper_bound_3std:.2f}]")

# Calculate the percentages of data within 1σ, 2σ, and 3σ
within_one_std = data[(data['Anomaly'] >= lower_bound) & (data['Anomaly'] <= upper_bound)]
percentage_within_one_std = (len(within_one_std) / len(data)) * 100

within_two_std = data[(data['Anomaly'] >= lower_bound_2std) & (data['Anomaly'] <= upper_bound_2std)]
percentage_within_two_std = (len(within_two_std) / len(data)) * 100

within_three_std = data[(data['Anomaly'] >= lower_bound_3std) & (data['Anomaly'] <= upper_bound_3std)]
percentage_within_three_std = (len(within_three_std) / len(data)) * 100

# Print the results
print(f"Percentage of data within μ ± σ: {percentage_within_one_std:.2f}%")
print(f"Percentage of data within μ ± 2σ: {percentage_within_two_std:.2f}%")
print(f"Percentage of data within μ ± 3σ: {percentage_within_three_std:.2f}%")

# Plot the histogram of the anomalies
plt.figure(figsize=(12, 6))
plt.hist(data['Anomaly'], bins=20, density=True, color='blue', alpha=0.6, edgecolor='black', label='Observed Data')

# Generate x values (covering μ ± 4σ for better visualization of the normal curve)
x = np.linspace(mean_anomaly - 4 * std_anomaly, mean_anomaly + 4 * std_anomaly, 1000)

# Calculate y values for the normal distribution
y = (1 / (std_anomaly * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_anomaly) / std_anomaly)**2)
# Plot the normal distribution curve
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution (Empirical Rule)')

# Highlight the mean ± standard deviation boundaries
plt.axvline(mean_anomaly, color='green', linestyle='--', linewidth=1.5, label='Mean (μ)')
plt.axvline(lower_bound, color='orange', linestyle='--', linewidth=1.5, label='μ - σ')
plt.axvline(upper_bound, color='orange', linestyle='--', linewidth=1.5, label='μ + σ')
plt.axvline(lower_bound_2std, color='purple', linestyle='--', linewidth=1.5, label='μ - 2σ')
plt.axvline(upper_bound_2std, color='purple', linestyle='--', linewidth=1.5, label='μ + 2σ')
plt.axvline(lower_bound_3std, color='brown', linestyle='--', linewidth=1.5, label='μ - 3σ')
plt.axvline(upper_bound_3std, color='brown', linestyle='--', linewidth=1.5, label='μ + 3σ')

# Annotate the percentages
plt.text(mean_anomaly, 0.6, f"{percentage_within_one_std:.2f}% within μ ± σ",
         color='black', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(mean_anomaly, 0.5, f"{percentage_within_two_std:.2f}% within μ ± 2σ",
         color='black', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(mean_anomaly, 0.4, f"{percentage_within_three_std:.2f}% within μ ± 3σ",
         color='black', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))

plt.title('Distribution with Empirical Rule and Standard Deviation Boundaries')
plt.xlabel('Monthly Temperature Anomaly')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()
