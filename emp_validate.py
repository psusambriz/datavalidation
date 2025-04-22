import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df = pd.read_csv('employees.csv')
# Existence assertions

# count rows where 'name' field is empty
invalid_name = df['name'].isnull().sum() + (df['name'] == '').sum()
print(f"Number of records that violate the assertion: {invalid_name}")


# Limit assertions

# every employee was hired no earlier than 2015
df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')

invalid_hire_date = (df['hire_date'].dt.year<2015).sum()

print(f"Number of records that violate hire date assertion: {invalid_hire_date}")

# Intra-record assertions

# each employee was born before they were hired
df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
invalid_birth_before_hire = (df['birth_date'] >= df['hire_date']).sum()

print(f"Number of records that violate birth-before hire assertion: {invalid_birth_before_hire}")

# Inter-record assertions

# each employee has a manager that knows an employee

known_employee_ids = set(df['eid'].dropna())
invalid_managers = ((~df['reports_to'].isin(known_employee_ids)) & df['reports_to'].notnull()).sum()
print(f"Number of records that violate employee/manager assertion: {invalid_managers}")

# Summary Assertion

# each city has more than one employee
city_count = df['city'].value_counts()

# identify cities with 1 employee
invalid_city_counts = (city_count == 1).sum()

print(f"Number of cities that violate the summary assertion: {invalid_city_counts}")

# Statistical Assertion

salaries = df['salary'].dropna()

# plot
plt.figure(figsize=(10,6))
count,bins,ignored = plt.hist(salaries,bins=30,density=True,alpha=0.6,color='skyblue',edgecolor='black')
# Plot normal distribution over it
mu, std = salaries.mean(), salaries.std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2)

plt.title('Histogram of Salaries with Normal Curve')
plt.xlabel('Salary')
plt.ylabel('Density')
plt.grid(True)
plt.show()