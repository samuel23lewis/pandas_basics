from matplotlib import pyplot as plt
import numpy as np

# Set random seed for reproducibility
# np.random.seed(42)

# Generate 100 household incomes between $20,000 and $200,000
household_income = np.random.uniform(20000, 200000, 100)

# Print the first 10 entries as an example
print(household_income[:500])

plt.hist(household_income,5)
plt.show()


