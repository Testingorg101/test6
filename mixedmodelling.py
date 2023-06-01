import pandas as pd
import statsmodels.api as sm

# Load the data
data = pd.read_csv('data.csv')

# Create the design matrix
X = data[['TV', 'Radio', 'Newspaper']]  # Independent variables (marketing channels)
y = data['Sales']  # Dependent variable (sales)

# Add a constant term to the design matrix
X = sm.add_constant(X)

# Fit the market mix model
model = sm.OLS(y, X)
results = model.fit()

# Print the results summary
print(results.summary())
