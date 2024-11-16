import numpy as np
import matplotlib.pyplot as plt

# Simulate portfolio data
np.random.seed(42)
numPortfolios = 10000
numAssets = 3
meanReturns = np.random.rand(numAssets) * 0.15  # Simulated expected returns (15% max)
cov_matrix = np.random.rand(numAssets, numAssets)
cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make symmetric (valid covariance matrix)
cov_matrix += np.eye(numAssets) * 0.01  # Add small variance to diagonal

# Generate random weights and compute portfolio returns and risks
results = np.zeros((3, numPortfolios))
for i in range(numPortfolios):
    weights = np.random.rand(numAssets)
    weights /= np.sum(weights)
    portfolio_return = np.dot(meanReturns, weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    results[0, i] = portfolio_return
    results[1, i] = portfolio_risk
    results[2, i] = portfolio_return / portfolio_risk  # Sharpe Ratio

# Extract optimal portfolios
max_sharpe_idx = np.argmax(results[2])
min_risk_idx = np.argmin(results[1])

# Visualize efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.6, marker='.')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], c='red', label='Max Sharpe Ratio')
plt.scatter(results[1, min_risk_idx], results[0, min_risk_idx], c='blue', label='Min Risk')
plt.title('Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.show()