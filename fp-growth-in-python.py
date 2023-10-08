from mlxtend.frequent_patterns import fpgrowth
import pandas as pd

# Load the one-hot encoded dataset
data = pd.DataFrame({
    'Transaction': ['T1', 'T2', 'T3', 'T4', 'T5'],
    'Milk': [1, 1, 1, 1, 0],
    'Bread': [1, 1, 0, 1, 1],
    'Butter': [1, 0, 0, 1, 1],
    'Diapers': [0, 1, 1, 0, 0],
    'Beer': [0, 0, 1, 0, 0]
})

# Use FP-Growth to find frequent itemsets with min_support=0.4
frequent_itemsets = fpgrowth(data.drop('Transaction', axis=1), min_support=0.4, use_colnames=True)

print(frequent_itemsets)
