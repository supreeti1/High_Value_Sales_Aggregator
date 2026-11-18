import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'Customer_Age': np.random.randint(18, 70, n_samples),
    'Annual_Income': np.random.randint(20000, 150000, n_samples),
    'Purchase_Amount': np.random.randint(100, 5000, n_samples),
    'Product_Category': np.random.choice([0, 1, 2, 3], n_samples), # 0: Electronics, 1: Apparel, 2: Home, 3: Luxury
    'Previous_Purchases': np.random.randint(0, 20, n_samples)
}

df = pd.DataFrame(data)
# Target: High Value Sale (1 if Purchase Amount > 2500 or Luxury category, else 0)
df['High_Value'] = np.where((df['Purchase_Amount'] > 2500) | (df['Product_Category'] == 3), 1, 0)

df.to_csv('data/sales_data.csv', index=False)
print("Dataset created successfully at data/sales_data.csv")
