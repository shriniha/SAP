import pandas as pd
import random
from datetime import datetime, timedelta

# Define the number of samples for the new test dataset
num_samples = 15  # Adjust the number as needed for your test dataset

# Generate synthetic data for expense reports
expenses = []
for i in range(num_samples):
    date = datetime.now() - timedelta(days=random.randint(1, 365))
    amount = round(random.uniform(10, 500), 2)
    category = random.choice(['Meals', 'Travel', 'Office Supplies', 'Miscellaneous', 'Equipment', 'Training', 'Utilities', 'Wifi', 'Software', 'Maintenance', 'Marketing'])
    description = f"{category} expense on {date.strftime('%Y-%m-%d')}"
    expenses.append({'Date': date, 'Amount': amount, 'Category': category, 'Description': description})

# Create a DataFrame from the generated data
df = pd.DataFrame(expenses)

# Save the dataset to a CSV file
df.to_csv('new_expense_data.csv', index=False)

print("New test dataset saved as 'new_expense_data.csv'")
