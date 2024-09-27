import pandas as pd
import random
from datetime import datetime, timedelta

# Define the number of samples
num_samples = 10000

# Generate synthetic data for expense reports
expenses = []
for i in range(num_samples):
    date = datetime.now() - timedelta(days=random.randint(1, 365))
    amount = round(random.uniform(10, 500), 2)
    category = random.choice(['Meals', 'Travel', 'Office Supplies', 'Miscellaneous', 'Equipment', 'Training', 'Utilities', 'Wifi', 'Software', 'Maintenance', 'Marketing'])
    description = f"{category} expense on {date.strftime('%Y-%m-%d')}"
    department = random.choice(['Finance', 'HR', 'IT', 'Operations'])
    urgency = random.choice(['Low', 'Medium', 'High'])
    complexity = random.choice(['Low', 'Medium', 'High'])
    auto_approve = False
    if category in ['Office Supplies', 'Utilities', 'Maintenance', 'Wifi']:
        auto_approve = True
    expenses.append({'Date': date, 'Amount': amount, 'Category': category, 'Description': description,
                     'Department': department, 'Urgency': urgency, 'Complexity': complexity, 'AutoApprove': auto_approve})

# Generate synthetic data for approvers
approvers = [
    {'Name': 'Alice', 'Position': 'Manager'},
    {'Name': 'Bob', 'Position': 'Supervisor'},
    {'Name': 'Charlie', 'Position': 'Director'},
    {'Name': 'David', 'Position': 'Manager'},
    {'Name': 'Eve', 'Position': 'Supervisor'}
]

# Generate synthetic data for spending limits
spending_limits = {'Alice': 1000, 'Bob': 1500, 'Charlie': 2000, 'David': 2500, 'Eve': 3000}

# Generate synthetic data for compliance rules
compliance_rules = {
    'Meals': {'Limit': 50, 'RequireReceipt': True},
    'Travel': {'Limit': 500, 'RequireReceipt': True},
    'Office Supplies': {'Limit': 100, 'RequireReceipt': False},
    'Miscellaneous': {'Limit': 200, 'RequireReceipt': False},
    'Equipment': {'Limit': 1000, 'RequireReceipt': True},
    'Training': {'Limit': 300, 'RequireReceipt': True},
    'Utilities': {'Limit': 200, 'RequireReceipt': False},
    'Wifi': {'Limit': 50, 'RequireReceipt': True},
    'Software': {'Limit': 200, 'RequireReceipt': True},
    'Maintenance': {'Limit': 500, 'RequireReceipt': True},
    'Marketing': {'Limit': 1000, 'RequireReceipt': True}
}

# Create a DataFrame from the generated data
df = pd.DataFrame(expenses)
df['Approver'] = df.apply(lambda x: random.choice(approvers), axis=1)
df['ApproverName'] = df['Approver'].apply(lambda x: x['Name'])
df['ApproverPosition'] = df['Approver'].apply(lambda x: x['Position'])
df['SpendingLimit'] = df['ApproverName'].map(spending_limits)
df['RequireReceipt'] = df['Category'].map(lambda x: compliance_rules[x]['RequireReceipt'])
df['ComplianceLimit'] = df['Category'].map(lambda x: compliance_rules[x]['Limit'])

# Save the dataset to a CSV file
df.to_csv('expense_dataset.csv', index=False)
