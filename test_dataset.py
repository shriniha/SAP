import pandas as pd
import random
from datetime import datetime, timedelta

# Load the Approvers dataset
df_approvers = pd.read_csv('approvers.csv')

# Define the number of samples for the test dataset
num_test_samples = 5

# Generate synthetic data for the test dataset
test_data = []
for _ in range(num_test_samples):
    # Generate a random date
    random_date = datetime.now() - timedelta(days=random.randint(1, 365))

    # Select a random approver
    approver = random.choice(df_approvers.to_dict(orient='records'))

    # Extract relevant information
    approver_name = approver['Name']
    department = approver['Department']
    position = approver['Position']
    date = random_date.strftime('%Y-%m-%d')

    # Append to the test data
    test_data.append({
        'ApproverName': approver_name,
        'Department': department,
        'Position': position,
        'Date': date
    })

# Create DataFrame for the test data
df_test = pd.DataFrame(test_data)

# Save the test DataFrame to a CSV file
df_test.to_csv('test_data.csv', index=False)

print("Test data saved as 'test_data.csv'")
