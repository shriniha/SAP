import pandas as pd
import random

# Load the approvers dataset
approvers = pd.read_csv('approvers.csv')

# Load the substitutes dataset
substitutes = pd.read_csv('substitutes.csv')

# Initialize an empty list to store the pairs of approver and substitute
pairs = []

# Iterate through each approver
for _, approver in approvers.iterrows():
    # Filter substitutes based on the same department as the approver
    department_substitutes = substitutes[substitutes['Department'] == approver['Department']]
    
    # If there are substitutes available for the same department
    if not department_substitutes.empty:
        # Choose a random substitute from the filtered list
        substitute = random.choice(department_substitutes['Name'].tolist())
        
        # Add the pair of approver and substitute to the list
        pairs.append({'ApproverName': approver['Name'], 'SubstituteName': substitute})

# Create a DataFrame from the list of pairs
pairs_df = pd.DataFrame(pairs)

# Save the dataset to a CSV file
pairs_df.to_csv('approver_substitute_pairs.csv', index=False)

print("Dataset containing approver-substitute pairs saved as 'approver_substitute_pairs.csv'")
