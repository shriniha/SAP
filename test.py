import pandas as pd
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# Load the saved model and tokenizer from the local directory
model = ElectraForSequenceClassification.from_pretrained('C:/Users/shrin/OneDrive/Desktop/Django/expense_approval/automatic_substitute/vacation_substitution_model')
tokenizer = ElectraTokenizer.from_pretrained('C:/Users/shrin/OneDrive/Desktop/Django/expense_approval/automatic_substitute/vacation_substitution_model')

# Load the test dataset
test_data = pd.read_csv('test_data.csv')

# Load the dataset containing approver-substitute pairs
substitute_pairs = pd.read_csv('approver_substitute_pairs.csv')

# Tokenize the test data
test_texts = test_data.apply(lambda x: f"{x['ApproverName']} {x['Department']} {x['Position']} {x['Date']}", axis=1).tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

# Move tensors to the appropriate device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
input_ids = test_encodings['input_ids'].to(device)
attention_mask = test_encodings['attention_mask'].to(device)

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits

# Get predictions
predictions = torch.argmax(logits, dim=1).tolist()

# Print predictions along with the respective substitute pair
for i, pred in enumerate(predictions):
    approver_name = test_data.loc[i, 'ApproverName']
    needs_substitution = "Yes" if pred == 1 else "No"
    
    # Find the respective substitute pair
    substitute_pair = substitute_pairs[substitute_pairs['ApproverName'] == approver_name]
    substitute_name = substitute_pair['SubstituteName'].values[0] if not substitute_pair.empty else "N/A"
        
    print(f'Approver: {approver_name}, Needs Substitution: {needs_substitution}, Substitute Name: {substitute_name}')
