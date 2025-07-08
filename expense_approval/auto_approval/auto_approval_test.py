import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# Load the saved model and tokenizer from the local directory
model = ElectraForSequenceClassification.from_pretrained(r'C:\Users\shrin\OneDrive\Desktop\Django\expense_approval\auto_approval\auto_approval_model')
tokenizer = ElectraTokenizer.from_pretrained(r'C:\Users\shrin\OneDrive\Desktop\Django\expense_approval\auto_approval\auto_approval_model')

# Load or prepare new test data
# For this example, let's assume you have a CSV file with new test data
new_data = pd.read_csv('new_expense_data.csv')
test_texts = new_data['Description'].tolist()

# Tokenize and encode the test data
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

# If you have ground truth labels for the test set, you can calculate accuracy
if 'AutoApprove' in new_data.columns:
    test_labels = new_data['AutoApprove'].tolist()
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Test Accuracy: {accuracy:.4f}')

# Print predictions
for text, pred in zip(test_texts, predictions):
    approval_status = "Auto-Approve" if pred == 1 else "Manual Review"
    print(f'Expense: {text} -> {approval_status}')
