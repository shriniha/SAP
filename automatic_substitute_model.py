import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd

# Load historical substitutions dataset
df_historical = pd.read_csv('historical_substitutions.csv')

# Load vacation schedule dataset
df_vacation = pd.read_csv('vacation_schedule.csv')

# Merge datasets based on 'ApproverName'
df_combined = pd.merge(df_historical, df_vacation, on='ApproverName', how='inner')

# Combine relevant columns for text input
df_combined['text'] = df_combined['VacationStartDate'] + ' ' + df_combined['VacationEndDate'] + ' ' + df_combined['ApproverName'] + ' ' + df_combined['SubstituteName']

# Assign labels: 1 if vacation scheduled, 0 otherwise
df_combined['label'] = df_combined['VacationStartDate'].notnull().astype(int)

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)

# Define your tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=2)

# Tokenize the texts
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True)

# Define dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = Dataset(train_encodings, train_df['label'].tolist())
val_dataset = Dataset(val_encodings, val_df['label'].tolist())

# Prepare DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            _, predicted_labels = torch.max(outputs.logits, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}')

# Save the model
model.save_pretrained('vacation_substitution_model')
tokenizer.save_pretrained('vacation_substitution_model')
