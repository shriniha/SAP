import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset
df = pd.read_csv('predictive_routing_dataset.csv')

# Encode categorical variables
label_encoders = {}
for column in ['Category', 'Department', 'Urgency', 'Complexity', 'ApproverName']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define a custom dataset class
class ExpenseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

# Combine text features into a single string for BERT
df['text'] = df['Category'].astype(str) + ' ' + df['Department'].astype(str) + ' ' + df['Urgency'].astype(str) + ' ' + df['Complexity'].astype(str)

X = df['text'].values
y = df['ApproverName'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = ExpenseDataset(X_train, y_train, tokenizer, max_len)
val_dataset = ExpenseDataset(X_val, y_val, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoders['ApproverName'].classes_))
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train_epoch(model, data_loader, criterion, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        loss = criterion(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, criterion, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            loss = criterion(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# Training the model
epochs = 3

for epoch in range(epochs):
    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        None,
        len(X_train)
    )

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_loader,
        criterion,
        device,
        len(X_val)
    )

    print(f'Validation loss {val_loss} accuracy {val_acc}')

# Save the trained model and tokenizer
model.save_pretrained('predictive_routing_bert_model')
tokenizer.save_pretrained('predictive_routing_bert_model')
y_val_preds = []
y_val_true = []

model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        y_val_preds.extend(preds.cpu().numpy())
        y_val_true.extend(labels.cpu().numpy())

val_accuracy = accuracy_score(y_val_true, y_val_preds)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(classification_report(y_val_true, y_val_preds, target_names=label_encoders['ApproverName'].classes_))
