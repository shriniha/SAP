import pandas as pd
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('auto_approval_dataset.csv')

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Description'], df['AutoApprove'], test_size=0.2, random_state=42)

# Load the pre-trained ELECTRA model and tokenizer
# Load the pre-trained ELECTRA model and tokenizer
model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=2)
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


# Tokenize the texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# Convert the labels to torch.long
train_labels = torch.tensor(train_labels.tolist(), dtype=torch.long)
val_labels = torch.tensor(val_labels.tolist(), dtype=torch.long)

# Convert the tokenized texts into torch tensors
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    train_labels
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    val_labels
)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Create a DataLoader for training and validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

# Train the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
for epoch in range(3):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds = []
    val_true = []
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()
        val_preds.extend(preds)
        val_true.extend(labels.tolist())

    val_accuracy = accuracy_score(val_true, val_preds)
    print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}')

# Save the trained model
model.save_pretrained('auto_approval_model')
tokenizer.save_pretrained('auto_approval_model')
