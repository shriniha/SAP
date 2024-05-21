import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
import pandas as pd

# Define dataset class
class SubstitutionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.loc[idx, 'ApproverName']
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return inputs

# Load historical substitution dataset
df_historical = pd.read_csv('historical_substitutions.csv')

# Load Electra tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=len(df_historical['SubstituteName'].unique()))

# Define dataset and dataloader
dataset = SubstitutionDataset(df_historical, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        labels = torch.tensor([df_historical[df_historical['ApproverName'] == text]['SubstituteName'].iloc[0] for text in batch['input_ids']]).to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')

# Save the model and tokenizer
model.save_pretrained('substitute_name_model')
tokenizer.save_pretrained('substitute_name_model')
