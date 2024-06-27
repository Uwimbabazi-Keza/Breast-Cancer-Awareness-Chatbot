import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import re
import torch.nn as nn

# Load and preprocess data
file_path = 'Breast_Cancer_Awareness_Chatbot.csv'
df = pd.read_csv(file_path)
df.dropna(inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['Questions'] = df['Questions'].apply(clean_text)
df['Answers'] = df['Answers'].apply(clean_text)
df['Patterns'] = df['Patterns'].apply(clean_text)
df['Tags'] = df['Tags'].apply(clean_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []

for question in df['Questions'].tolist():
    encoded_dict = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

df['Combined'] = df['Answers'] + ' [SEP] ' + df['Tags']
label_map = {combined: i for i, combined in enumerate(df['Combined'].unique())}
labels = [label_map[combined] for combined in df['Combined']]
labels = torch.tensor(labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
model.to(device)

batch_size = 16
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 50
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        model.zero_grad()

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        loss = loss_fn(outputs.logits, b_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

print("Training complete.")

model_save_path = 'bert_breast_cancer_model.pth'
torch.save(model.state_dict(), model_save_path)
print("Model saved.")