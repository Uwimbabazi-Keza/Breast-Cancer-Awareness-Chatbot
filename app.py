from flask import Flask, render_template, request, session
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import re
import torch.nn as nn

app = Flask(__name__)
app.secret_key = 'your_secret_key'

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

df['Combined'] = df['Answers'] + ' [SEP] ' + df['Tags']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './bert_breast_cancer_model.pth'

num_labels = len(df['Combined'].unique())

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.route('/')
def home():
    session['conversation'] = []
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_question = request.form['user_question']

    encoded_dict = tokenizer.encode_plus(
        user_question,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    predicted_answer = df[df['Combined'] == df['Combined'].unique()[predicted_label]]['Answers'].values[0]

    conversation = session.get('conversation', [])
    conversation.append({'question': user_question, 'answer': predicted_answer})
    session['conversation'] = conversation

    return render_template('index.html', conversation=conversation)

if __name__ == '__main__':
    app.run(debug=True)
