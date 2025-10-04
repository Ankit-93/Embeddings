import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# --------------------------
# 1. Define PT terms
# --------------------------
pt_terms = [
    "Headache", "Nausea", "Chest pain", "Rash", "Dizziness",
    "Abdominal pain", "Pyrexia", "Dyspnoea", "Vomiting", "Fatigue", "Fever"
]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
base_encoder = AutoModel.from_pretrained("bert-base-uncased")

# --------------------------
# 2. Get embeddings for PT terms
# --------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [batch, seq, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / attention_mask.sum(1, keepdim=True)

pt_enc = tokenizer(pt_terms, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    pt_model_out = base_encoder(pt_enc["input_ids"], attention_mask=pt_enc["attention_mask"])
    pt_emb = mean_pooling(pt_model_out, pt_enc["attention_mask"])  # shape: [num_terms, hidden_dim]

print("PT embeddings shape:", pt_emb.shape)

# --------------------------
# 3. Dataset for classification fine-tuning
# --------------------------
class PTDataset(Dataset):
    def __init__(self, pt_terms, tokenizer, max_len=10):
        self.pt_terms = pt_terms
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pt_terms)

    def __getitem__(self, idx):
        text = self.pt_terms[idx]
        label = idx  # unique ID for each PT
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}, label

dataset = PTDataset(pt_terms, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# --------------------------
# 4. PT Classifier model
# --------------------------
class PTClassifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled)
        return logits

num_labels = len(pt_terms)
clf_model = PTClassifier(AutoModel.from_pretrained("bert-base-uncased"), num_labels)
optimizer = torch.optim.Adam(clf_model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# --------------------------
# 5. Training loop
# --------------------------
for epoch in range(3):  # keep epochs small for demo
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        logits = clf_model(inputs['input_ids'], inputs['attention_mask'])
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# --------------------------
# 6. Test scenarios (semantic similarity search)
# --------------------------
cos = nn.CosineSimilarity(dim=0)

test_cases = [
    ("Headache", "Headache"),  # Exact match
    ("Pain in the head", "Headache"),  # Synonym
    ("Nausee", "Nausea"),  # Misspelling
    ("Complains of chest discomfort", "Chest pain"),  # Clinical note
    ("Patient feels tired all day", "Fatigue"),  # Same meaning
    ("Patient is not feeling well", None),  # Ambiguous
    ("Fever and vomiting observed", ["Pyrexia", "Vomiting"]),  # Multiple symptoms
    ("No complaints", None),  # Negative case
    ("SOB", "Dyspnoea"),  # Abbreviation
    ("Dolor de cabeza", "Headache")  # Spanish
    ("")
]

for text, expected in test_cases:
    enc = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = base_encoder(enc["input_ids"], attention_mask=enc["attention_mask"])
        emb = mean_pooling(output, enc["attention_mask"]).squeeze(0)

    # Compare against PT embeddings
    similarities = [cos(emb, pt_emb[i]).item() for i in range(len(pt_terms))]
    min_idx = similarities.index(max(similarities))
    predicted_pt = pt_terms[min_idx]

    print(f"Input: {text}")
    print(f"Expected PT: {expected}")
    print(f"Predicted PT: {predicted_pt}")
    print(f"Cosine similarities: {similarities}\n")
