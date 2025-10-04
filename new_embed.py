import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models

#pip install datasets

# -------------------------------
# 1. Example SMQ → PT mappings
# -------------------------------
smq_to_pts = {
    "Fever and vomiting observed": ["Pyrexia", "Vomiting"],
    "Cough and cold symptoms": ["Cough", "Cold", "Sneezing"],
    "Patient reports chest discomfort": ["Chest pain", "Angina"],
    "Head spinning and loss of balance": ["Dizziness", "Vertigo"],
}

# Collect all PT terms for negative sampling
all_pts = list({pt for pts in smq_to_pts.values() for pt in pts})

# -------------------------------
# 2. Build Training Examples
# -------------------------------
train_examples = []

for smq, pos_pts in smq_to_pts.items():
    for pt in pos_pts:
        # Positive pair
        train_examples.append(InputExample(texts=[smq, pt], label=1.0))

    # Add negative pairs by sampling unrelated PTs
    neg_pts = random.sample([p for p in all_pts if p not in pos_pts], k=2)
    for pt in neg_pts:
        train_examples.append(InputExample(texts=[smq, pt], label=0.0))

print(f"Total training pairs: {len(train_examples)}")

# -------------------------------
# 3. Define Transformer Model
# -------------------------------
# Use BioBERT for medical text OR "bert-base-uncased" for generic
word_embedding_model = models.Transformer("bert-base-uncased")
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# -------------------------------
# 4. Training Setup
# -------------------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    show_progress_bar=True
)

# -------------------------------
# 5. Save Trained Model
# -------------------------------
output_path = "output/medical-embedding-model"
model.save(output_path)
print(f"Model saved at: {output_path}")

# -------------------------------
# 6. Test Inference
# -------------------------------
new_model = SentenceTransformer(output_path)

query = "Fever and vomiting observed"
candidate_pts = ["Pyrexia", "Vomiting", "Abdominal pain", "Headache"]

emb1 = new_model.encode(query, convert_to_tensor=True)
emb2 = new_model.encode(candidate_pts, convert_to_tensor=True)

cosine_scores = torch.nn.functional.cosine_similarity(emb1, emb2)

print("\nQuery:", query)
for pt, score in zip(candidate_pts, cosine_scores):
    print(f"  {pt:20s} → {score.item():.4f}")
