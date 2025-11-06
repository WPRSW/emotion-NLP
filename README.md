# 🧠 Emotion Detection using BERT  

This project aims to classify English sentences into **six emotional categories** — `joy`, `sadness`, `anger`, `fear`, `love`, and `surprise` — using a fine-tuned **BERT (bert-base-uncased)** model.  
All preprocessing, tokenization, and training steps are implemented from scratch with detailed text cleaning and normalization.  

---

## 📚 Dataset  

Each line in the dataset is in the following format:  
sentence;text_label
For example:
i am feeling happy today;joy
i didnt feel humiliated;sadness
i feel irritated and rejected;anger


The project uses two text files:
- `train.txt` → Training data  
- `test.txt` → Validation data  

---

## 🧩 Emotion Categories  

| Emotion   | Label |
|------------|--------|
| joy        | 0 |
| sadness    | 1 |
| anger      | 2 |
| fear       | 3 |
| love       | 4 |
| surprise   | 5 |

---

## ⚙️ Preprocessing Pipeline  

The text cleaning and preprocessing steps applied before training include:

1. **Lowercasing all text**  
2. **Removing punctuation and symbols**  
3. **Tokenization** using NLTK  
4. **Removing short/meaningless tokens** (e.g. one-letter words)  
5. **Spell correction** using `pyspellchecker`  
6. **Handling contractions** (e.g. `"don't"` → `"do not"`)  
7. **Removing empty tokens or `None` values**  
8. **POS tagging** using NLTK  
9. **POS normalization** (`N → n`, `V → v`, `J → a`, `R → r`)  
10. **Lemmatization** using `WordNetLemmatizer`  
11. **Reconstruction** of the cleaned sentence from tokens  

All steps are modularized as functions (e.g. `remove_sym`, `lemmatizing`, `fix_tag`, etc.).

---

## 🤖 Model Architecture  

The project uses **BERT** (`bert-base-uncased`) from the Hugging Face Transformers library for sequence classification.

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
```
The model is trained with:

Loss: CrossEntropyLoss

Optimizer: AdamW (lr=2e-5)

Scheduler: Linear Warmup Scheduler

Batch Size: 16

Epochs: 3

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```
Loss is printed at the end of each epoch.
After training, the model is evaluated on the test set using accuracy:
```python
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(input_ids_test, attention_mask=attention_mask_test)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels_test).sum().item()
        total += labels_test.size(0)
accuracy = correct / total
print("Validation Accuracy:", accuracy)
```
Validation Accuracy: 0.92

After full preprocessing and fine-tuning, the model achieves an average accuracy of 90–95% on the test dataset depending on random seed and preprocessing.
# Developed by Mohammad Parsa Torabi
