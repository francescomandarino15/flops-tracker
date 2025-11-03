import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from flops_tracker import FlopsTracker
from flops_tracker.estimators import TorchAutoEstimator
from flops_tracker.hf.adapters import HFEncoderAdapter

class BertClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item

def collate(records):
    import torch
    keys = records[0].keys()
    batch = {k: torch.stack([r[k] for r in records], dim=0) for k in keys}
    data = {k: batch[k] for k in ("input_ids","attention_mask","labels")}
    target = batch["labels"]
    return data, target

def main():
    model_name = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_hf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = HFEncoderAdapter(model_hf)

    texts = ["I liked this movie", "I hated this movie", "Amazing film!", "Terrible acting"]
    labels = [1, 0, 1, 0]
    ds = BertClsDataset(texts, labels, tok, max_len=64)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)

    est = TorchAutoEstimator(
        model=model,
        input_example=(
            torch.zeros(1, 64, dtype=torch.long),  
            torch.ones(1, 64, dtype=torch.long),  
            torch.zeros(1, dtype=torch.long),      
        ),
        include_softmax_cost=True,
        training_factor=3.0,
    )

    ft = (FlopsTracker(est, run_name="bert_cls")
          .torch_bind(model, optim, lambda out, y: out.loss, loader, device=device))

    total = ft.run(epochs=2, print_level="epoch", export="epoch", export_prefix="bert_cls")
    print("FLOPs totali BERT:", total)

if __name__ == "__main__":
    main()
