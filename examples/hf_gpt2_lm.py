import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from flops_tracker import FlopsTracker
from flops_tracker.estimators import TorchAutoEstimator
from flops_tracker.hf.adapters import HFDecoderAdapter

class ToyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        self.enc = enc
    def __len__(self): return self.enc["input_ids"].size(0)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}

def collate(records):
    keys = records[0].keys()
    batch = {k: torch.stack([r[k] for r in records], dim=0) for k in keys}
    data = {k: batch[k] for k in ("input_ids","attention_mask","labels")}
    target = batch["labels"]
    return data, target

def main():
    model_name = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = HFDecoderAdapter(hf_model)

    texts = [
        "Who won the Super Bowl?",
        "Explain backpropagation in simple terms.",
        "List three prime numbers.",
        "What is the capital of Italy?"
    ]
    ds = ToyTextDataset(texts, tok, max_len=64)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)

  
    est = TorchAutoEstimator(
        model=model,
        input_example=(  
            torch.zeros(1, 64, dtype=torch.long),
            torch.ones(1, 64, dtype=torch.long),
            torch.zeros(1, 64, dtype=torch.long),
        ),
        include_softmax_cost=True,
        training_factor=3.0,
    )

    ft = (FlopsTracker(est, run_name="gpt2_lm")
          .torch_bind(model, optim, lambda out, y: out.loss, loader, device=device))

    total = ft.run(epochs=2, print_level="epoch", export="epoch", export_prefix="gpt2_lm")
    print("FLOPs totali GPT-2:", total)

if __name__ == "__main__":
    main()
