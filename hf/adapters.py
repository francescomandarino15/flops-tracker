import torch
import torch.nn as nn

class HFDecoderAdapter(nn.Module):
    """
    Adatta un AutoModelForCausalLM (GPT-2, ecc.) al FlopsTracker.
    Accetta:
      - dict con chiavi ("input_ids","attention_mask","labels") per training;
      - tuple (input_ids, attention_mask, labels) per dry-run dell'estimatore.
    Ritorna il ModelOutput HF con .loss (se labels presenti) e .logits.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, batch):
        if isinstance(batch, dict):
            return self.hf(**batch)
        # altrimenti assumo tupla ordinata
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
            labels = batch[2] if len(batch) > 2 else None
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None: kwargs["attention_mask"] = attention_mask
            if labels is not None: kwargs["labels"] = labels
            return self.hf(**kwargs)
        raise ValueError("HFDecoderAdapter: batch deve essere dict o (input_ids, attention_mask, [labels])")

class HFEncoderAdapter(nn.Module):
    """
    Adatta un AutoModelForSequenceClassification (BERT, ecc.) al FlopsTracker.
    Stessa logica del decoder adapter.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, batch):
        if isinstance(batch, dict):
            return self.hf(**batch)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
            labels = batch[2] if len(batch) > 2 else None
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None: kwargs["attention_mask"] = attention_mask
            if labels is not None: kwargs["labels"] = labels
            return self.hf(**kwargs)
        raise ValueError("HFEncoderAdapter: batch deve essere dict o (input_ids, attention_mask, [labels])")
