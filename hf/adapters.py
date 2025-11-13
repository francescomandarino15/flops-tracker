import torch
import torch.nn as nn

def _ensure_long(x):
    # labels per i modelli HF di solito devono essere Long
    return x if (x is None or x.dtype == torch.long) else x.long()

class HFDecoderAdapter(nn.Module):
    """
    Adapter per AutoModelForCausalLM (GPT-2 & co.) in modo che il FlopsTracker
    possa chiamare model(data) dove `data` può essere:
      - dict con chiavi: input_ids[, attention_mask][, labels]
      - tuple/list: (input_ids, attention_mask[, labels])
      - tensore singolo: input_ids
    Ritorna ModelOutput HF con .loss (se labels presenti) e .logits.
    """
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf = hf_model

    def forward(self, batch):
        # Caso 1: dict (forma nativa per HF)
        if isinstance(batch, dict):
            kw = dict(batch)
            if "labels" in kw and kw["labels"] is not None:
                kw["labels"] = _ensure_long(kw["labels"])
            return self.hf(**kw)

        # Caso 2: sequenza (tuple/list) -> mappo in kwargs
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                ids, mask, labels = batch
                return self.hf(input_ids=ids, attention_mask=mask, labels=_ensure_long(labels))
            if len(batch) == 2:
                ids, mask = batch
                return self.hf(input_ids=ids, attention_mask=mask)
            raise ValueError("HFDecoderAdapter: attesa tuple/list con 2 o 3 elementi.")

        # Caso 3: tensore singolo = solo input_ids
        if torch.is_tensor(batch):
            return self.hf(input_ids=batch)

        raise ValueError("HFDecoderAdapter: batch deve essere dict, (ids,mask[,labels]) oppure Tensor (input_ids).")


class HFEncoderAdapter(nn.Module):
    """
    Adapter per AutoModel / AutoModelForSequenceClassification (BERT & co.).
    Stessa logica del decoder.
    """
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf = hf_model

    def forward(self, batch):
        if isinstance(batch, dict):
            kw = dict(batch)
            if "labels" in kw and kw["labels"] is not None:
                kw["labels"] = _ensure_long(kw["labels"])
            return self.hf(**kw)

        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                ids, mask, labels = batch
                return self.hf(input_ids=ids, attention_mask=mask, labels=_ensure_long(labels))
            if len(batch) == 2:
                ids, mask = batch
                return self.hf(input_ids=ids, attention_mask=mask)
            raise ValueError("HFEncoderAdapter: attesa tuple/list con 2 o 3 elementi.")

        if torch.is_tensor(batch):
            return self.hf(input_ids=batch)

        raise ValueError("HFEncoderAdapter: batch deve essere dict, (ids,mask[,labels]) oppure Tensor (input_ids).")
