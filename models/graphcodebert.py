# benchmark/models/graphcodebert.py

import torch

from transformers import AutoTokenizer, AutoModel

from ..core.base_model import BaseModel

class GraphCodeBERT(BaseModel):
    MODEL_ID = "microsoft/graphcodebert-base"
    MODEL_NAME = "graphcodebert"
    MAX_LENGTH = 512

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model     = AutoModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]
    
    def encode_batch(self, codes: list[str], batch_size: int = 32, is_query: bool = False) -> list[torch.Tensor]:
        embeddings = []

        for i in range(0, len(codes), batch_size):
            batch  = codes[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
            
            batch_embeddings = self._mean_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            embeddings.extend(batch_embeddings.cpu())

        return embeddings
    
    def _mean_pool(self, last_hidden_state: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

"""

def create_embeddings_graph(code_snippet,tokenizer,model,device,max_length):
    # Tokenize the code
    inputs = tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.inference_mode():
        output = model(**inputs)

    embedding = average_pool(output.last_hidden_state, attention_mask=inputs["attention_mask"])
    return embedding.detach(), time_end-time_init

case "graphcodebert":
            return create_embeddings_graph(source_code,tokenizer,model,device,512)

model_name == "graphcodebert":
        model_id = "microsoft/graphcodebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        return (model,tokenizer)

def create_embeddings_graph(code_snippet,tokenizer,model,device,max_length):
    # Tokenize the code
    inputs = tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.inference_mode():
        output = model(**inputs)

    embedding = average_pool(output.last_hidden_state, attention_mask=inputs["attention_mask"])
    return embedding.detach(), time_end-time_init

"""