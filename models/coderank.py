# benchmark/models/coderank.py

import torch

from ..core.base_model import BaseModel


class CodeRank(BaseModel):

    MODEL_ID   = "nomic-ai/CodeRankEmbed"
    MODEL_NAME = "coderank"
    MAX_LENGTH = 2048

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        from sentence_transformers import SentenceTransformer

        self.tokenizer = None
        self.model     = SentenceTransformer(self.MODEL_ID, trust_remote_code=True)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 32, is_query: bool = False) -> list[torch.Tensor]:
        self.model.max_seq_length = self.MAX_LENGTH

        with torch.inference_mode():
            embeddings = model.encode(codes, convert_to_tensor=True)

        return embedding.detach()