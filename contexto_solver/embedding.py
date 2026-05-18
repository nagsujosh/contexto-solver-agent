from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import (
    GLOVE_PATH,
    VOCAB_SIZE,
    EMB_CENTER,
    EMB_REMOVE_TOP_K,
    EMB_FP16,
)


class GloveEmbedding:
    def __init__(self, glove_path: str = GLOVE_PATH, dim: int = 300):
        self.dim = dim
        self.word_to_vec: Dict[str, np.ndarray] = {}
        self._load_glove(glove_path)
        if not self.word_to_vec:
            raise RuntimeError(
                f"No vectors loaded from '{glove_path}'. Ensure the GloVe 6B 300d file exists and is populated. "
                "You can download it from https://nlp.stanford.edu/data/glove.6B.zip and place 'glove.6B.300d.txt' in the project root."
            )

        # Build dense vocab & matrix
        self.vocab = list(self.word_to_vec.keys())
        if VOCAB_SIZE and len(self.vocab) > VOCAB_SIZE:
            self.vocab = self.vocab[:VOCAB_SIZE]
        X = np.vstack([self.word_to_vec[w] for w in self.vocab]).astype(np.float32)

        # Optional isotropy fix: mean-center + remove top PCs
        if EMB_CENTER or EMB_REMOVE_TOP_K > 0:
            X = self._whiten_isotropic(X, center=EMB_CENTER, remove_top_k=EMB_REMOVE_TOP_K)

        # Re-normalize unit length
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        if EMB_FP16:
            X = X.astype(np.float16)

        # Persist back into mapping
        self.word_to_vec.clear()
        self.emb_matrix = X  # (V, D)
        self.vocab_index = {w: i for i, w in enumerate(self.vocab)}

    def _load_glove(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < self.dim + 1:
                    continue
                word = parts[0]
                try:
                    vec = np.asarray(parts[1 : 1 + self.dim], dtype=np.float32)
                except Exception:
                    continue
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n
                self.word_to_vec[word] = vec

    @staticmethod
    def _whiten_isotropic(X: np.ndarray, center: bool, remove_top_k: int) -> np.ndarray:
        Xw = X
        if center:
            mu = Xw.mean(axis=0, keepdims=True)
            Xw = Xw - mu
        if remove_top_k > 0:
            U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
            Vk = Vt[:remove_top_k].T  # (D, k)
            proj = Xw @ Vk @ Vk.T
            Xw = Xw - proj
        return Xw

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        idxs: List[int] = []
        for t in texts:
            key = t.strip().lower()
            i = self.vocab_index.get(key, -1)
            idxs.append(i)

        out = []
        for i in idxs:
            if i < 0:
                out.append(np.zeros(self.emb_matrix.shape[1], dtype=self.emb_matrix.dtype))
            else:
                out.append(self.emb_matrix[i])
        return np.vstack(out)

    def vocab_and_matrix(self) -> Tuple[List[str], np.ndarray]:
        return self.vocab, self.emb_matrix
