from src.chunk import Chunk
from src.processed_data_read import ProcessedDataReader
from src.tokenizer import tokenize
from sentence_transformers import SentenceTransformer
import numpy as np


class Search:
    def __init__(self) -> None:
        reader = ProcessedDataReader()
        self.chunks = reader.read_chunks()
        config = reader.read_config()
        self.score_board = reader.read_score_board()
        self.start_points = reader.read_start_points()
        self.embed_score = reader.read_embed_score()
        reader.check_consistency(
            self.chunks, config, self.score_board,
            self.start_points, self.embed_score
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.rrf_k = 60

    @staticmethod
    def rank_by_score(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    def bm25_ranks(self, question: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        for word in set(tokenize(question)):
            for chunk_id, weight in self.score_board.get(word, []):
                scores[chunk_id] += weight
        return self.rank_by_score(scores)

    def vector_ranks(self, question: str) -> np.ndarray:
        question_vector = self.model.encode(
            question, normalize_embeddings=True
        )
        window_scores = self.embed_score @ question_vector
        scores = np.maximum.reduceat(window_scores, self.start_points[:-1])
        return self.rank_by_score(scores)

    def search(self, question: str, k: int) -> list[Chunk]:
        rrf_scores = (
            1 / (self.rrf_k + self.bm25_ranks(question))
            + 1 / (self.rrf_k + self.vector_ranks(question))
        )
        top = np.argsort(-rrf_scores, kind="stable")[:k]
        return [self.chunks[chunk_id] for chunk_id in top]
