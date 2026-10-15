import shutil

from src.chunk import Chunker, Chunk
import pathlib
import json
import re
import collections
import math
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel


class Config(BaseModel):
    chunk_num: int
    vector_num: int
    file_num: int
    max_chunk_size: int
    sub_py: int
    sub_md_txt: int

    @classmethod
    def write_config_file(
        cls, chunks: list[Chunk],
        embedding_score: np.ndarray, file_num: int,
        max_chunk_size: int
            ) -> "Config":
        chunk_num = len(chunks)
        vector_num = embedding_score.shape[0]
        return cls(
            chunk_num=chunk_num, vector_num=vector_num,
            file_num=file_num, max_chunk_size=max_chunk_size,
            sub_py=1000, sub_md_txt=500
            )


class Index:
    def __init__(self, max_chunk_size: int):
        self.chunker = Chunker(chunk_size=max_chunk_size)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def split_chunks(self, chunks: list[Chunk]) -> list[list[str]]:
        split_chunks = []
        for chunk in chunks:
            split_content = re.findall(
                r"[a-zA-Z0-9_]+", chunk.text
            )
            split_chunks.append(split_content)
        return split_chunks

    def trim_chunks(self, words: list[list[str]]) -> list[list[str]]:
        trimmed_chunks: list[list[str]] = []
        for word_list in words:
            keep_words: list[str] = []
            for word in word_list:
                parts = re.findall(
                    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", word)
                keep_words.extend(parts)
                if len(parts) > 1:
                    keep_words.append(word)
            trimmed_chunks.append(keep_words)
        return trimmed_chunks

    def lower_case_chunks(self, words: list[list[str]]) -> list[list[str]]:
        lower_case_chunks = []
        for word_list in words:
            lower_case_list = [word.lower() for word in word_list]
            lower_case_chunks.append(lower_case_list)
        return lower_case_chunks

    def count_words_in_chunks(
            self, arranged_chunks: list[list[str]]
    ) -> list[dict[str, int]]:
        word_counts: list[dict[str, int]] = []
        for word_list in arranged_chunks:
            counter = collections.Counter(word_list)
            word_counts.append(dict(counter))
        return word_counts

    def calculate_dl(self, arranged_chunks: list[list[str]]) -> list[int]:
        dl_list: list[int] = []
        for chunk in arranged_chunks:
            dl = len(chunk)
            dl_list.append(dl)
        return dl_list

    def calculate_N(self, chunks: list[Chunk]) -> int:
        return len(chunks)

    def calculate_df(self, word_counts: list[dict[str, int]]) -> dict[str, int]:
        df_dict: dict[str, int] = {}
        for word_count in word_counts:
            for word in word_count.keys():
                if word in df_dict:
                    df_dict[word] += 1
                else:
                    df_dict[word] = 1
        return df_dict

    def calculate_idf(self, df_dict: dict[str, int], N: int) -> dict[str, float]:
        idf_dict: dict[str, float] = {}
        for word, df in df_dict.items():
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            idf_dict[word] = idf
        return idf_dict

    def calculate_avg_dl(self, dl_list: list[int]) -> float:
        if not dl_list:
            return 0.0
        return sum(dl_list) / len(dl_list)

    def calculate_bm25(
            self, word_counts: list[dict[str, int]], idf_dict: dict[str, float],
            dl_list: list[int], avg_dl: float, k1: float = 1.2, b: float = 0.75
    ) -> list[dict[str, float]]:
        bm25_scores: list[dict[str, float]] = []
        for i, word_count in enumerate(word_counts):
            dl = dl_list[i]
            scores: dict[str, float] = {}
            for word, tf in word_count.items():
                idf = idf_dict.get(word, 0)
                score = idf * (
                    tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl)
                                      )
                scores[word] = score
            bm25_scores.append(scores)
        return bm25_scores

    def chunks_scores_by_word(
            self, bm25_scores: list[dict[str, float]]
    ) -> dict[str, list[tuple[int, float]]]:
        word_to_chunk_scores: dict[str, list[tuple[int, float]]] = {}
        for chunk_index, score_dict in enumerate(bm25_scores):
            for word, score in score_dict.items():
                if word not in word_to_chunk_scores:
                    word_to_chunk_scores[word] = []
                word_to_chunk_scores[word].append(
                    (chunk_index, round(score, 6))
                )
        return word_to_chunk_scores

    def make_score_board(self, chunks: list[Chunk]) -> dict[str, list[tuple[int, float]]]:
        arranged_chunks = self.split_chunks(chunks)
        trimmed_chunks = self.trim_chunks(arranged_chunks)
        lower_case_chunks = self.lower_case_chunks(trimmed_chunks)
        word_counts = self.count_words_in_chunks(lower_case_chunks)
        dl_list = self.calculate_dl(lower_case_chunks)
        N = self.calculate_N(chunks)
        df_dict = self.calculate_df(word_counts)
        idf_dict = self.calculate_idf(df_dict, N)
        avg_dl = self.calculate_avg_dl(dl_list)
        bm25_scores = self.calculate_bm25(
            word_counts, idf_dict, dl_list, avg_dl
        )
        score_board = self.chunks_scores_by_word(bm25_scores)
        return score_board

    @staticmethod
    def open_small_window(chunks: list[Chunk]) -> list[list[str]]:
        small_window_list: list[list[str]] = []
        for chunk in chunks:
            small_window: list[str] = []
            if pathlib.Path(chunk.file_path).suffix == ".py":
                window_size: int = 1000
            else:
                window_size = 500
            if len(chunk.text) > window_size:
                for i in range(0, len(chunk.text), window_size):
                    chunk_text = chunk.file_path + "\n" + chunk.text[
                        i:i + window_size
                        ]
                    small_window.append(chunk_text)
            else:
                small_window.append(chunk.file_path + "\n" + chunk.text)
            small_window_list.append(small_window)
        return small_window_list

    @staticmethod
    def make_startpoint_list(small_window_list: list[list[str]]) -> list[int]:
        start_points: list[int] = []
        current_point: int = 0
        start_points.append(0)
        for small_window in small_window_list:
            current_point += len(small_window)
            start_points.append(current_point)
        return start_points

    def embed(self, small_window_list: list[list[str]]) -> np.ndarray:
        windows: list[str] = []
        for wins in small_window_list:
            for win in wins:
                windows.append(win)
        V = self.model.encode(
            windows, normalize_embeddings=True, show_progress_bar=True
            )
        return V

    @staticmethod
    def save_files(
        chunks: list[Chunk],
        score_board: dict[str, list[tuple[int, float]]],
        start_points: list[int],
        embedding_score: np.ndarray,
        config: "Config"
    ) -> None:
        process_dir = pathlib.Path("data/in_progress")
        shutil.rmtree(process_dir, ignore_errors=True)
        process_dir.mkdir(parents=True, exist_ok=True)
        chunks_file_path = process_dir / "chunks.json"
        score_board_file_path = process_dir / "score_board.json"
        start_points_file_path = process_dir / "start_points.npy"
        embedding_score_file_path = process_dir / "embed_score.npy"
        config_file_path = process_dir / "configs.json"
        with open(chunks_file_path, "w", encoding="utf-8") as f:
            json.dump(
                [chunk.model_dump() for chunk in chunks],
                f, ensure_ascii=False, indent=4
            )
        with open(score_board_file_path, "w", encoding="utf-8") as f:
            json.dump(
                score_board,
                f, ensure_ascii=False
            )
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(
                config.model_dump(),
                f, ensure_ascii=False,
            )
        np_start_points = np.array(start_points)
        np.save(start_points_file_path, np_start_points)
        np.save(embedding_score_file_path, embedding_score)
        save_dir = pathlib.Path("data/processed")
        buckup_dir = pathlib.Path("data/backup")
        if save_dir.exists():
            if buckup_dir.exists():
                shutil.rmtree(buckup_dir)
            save_dir.replace(buckup_dir)
        process_dir.replace(save_dir)
        if buckup_dir.exists():
            shutil.rmtree(buckup_dir)

    def index(self, file_path: str = "data/raw") -> None:
        chunks: list[Chunk] = []
        root_path = pathlib.Path(file_path)
        all_file_paths = list(root_path.rglob("*"))
        sorted_file_paths = sorted(all_file_paths, key=lambda p: str(p))
        for path in sorted_file_paths:
            if path.is_file() and path.suffix in [
                ".py", ".txt", ".md"
            ]:
                keep = self.chunker.chunk(str(path))
                if keep is not None:
                    chunks.extend(keep)
        score_board = self.make_score_board(chunks)
        small_window_list = self.open_small_window(chunks)
        start_points = self.make_startpoint_list(small_window_list)
        embedding_score = self.embed(small_window_list)
        file_num = len({chunk.file_path for chunk in chunks})
        config = Config.write_config_file(
            chunks, embedding_score, file_num, self.chunker.chunk_size
            )

        self.save_files(
            chunks, score_board, start_points, embedding_score, config
            )
