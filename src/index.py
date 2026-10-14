import shutil

from src.chunk import Chunker, Chunk
import pathlib
import json
import re
import collections
import math


class Index:
    def __init__(self, max_chunk_size: int):
        self.chunker = Chunker(chunk_size=max_chunk_size)

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

    def save_files(
        self,
        chunks: list[Chunk],
        score_board: dict[str, list[tuple[int, float]]]
    ) -> None:
        process_dir = pathlib.Path("data/in_progress")
        shutil.rmtree(process_dir, ignore_errors=True)
        process_dir.mkdir(parents=True, exist_ok=True)
        chunks_file_path = process_dir / "chunks.json"
        score_board_file_path = process_dir / "score_board.json"
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
        self.save_files(chunks, score_board)
