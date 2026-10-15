from src.chunk import Chunk
import json
import pathlib
from src.index import Config
import numpy as np


class ProcessedDataReader:
    def __init__(self) -> None:
        self.processed_dir = pathlib.Path("data/processed")
        self.backup_dir = pathlib.Path("data/backup")
        self.current_dir: pathlib.Path
        if self.processed_dir.exists():
            self.current_dir = self.processed_dir
        elif self.backup_dir.exists():
            self.current_dir = self.backup_dir
        else:
            raise IndexError  # これは仮置きです

    def read_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        path = self.current_dir / "chunks.json"
        with open(path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            for data in data_list:
                chunks.append(Chunk.model_validate(data))
        return chunks

    def read_config(self) -> Config:
        path = self.current_dir / "configs.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            config = Config.model_validate(data)
        return config

    def read_score_board(self) -> dict[str, list[tuple[int, float]]]:
        path = self.current_dir / "score_board.json"
        score_board: dict[str, list[tuple[int, float]]] = {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for word, pairs in data.items():
                score_board[word] = [
                    (int(chunk_id), float(score)) for chunk_id, score in pairs
                ]
        return score_board

    def read_embed_score(self) -> np.ndarray:
        path = self.current_dir / "embed_score.npy"
        score = np.load(path)
        return score

    def read_start_points(self) -> np.ndarray:
        path = self.current_dir / "start_points.npy"
        points = np.load(path)
        return points

    @staticmethod
    def check_consistency(
        chunks: list[Chunk],
        config: Config,
        score_board: dict[str, list[tuple[int, float]]],
        start_points: np.ndarray,
        embed_score: np.ndarray
    ) -> None:
        chunk_num = len(chunks)
        vector_num = embed_score.shape[0]
        file_num = len({chunk.file_path for chunk in chunks})
        max_chunk_id = max(
            (chunk_id for pairs in score_board.values()
             for chunk_id, _ in pairs),
            default=-1
        )
        problems: list[str] = []
        if config.chunk_num != chunk_num:
            problems.append(
                f"chunk_num in configs.json is {config.chunk_num}, "
                f"but chunks.json has {chunk_num} chunks"
            )
        if config.vector_num != vector_num:
            problems.append(
                f"vector_num in configs.json is {config.vector_num}, "
                f"but embed_score.npy has {vector_num} rows"
            )
        if config.file_num != file_num:
            problems.append(
                f"file_num in configs.json is {config.file_num}, "
                f"but chunks.json has {file_num} files"
            )
        if len(start_points) != chunk_num + 1:
            problems.append(
                f"start_points.npy has {len(start_points)} entries, "
                f"expected {chunk_num + 1}"
            )
        elif start_points[0] != 0 or start_points[-1] != vector_num:
            problems.append(
                f"start_points.npy runs from {start_points[0]} "
                f"to {start_points[-1]}, expected 0 to {vector_num}"
            )
        if max_chunk_id >= chunk_num:
            problems.append(
                f"score_board.json refers to chunk {max_chunk_id}, "
                f"but chunks.json has only {chunk_num} chunks"
            )
        if problems:
            raise ValueError(  # これは仮置きです
                "index files do not match "
                f"(built with max_chunk_size={config.max_chunk_size}): "
                + "; ".join(problems)
                + ". Run the index command again."
            )
