import shutil

from src.chunk import Chunker, Chunk
import pathlib
import json
import re

class Index:
    def __init__(self, max_chunk_size: int):
        self.chunker = Chunker(chunk_size=max_chunk_size)

    def save_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:
        process_dir = pathlib.Path("data/in_progress")
        shutil.rmtree(process_dir, ignore_errors=True)
        process_dir.mkdir(parents=True, exist_ok=True)
        json_file_path = process_dir / "chunks.json"
        if json_file_path.exists():
            json_file_path.unlink()
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(
                [chunk.model_dump() for chunk in chunks],
                f, ensure_ascii=False, indent=4
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

    def index(self, file_path: str = "data/raw"):
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
        self.save_chunks(chunks)

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

    def score_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            chunk.score = self.chunker.score_chunk(chunk)
        return chunks