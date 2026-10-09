from models import MinimalSource
import ast
import pathlib

class Chunk(MinimalSource):
    text: str


class Chunker():
    def __init__(self, chunk_size: int = 2000, file_path: str = ""):
        self.chunk_size = chunk_size
        self.file_path = file_path

    def convert_line_to_character_index(self, text: str) -> list[int]:
        lines = text.splitlines(keepends=True)
        indexes = [0]
        for line in lines:
            indexes.append(indexes[-1] + len(line))
        return indexes

    def chunk_hard_split(self, chunks: list[Chunk]) -> list[Chunk]:
        hard_split_chunks: list[Chunk] = []
        for chunk in chunks:
            if len(chunk.text) > self.chunk_size:
                for i in range(0, len(chunk.text), self.chunk_size):
                    chunk_text = chunk.text[i:i + self.chunk_size]
                    hard_split_chunk = Chunk(
                        file_path=chunk.file_path,
                        first_character_index=chunk.first_character_index + i,
                        last_character_index=chunk.first_character_index + i + len(chunk_text),
                        text=chunk_text
                    )
                    hard_split_chunks.append(hard_split_chunk)
            else:
                hard_split_chunks.append(chunk)
        return hard_split_chunks

    def python_chunker(self, text: str) -> list[Chunk]:
        tree = ast.parse(text)
        chunks: list[Chunk] = []
        indexes = self.convert_line_to_character_index(text)
        start_index = 0
        end_index = 0
        for node in tree.body:
            if isinstance(node, (
                ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef
            )):
                if node.decorator_list:
                    top_decorator = node.decorator_list[0]
                    end_index = indexes[top_decorator.lineno - 1]
                else:
                    end_index = indexes[node.lineno - 1]
                if end_index - start_index == 0:
                    continue
                chunk_text = text[start_index:end_index]
                chunk = Chunk(
                    file_path=self.file_path,
                    first_character_index=start_index,
                    last_character_index=end_index,
                    text=chunk_text
                )
                chunks.append(chunk)
                start_index = end_index
        if start_index < len(text):
            chunk_text = text[start_index:]
            chunk = Chunk(
                file_path=self.file_path,
                first_character_index=start_index,
                last_character_index=len(text),
                text=chunk_text
            )
            chunks.append(chunk)
        hard_split_chunks = self.chunk_hard_split(chunks)
        return hard_split_chunks

    def text_chunker(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunks.append(Chunk(
            file_path=self.file_path,
            first_character_index=0,
            last_character_index=len(text),
            text=text
        ))
        return self.chunk_hard_split(chunks)

    def markdown_chunker(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        in_code_block = False
        lines = text.splitlines(keepends=True)
        start_index = 0
        end_index = 0
        current_line_index = 0
        indexes = self.convert_line_to_character_index(text)
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            current_line_index += 1
            if in_code_block:
                continue
            if line.strip().startswith("#"):
                end_index = indexes[current_line_index - 1]
                if end_index - start_index == 0:
                    continue
                chunk_text = text[start_index:end_index]
                chunk = Chunk(
                    file_path=self.file_path,
                    first_character_index=start_index,
                    last_character_index=end_index,
                    text=chunk_text
                )
                chunks.append(chunk)
                start_index = end_index
        if start_index < len(text):
            chunk_text = text[start_index:]
            chunk = Chunk(
                file_path=self.file_path,
                first_character_index=start_index,
                last_character_index=len(text),
                text=chunk_text
            )
            chunks.append(chunk)
        return self.chunk_hard_split(chunks)

    def chunk(self) -> list[Chunk] | None:
        path = pathlib.Path(self.file_path)
        content = path.read_text(encoding="utf-8", newline="")
        if path.suffix == ".py":
            chunks = self.python_chunker(content)
        elif path.suffix == ".txt":
            chunks = self.text_chunker(content)
        elif path.suffix == ".md":
            chunks = self.markdown_chunker(content)
        else:
            return None
        return chunks
