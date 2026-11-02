from pathlib import Path
from typing import Annotated

from pydantic import Field, validate_call
from tqdm import tqdm

from src import index
from src.evaluate import Evaluate
from src.models import (
    CheckedDataset, MinimalSearchResults, MinimalSource, NonBlankStr,
    StudentSearchResults
)
from src.search import Search

PositiveInt = Annotated[int, Field(gt=0)]


class Cli:
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = index.Index(max_chunk_size=max_chunk_size)
        indexer.index()

    @validate_call(config={"strict": True})
    def search(self, query: NonBlankStr, k: PositiveInt) -> None:
        searcher = Search()
        for rank, chunk in enumerate(searcher.search(query, k), start=1):
            print(
                f"{rank}. {chunk.file_path} "
                f"[{chunk.first_character_index}:"
                f"{chunk.last_character_index}]"
            )

    @validate_call(config={"strict": True})
    def search_dataset(
        self, dataset_path: str, k: PositiveInt, save_directory: str
    ) -> None:
        dataset = CheckedDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        searcher = Search()
        search_results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            chunks = searcher.search(question.question, k)
            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=chunk.file_path,
                        first_character_index=chunk.first_character_index,
                        last_character_index=chunk.last_character_index
                    )
                    for chunk in chunks
                ]
            ))
        results = StudentSearchResults(search_results=search_results, k=k)
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(dataset_path).name
        output_path.write_text(
            results.model_dump_json(indent=4), encoding="utf-8"
        )
        print(f"Saved student_search_results to {output_path}")

    def answer(self, query: str, k: int):
        ...

    def answer_dataset(
            self, student_search_results_pat: str,
            save_directory: str
            ):
        ...

    @validate_call(config={"strict": True})
    def evaluate(
        self, student_search_results_path: str, dataset_path: str,
        k: PositiveInt = 5
    ) -> None:
        evaluator = Evaluate(student_search_results_path, dataset_path)
        recall = evaluator.recall(k)
        print(
            f"recall@{k}: {recall * 100:.1f}% "
            f"({len(evaluator.dataset.rag_questions)} questions)"
        )
