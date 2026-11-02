from collections import Counter
from pathlib import Path

from src.error import InvalidResultsError
from src.models import AnsweredDataset, MinimalSource, StudentSearchResults


class Evaluate:
    def __init__(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        self.max_context_length = 2000
        self.iou_threshold = 0.05
        self.results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        self.dataset = AnsweredDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )

    def find_problems(self) -> list[str]:
        problems: list[str] = []
        result_ids = [
            result.question_id for result in self.results.search_results
        ]
        dataset_ids = {
            question.question_id for question in self.dataset.rag_questions
        }
        for question_id, count in Counter(result_ids).items():
            if count > 1:
                problems.append(
                    f"question {question_id} appears {count} times"
                )
        for question_id in result_ids:
            if question_id not in dataset_ids:
                problems.append(
                    f"question {question_id} is not in the dataset"
                )
        for question_id in sorted(dataset_ids - set(result_ids)):
            problems.append(
                f"question {question_id} is missing from the results"
            )
        for result in self.results.search_results:
            if len(result.retrieved_sources) != self.results.k:
                problems.append(
                    f"question {result.question_id} has "
                    f"{len(result.retrieved_sources)} retrieved sources, "
                    f"expected {self.results.k}"
                )
            for source in result.retrieved_sources:
                width = (
                    source.last_character_index
                    - source.first_character_index
                )
                if width <= 0 or width > self.max_context_length:
                    problems.append(
                        f"question {result.question_id} has a source "
                        f"{source.file_path} "
                        f"[{source.first_character_index}:"
                        f"{source.last_character_index}] "
                        f"of width {width}"
                    )
        return problems

    def check_valid(self) -> None:
        problems = self.find_problems()
        if problems:
            shown = "; ".join(problems[:5])
            if len(problems) > 5:
                shown += f"; and {len(problems) - 5} more"
            raise InvalidResultsError(
                f"Invalid search results ({len(problems)} problems): {shown}"
            )

    def iou(self, retrieved: MinimalSource, correct: MinimalSource) -> float:
        if retrieved.file_path != correct.file_path:
            return 0.0
        overlap = max(
            0,
            min(retrieved.last_character_index,
                correct.last_character_index)
            - max(retrieved.first_character_index,
                  correct.first_character_index)
        )
        retrieved_width = (
            retrieved.last_character_index - retrieved.first_character_index
        )
        correct_width = (
            correct.last_character_index - correct.first_character_index
        )
        union = retrieved_width + correct_width - overlap
        return overlap / union

    def recall(self, k: int) -> float:
        if k > self.results.k:
            raise ValueError(
                f"k={k} is larger than k={self.results.k} "
                "in the search results"
            )
        self.check_valid()
        if not self.dataset.rag_questions:
            raise InvalidResultsError("The dataset has no questions")
        retrieved_by_id = {
            result.question_id: result.retrieved_sources[:k]
            for result in self.results.search_results
        }
        total = 0.0
        for question in self.dataset.rag_questions:
            retrieved = retrieved_by_id[question.question_id]
            found = sum(
                any(
                    self.iou(source, correct) >= self.iou_threshold
                    for source in retrieved
                )
                for correct in question.sources
            )
            total += found / len(question.sources)
        return total / len(self.dataset.rag_questions)
