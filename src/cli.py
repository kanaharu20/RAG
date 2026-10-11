from src import index


class Cli:
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = index.Index(max_chunk_size=max_chunk_size)
        indexer.index()

    def search(self, query: str, k: int):
        ...

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        ...

    def answer(self, query: str, k: int):
        ...

    def answer_dataset(
            self, student_search_results_pat: str, 
            save_directory: str
            ):
        ...

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        ...
