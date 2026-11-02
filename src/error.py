class RagError(Exception):
    pass


class IndexNotFoundError(RagError):
    pass


class IndexMismatchError(RagError):
    pass


class InvalidResultsError(RagError):
    pass
