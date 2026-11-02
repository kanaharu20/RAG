class RagError(Exception):
    pass


class IndexNotFoundError(RagError):
    pass


class IndexMismatchError(RagError):
    pass
