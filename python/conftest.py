from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.evaluators.doctest import NUMBER
from sybil.parsers.rest import DocTestParser

pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE | NUMBER),
    ],
    patterns=["*.py", "*.pyi"],
).pytest()
