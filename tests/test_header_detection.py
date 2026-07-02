"""Header detection must survive the csv.Sniffer's known blind spots.

The Sniffer votes by column type/width, so short header names over
uniform-width string columns ("a,b" / "nome,valore") get misread as data —
which silently renames every column to column_1, column_2, ...
"""

from auto_cleaner.ingest.detect import detect_header


def test_short_names_over_string_column_is_a_header():
    assert detect_header("a,b\n1.0,x\n2.0,y\n", ",") is True


def test_word_names_over_string_and_numeric_is_a_header():
    assert detect_header("nome,valore\nanna,10\nluca,20\n", ",") is True


def test_headerless_numeric_data_is_not_a_header():
    assert detect_header("1,2\n3,4\n5,6\n", ",") is False
