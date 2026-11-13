"""Regression tests using real decoders without accessing DNA files."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open

import pytest

from converter import encode_fibonacci


@pytest.fixture
def decoder():
    """Execute a fresh module with only its file I/O replaced."""
    path = Path(__file__).resolve().parents[1] / "decompressor.py"
    spec = importlib.util.spec_from_file_location("_decompressor_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.open = mock_open(read_data="")
    spec.loader.exec_module(module)
    # Import closes the initial truncation handle; ignore those setup calls.
    module.output_file.reset_mock()
    return module


def test_parse_number(decoder):
    decoder.input_file = "0011"
    assert decoder.parse_number(0) == (3, 4)


def test_parse_number_position_binary(decoder):
    decoder.input_file = "010"
    assert decoder.parse_number_position(0, 8) == (2, 3)


def test_parse_bases_all_encodings(decoder):
    decoder.input_file = "11100100"
    assert decoder.parse_bases(4, 0) == list("ACTG")


def test_parse_factors(decoder):
    decoder.input_file = "0011001"
    draft = list("ACTG")
    assert decoder.parse_factors(1, 0, draft) == ([(3, "0", 0)], 7)
    assert draft == list("ACTG")


def test_decode_factors_direct_copy(decoder):
    draft = list("ACTG")
    result = decoder.decode_factors([(4, "0", 0)], draft)
    assert result is draft
    assert result == list("ACTGACTG")


def test_main_literal_only(decoder):
    decoder.input_file = "1011" + "11100100"
    decoder.main()
    decoder.output_file.write.assert_called_once_with("ACTG")
    decoder.output_file.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("bits", "expected"),
    [("11", 1), ("011", 2), ("1011", 4), ("000011", 8)],
)
def test_parse_number_at_offset_stops_at_first_terminator(decoder, bits, expected):
    decoder.input_file = "11100" + bits + "11"
    assert decoder.parse_number(5) == (expected, len(bits))


@pytest.mark.parametrize("number", [5, 13, 144, 10946])
def test_parse_number_large_fibonacci_round_trip(decoder, number):
    bits = encode_fibonacci(number)
    decoder.input_file = bits
    assert decoder.parse_number(0) == (number, len(bits))


@pytest.mark.parametrize(
    ("bits", "window", "expected"),
    [
        pytest.param("1", 2, 1, id="one-bit-binary"),
        pytest.param("101", 5, 5, id="non-power-of-two-window"),
        pytest.param("111", 4, 3, id="escaped-binary-at-start"),
        pytest.param("0111", 8, 3, id="escaped-binary-at-end"),
        pytest.param("11101", 16, 13, id="escaped-binary-with-suffix"),
        pytest.param("110", 8, 1, id="fibonacci-one"),
        pytest.param("0110", 16, 2, id="fibonacci-two"),
        pytest.param("00110", 32, 3, id="fibonacci-three"),
    ],
)
def test_parse_number_position_encodings_at_offset(decoder, bits, window, expected):
    decoder.input_file = "00" + bits + "101"
    assert decoder.parse_number_position(2, window) == (expected, len(bits))


def test_parse_bases_offset_and_count_exclude_surrounding_bits(decoder):
    decoder.input_file = "0" + "1001" + "11"
    assert decoder.parse_bases(2, 1) == list("CT")


@pytest.mark.parametrize(("bits", "position"), [("", 0), ("1110", 2)])
def test_parse_bases_zero_count(decoder, bits, position):
    decoder.input_file = bits
    assert decoder.parse_bases(0, position) == []


def test_parse_bases_short_even_chunk_returns_available_bases(decoder):
    # Existing slicing behavior does not reject a short, complete-pair chunk.
    decoder.input_file = "1110"
    assert decoder.parse_bases(3, 0) == list("AC")


def test_parse_bases_incomplete_pair_raises_key_error(decoder):
    decoder.input_file = "111"
    with pytest.raises(KeyError) as error:
        decoder.parse_bases(2, 0)
    assert error.value.args == ("1",)


def test_parse_factors_zero_count_preserves_position_and_draft(decoder):
    decoder.input_file = ""
    draft = []
    assert decoder.parse_factors(0, 7, draft) == ([], 7)
    assert draft == []


def test_parse_factors_grows_window_across_binary_width_boundary(decoder):
    # First copy grows window 4 -> 5, so the next position needs three bits.
    first = "11" + "0" + "01"
    second = "011" + "1" + "100"
    decoder.input_file = "00" + first + second + "11"
    draft = list("ACTG")
    factors, end = decoder.parse_factors(2, 2, draft)
    assert factors == [(1, "0", 0), (2, "1", 3)]
    assert end == 14
    assert draft == list("ACTG")


def test_parse_factors_fibonacci_position_consumes_discriminator(decoder):
    decoder.input_file = "0011" + "0" + "110" + "101"
    assert decoder.parse_factors(1, 0, list("ACTGACTG")) == (
        [(3, "0", 0)], 8
    )


@pytest.mark.parametrize(
    ("seed", "factors", "expected"),
    [
        pytest.param("ACTG", [(2, "0", 1)], "ACTGCT", id="nonzero-source"),
        pytest.param("AC", [(5, "0", 0)], "ACACACA", id="overlapping-copy"),
        pytest.param("ACTG", [(4, "1", 4)], "ACTGTGAC", id="all-complements"),
        pytest.param("ACTG", [(2, "1", 2)], "ACTGAC", id="relative-complement"),
        pytest.param("A", [(5, "1", 1)], "ATATAT", id="overlapping-complement"),
        pytest.param(
            "AC", [(2, "0", 0), (4, "1", 4), (3, "0", 4)],
            "ACACTGTGTGT", id="chained-copies"
        ),
    ],
)
def test_decode_factors_copy_variants(decoder, seed, factors, expected):
    draft = list(seed)
    original_factors = list(factors)
    result = decoder.decode_factors(factors, draft)
    assert result is draft
    assert result == list(expected)
    assert factors == original_factors


@pytest.mark.parametrize(
    ("seed", "factors"),
    [("", []), ("ACTG", []), ("ACTG", [(0, "0", 0), (0, "1", 4)])],
)
def test_decode_factors_noop_preserves_list_identity(decoder, seed, factors):
    draft = list(seed)
    assert decoder.decode_factors(factors, draft) is draft
    assert draft == list(seed)


@pytest.mark.parametrize("factor", [(1, "0", 4), (1, "1", 0)])
def test_decode_factors_out_of_range_source_raises(decoder, factor):
    draft = list("ACTG")
    with pytest.raises(IndexError):
        decoder.decode_factors([factor], draft)
    assert draft == list("ACTG")


def test_main_empty_stream(decoder):
    decoder.input_file = ""
    decoder.main()
    decoder.output_file.write.assert_called_once_with("")
    decoder.output_file.close.assert_called_once_with()


def test_main_alternates_blocks_and_reports_consumed_bits(decoder, monkeypatch):
    literal_block = "1011" + "11100100"  # ACTG
    factor_block = "11" + "1011" + "0" + "01"  # One copy of four bases.
    final_block = "011" + "0011"  # GA
    decoder.input_file = literal_block + factor_block + final_block
    progress_factory = MagicMock()
    monkeypatch.setattr(decoder, "tqdm", progress_factory)

    decoder.main()

    decoder.output_file.write.assert_called_once_with("ACTGACTGGA")
    decoder.output_file.close.assert_called_once_with()
    progress_factory.assert_called_once_with(
        total=len(decoder.input_file), desc="Decompressing", unit="bytes"
    )
    progress = progress_factory.return_value.__enter__.return_value
    assert progress.update.call_args_list == [call(12), call(9), call(7)]
    progress_factory.return_value.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        pytest.param(
            "011" + "1110" + "11" + "00011" + "0" + "1",
            "ACACACA", id="overlapping-factor-at-eof"
        ),
        pytest.param(
            "000011" + "1110010011100100" + "11" + "1011" + "1" + "101",
            "ACTGACTGTGAC", id="complement-factor-at-eof"
        ),
    ],
)
def test_main_factor_variants_at_end_of_stream(decoder, stream, expected):
    decoder.input_file = stream
    decoder.main()
    decoder.output_file.write.assert_called_once_with(expected)
    decoder.output_file.close.assert_called_once_with()
