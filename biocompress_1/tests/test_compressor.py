from io import StringIO
from itertools import product
import random

import pytest

from compressor import (
    Compressor,
    EncodedSegment,
    Match,
    RunWriter,
    encode_fibonacci,
    encode_position,
)


def test_initialization_allocates_only_root():
    compressor = Compressor("ACGT", 3)

    assert compressor.content == "ACGT"
    assert compressor.complement == "TGCA"
    assert compressor.length == 4
    assert compressor.height == 3
    assert compressor.tree.level == 0
    assert compressor.tree.positions == []
    assert all(
        compressor.tree._child_for_base(base) is None for base in "ACGT"
    )


@pytest.mark.parametrize("base,bits", [("A", "11"), ("C", "10"), ("G", "00"), ("T", "01")])
def test_first_token_is_literal(base, bits):
    compressor = Compressor(base, 2)

    assert compressor.process(0) == EncodedSegment(bits, "base", 1)
    assert compressor.tree._child_for_base(base).positions == [0]


def test_find_direct_factor():
    compressor = Compressor("ACGACG", 3)
    compressor.process(0)

    assert compressor.longest_factor_or_palindrome(3) == Match(0, 3, "factor")


def test_find_forward_complement_not_reverse_complement():
    compressor = Compressor("ACGTGC", 3)
    compressor.process(0)

    assert compressor.longest_factor_or_palindrome(3) == Match(0, 3, "palindrome")


@pytest.mark.parametrize(
    "number,bits", [(1, "11"), (2, "011"), (3, "0011"), (4, "1011"), (5, "00011")]
)
def test_fibonacci_known_codes(number, bits):
    assert encode_fibonacci(number) == bits


def test_writer_emits_literal_run():
    output = StringIO()
    writer = RunWriter(output)
    writer.encode(EncodedSegment("11", "base", 1))
    writer.encode(EncodedSegment("10", "base", 1))

    assert output.getvalue() == ""
    writer.write_buffer()
    assert output.getvalue() == "0111110"
    assert writer.parts == []
    assert writer.count == 0


@pytest.mark.parametrize("height", [0, -1, -10])
def test_rejects_nonpositive_height(height):
    with pytest.raises(ValueError, match="HEIGHT must be positive"):
        Compressor("ACGT", height)


@pytest.mark.parametrize("content", ["acgt", "N", "ACNT", "AC GT", "ACGT\n", "A1", "Á"])
def test_rejects_non_dna_content(content):
    with pytest.raises(ValueError, match="only uppercase A, C, G and T"):
        Compressor(content, 2)


@pytest.mark.parametrize("content,index", [("", 0), ("", -1), ("AC", -1), ("AC", 2), ("AC", 3)])
@pytest.mark.parametrize("method", ["process", "longest_factor_or_palindrome"])
def test_out_of_range_positions_raise_without_indexing(content, index, method):
    compressor = Compressor(content, 2)

    with pytest.raises(IndexError, match="outside the DNA sequence"):
        getattr(compressor, method)(index)

    assert compressor.tree.positions == []
    assert all(compressor.tree._child_for_base(base) is None for base in "ACGT")


def test_instances_do_not_share_trie_positions():
    first = Compressor("AAAA", 2)
    second = Compressor("AAAA", 2)
    first.process(0)

    assert second.longest_factor_or_palindrome(1) is None
    second.process(0)
    first._index_segment(1)
    assert first.tree.a_branch.a_branch.positions == [0, 1]
    assert second.tree.a_branch.a_branch.positions == [0]
    assert first.tree is not second.tree


def test_sparse_index_allocates_only_observed_path_even_at_large_height():
    compressor = Compressor("ACG", 10000)
    compressor._index_segment(0)
    node = compressor.tree

    for level, base in enumerate("ACG", start=1):
        assert all(
            node._child_for_base(other) is None
            for other in "ACGT" if other != base
        )
        node = node._child_for_base(base)
        assert node.level == level
        assert node.positions == [0]
        assert node.is_leaf is False

    assert all(node._child_for_base(base) is None for base in "ACGT")
    assert compressor.tree.positions == []


def test_sparse_index_retains_first_internal_and_all_leaf_positions():
    compressor = Compressor("AAAAA", 3)
    for index in range(3):
        compressor._index_segment(index)

    first = compressor.tree.a_branch
    second = first.a_branch
    leaf = second.a_branch
    assert first.positions == [0]
    assert second.positions == [0]
    assert leaf.positions == [0, 1, 2]
    assert leaf.is_leaf is True
    assert all(leaf._child_for_base(base) is None for base in "ACGT")


def test_height_one_records_every_occurrence_at_root_children():
    compressor = Compressor("ACA", 1)
    for index in range(3):
        compressor._index_segment(index)

    assert compressor.tree.a_branch.positions == [0, 2]
    assert compressor.tree.c_branch.positions == [1]
    assert compressor.tree.a_branch.is_leaf is True
    assert compressor.tree.c_branch.is_leaf is True


def test_indexing_final_base_does_not_create_padding_nodes():
    compressor = Compressor("ACGT", 3)
    compressor._index_segment(3)
    node = compressor.tree.t_branch

    assert node.level == 1
    assert node.positions == [3]
    assert node.is_leaf is False
    assert all(node._child_for_base(base) is None for base in "ACGT")
    assert compressor.tree.a_branch is None


@pytest.mark.parametrize("content", ["ACGACT", "ACGAC"])
def test_match_returns_shorter_indexed_prefix_on_mismatch_or_eof(content):
    compressor = Compressor(content, 3)
    compressor._index_segment(0)

    assert compressor.longest_factor_or_palindrome(3) == Match(0, 2, "factor")


def test_lookup_does_not_mutate_sparse_tree():
    compressor = Compressor("ACGCCC", 3)
    compressor._index_segment(0)

    assert compressor.longest_factor_or_palindrome(3) is None
    assert compressor.tree.c_branch is None
    assert compressor.tree.g_branch is None
    assert compressor.tree.a_branch.c_branch.g_branch.positions == [0]


@pytest.mark.parametrize(
    "content,index,expected", [("AAAAAAAA", 1, 7), ("ACGACGAT", 3, 4)]
)
def test_extension_allows_overlap_and_stops_at_mismatch_or_eof(content, index, expected):
    compressor = Compressor(content, 2)
    compressor._index_segment(0)

    assert compressor.extended_search(index, 0, compressor.content) == expected
    assert compressor.longest_factor_or_palindrome(index) == Match(0, expected, "factor")


def test_complement_extension_allows_overlap():
    compressor = Compressor("ATATATAT", 2)
    compressor._index_segment(0)

    assert compressor.extended_search(1, 0, compressor.complement) == 7
    assert compressor.longest_factor_or_palindrome(1) == Match(0, 7, "palindrome")


def test_later_leaf_candidate_can_produce_longer_match():
    compressor = Compressor("ACGTAACGTTACGTT", 3)
    compressor._index_segment(0)
    compressor._index_segment(5)

    assert compressor.longest_factor_or_palindrome(10) == Match(5, 5, "factor")


def test_equal_extended_candidates_keep_earliest_source():
    compressor = Compressor("ACGTAACGTAACGTAC", 3)
    compressor._index_segment(0)
    compressor._index_segment(5)

    assert compressor.longest_factor_or_palindrome(10) == Match(0, 5, "factor")


def test_equal_direct_and_complement_lengths_prefer_direct():
    compressor = Compressor("ACTGACG", 2)
    compressor._index_segment(0)
    compressor._index_segment(2)

    assert compressor._find_match(4, "factor") == Match(0, 2, "factor")
    assert compressor._find_match(4, "palindrome") == Match(2, 2, "palindrome")
    assert compressor.longest_factor_or_palindrome(4) == Match(0, 2, "factor")


def test_longer_complement_beats_available_direct_match():
    compressor = Compressor("ACTGCTACG", 2)
    compressor._index_segment(0)
    compressor._index_segment(2)

    assert compressor._find_match(6, "factor") == Match(0, 2, "factor")
    assert compressor.longest_factor_or_palindrome(6) == Match(2, 3, "palindrome")


@pytest.mark.parametrize("number", [0, -1, -100])
def test_fibonacci_rejects_nonpositive_numbers(number):
    with pytest.raises(ValueError, match="positive integer"):
        encode_fibonacci(number)


@pytest.mark.parametrize("number,window", [(0, 8), (-1, 8), (1, 0), (1, -1)])
def test_position_rejects_nonpositive_arguments(number, window):
    with pytest.raises(ValueError, match="must be positive"):
        encode_position(number, window)


@pytest.mark.parametrize(
    "number,window,expected",
    [
        (1, 1, None),       # Zero decoder width cannot represent a position.
        (1, 2, "1"),
        (1, 3, "01"),
        (1, 4, "01"),      # Exact power of two still uses ceil(log2(window)).
        (1, 5, "001"),
        (3, 4, "111"),     # Binary '11' is escaped.
        (3, 8, "0111"),
        (7, 8, "1111"),    # Only the first '11' is escaped.
        (3, 16, "00110"),  # Equal-cost Fibonacci field ends at decoder width.
        (1, 16, "0001"),   # Short Fibonacci codes are unsafe at EOF.
        (4, 4, None),       # Neither field is representable in this window.
    ],
)
def test_position_encoding_width_escape_and_eof_boundaries(number, window, expected):
    assert encode_position(number, window) == expected


@pytest.mark.parametrize(
    "content,index,match,expected",
    [
        ("A" * 12, 4, Match(0, 8, "factor"), EncodedSegment("000011001", "factor", 8)),
        # Complement uses relative position (8 - 2) + 1 = 7, not source + 1 = 3.
        ("CCAACCGGTTGGCCAA", 8, Match(2, 8, "palindrome"), EncodedSegment("00001111111", "palindrome", 8)),
    ],
)
def test_profitable_pointer_serialization(content, index, match, expected):
    compressor = Compressor(content, 2)

    assert compressor._encode_match(match, index) == expected


def test_pointer_cost_tie_emits_literals():
    compressor = Compressor("AAAAA", 2)
    # Fibonacci length 0011 + kind 0 + position 1 costs exactly six bits.
    assert compressor._encode_match(Match(0, 3, "factor"), 2) == EncodedSegment("111111", "base", 3)


def test_representable_costly_complement_falls_back_to_entire_target():
    compressor = Compressor("AAAATT", 2)
    # Relative position (4 - 2) + 1 = 3 is representable; pointer is not cheaper.
    # Encode the target T bases, not the earlier A bases.
    assert compressor._encode_match(Match(2, 2, "palindrome"), 4) == EncodedSegment("0101", "base", 2)


def test_unrepresentable_pointer_consumes_one_literal_for_retry():
    compressor = Compressor("AAAAAAAA", 2)

    assert compressor._encode_match(Match(0, 7, "factor"), 1) == EncodedSegment("11", "base", 1)


def test_processing_indexes_token_starts_not_pointer_interior():
    compressor = Compressor("A" * 12, 2)

    assert compressor.process(0) == EncodedSegment("11", "base", 1)
    assert compressor.process(1) == EncodedSegment("11", "base", 1)
    assert compressor.process(2) == EncodedSegment("01001101", "factor", 10)
    assert compressor.tree.a_branch.positions == [0]
    assert compressor.tree.a_branch.a_branch.positions == [0, 1, 2]


def test_writer_groups_pointer_kinds_and_counts_tokens_not_lengths():
    output = StringIO()
    writer = RunWriter(output)
    writer.encode(EncodedSegment("1110", "base", 2))
    writer.encode(EncodedSegment("010", "factor", 20))

    assert output.getvalue() == "0111110"
    writer.encode(EncodedSegment("101", "palindrome", 30))
    assert output.getvalue() == "0111110"
    assert writer.count == 2
    writer.encode(EncodedSegment("00", "base", 1))
    assert output.getvalue() == "0111110" + "011010101"
    writer.write_buffer()
    assert output.getvalue() == "0111110" + "011010101" + "1100"


def test_empty_and_repeated_final_flush_do_not_emit_headers():
    output = StringIO()
    writer = RunWriter(output)
    writer.write_buffer()
    assert output.getvalue() == ""
    writer.encode(EncodedSegment("01", "base", 1))
    writer.write_buffer()
    writer.write_buffer()

    assert output.getvalue() == "1101"
    assert writer.parts == []
    assert writer.count == 0


# Independent wire reader: no production encoders or matching methods are used
# to reconstruct the DNA. This checks headers, position fields, and overlap.
def _read_fibonacci(bits, offset):
    end = bits.find("11", offset)
    assert end >= offset, "Missing Fibonacci terminator"
    value = 0
    current, following = 1, 2
    for bit in bits[offset:end + 1]:
        if bit == "1":
            value += current
        current, following = following, current + following
    return value, end + 2


def _decode_stream(bits):
    decoded = []
    offset = 0
    is_base = True
    bases = {"11": "A", "10": "C", "00": "G", "01": "T"}
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    assert set(bits) <= {"0", "1"}
    while offset < len(bits):
        count, offset = _read_fibonacci(bits, offset)
        assert count > 0
        if is_base:
            for _ in range(count):
                decoded.append(bases[bits[offset:offset + 2]])
                offset += 2
        else:
            for _ in range(count):
                length, offset = _read_fibonacci(bits, offset)
                kind = bits[offset]
                offset += 1
                width = (len(decoded) - 1).bit_length()
                field = bits[offset:offset + width]
                assert len(field) == width
                escape = field.find("11")
                if escape < 0:
                    position = int(field, 2)
                    offset += width
                else:
                    field = bits[offset:offset + width + 1]
                    assert len(field) == width + 1
                    if field[escape + 2] == "0":
                        position, end = _read_fibonacci(field, 0)
                        assert end == width
                    else:
                        binary = field[:escape + 2] + field[escape + 3:]
                        position = int(binary, 2)
                    offset += width + 1
                source = position - 1 if kind == "0" else len(decoded) - (position - 1)
                assert 0 <= source < len(decoded)
                for index in range(length):
                    base = decoded[source + index]
                    decoded.append(base if kind == "0" else complement[base])
        is_base = not is_base
    assert offset == len(bits)
    return "".join(decoded)


def _compress_stream(content, height):
    compressor = Compressor(content, height)
    output = StringIO()
    writer = RunWriter(output)
    index = 0
    segments = []
    while index < len(content):
        segment = compressor.process(index)
        assert 0 < segment.length <= len(content) - index
        assert set(segment.bits) <= {"0", "1"}
        segments.append(segment)
        writer.encode(segment)
        index += segment.length
    writer.write_buffer()
    assert index == len(content)
    return output.getvalue(), segments


@pytest.mark.parametrize("number", [6, 7, 8, 13, 21, 55, 1024, 10**20])
def test_large_fibonacci_codes_round_trip_independently(number):
    bits = encode_fibonacci(number)

    assert bits.endswith("11")
    assert "11" not in bits[:-1]
    assert _read_fibonacci(bits, 0) == (number, len(bits))


@pytest.mark.parametrize("height", [1, 2, 4, 50])
@pytest.mark.parametrize(
    "content",
    ["", "ACGT", "A" * 64, "AT" * 32, "ACGT" * 16,
     "A" * 8 + "T" * 20, "ACGTTGCA" * 10, "ACGACGTTTACGACGCCC"],
)
def test_encoded_stream_round_trips(content, height):
    bits, _ = _compress_stream(content, height)

    assert _decode_stream(bits) == content


def test_round_trip_actually_exercises_both_pointer_kinds():
    # At position 9, complement distance + 1 fits the four-bit position field.
    content = "A" * 9 + "T" * 20
    bits, segments = _compress_stream(content, 3)

    assert {segment.kind for segment in segments} == {"base", "factor", "palindrome"}
    assert _decode_stream(bits) == content


@pytest.mark.parametrize("height", [1, 3, 8])
def test_seeded_random_sequences_round_trip(height):
    rng = random.Random(1729)
    for length in [7, 31, 128, 513]:
        content = "".join(rng.choice("ACGT") for _ in range(length))
        bits, _ = _compress_stream(content, height)
        assert _decode_stream(bits) == content, (height, content)


@pytest.mark.parametrize("height", [1, 2, 5])
def test_exhaustive_short_sequences_round_trip(height):
    for length in range(1, 5):
        for letters in product("ACGT", repeat=length):
            content = "".join(letters)
            bits, _ = _compress_stream(content, height)
            assert _decode_stream(bits) == content, (height, content)
