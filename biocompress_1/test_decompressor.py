import builtins
import importlib
import io
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decompressor
from compressor import Compressor, RunWriter, encode_fibonacci, encode_position
from decompressor import Decompressor, decompress


BASE_BITS = {"A": "11", "C": "10", "T": "01", "G": "00"}


def literal(sequence):
    return encode_fibonacci(len(sequence)) + "".join(BASE_BITS[b] for b in sequence)


def pointer(length, complement, position, window):
    encoded = encode_position(position, window)
    assert encoded is not None
    return encode_fibonacci(length) + str(int(complement)) + encoded


def compress(sequence, height):
    output = io.StringIO()
    compressor = Compressor(sequence, height)
    writer = RunWriter(output)
    position = 0
    while position < len(sequence):
        segment = compressor.process(position)
        writer.encode(segment)
        position += segment.length
    writer.write_buffer()
    return output.getvalue()


@pytest.mark.parametrize("number,bits", [
    (1, "11"), (2, "011"), (3, "0011"), (4, "1011"),
    (5, "00011"), (6, "10011"), (7, "01011"), (8, "000011"),
])
def test_fibonacci_vectors(number, bits):
    decoder = Decompressor(bits)
    assert decoder.read_number() == number
    assert decoder.position == len(bits)


def test_fibonacci_large_values():
    for number in [1, 13, 123456, 2 ** 128, 2 ** 512 + 123]:
        bits = encode_fibonacci(number)
        decoder = Decompressor(bits)
        assert decoder.read_number() == number
        assert decoder.position == len(bits)


def test_literal_vectors():
    assert decompress("") == ""
    assert decompress("1111") == "A"
    assert decompress("101111100100") == "ACTG"


@pytest.mark.parametrize("window", [2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128])
def test_position_fields(window):
    width = (window - 1).bit_length()
    for number in range(1, window + 2):
        fields = []
        encoded = encode_position(number, window)
        if encoded is not None:
            fields.append(encoded)
        fibonacci = encode_fibonacci(number)
        if len(fibonacci) <= width:
            fields.append(fibonacci + "0")
        binary = format(number, f"0{width}b")
        if len(binary) == width:
            index = binary.find("11")
            if index != -1:
                binary = binary[:index + 2] + "1" + binary[index + 2:]
            fields.append(binary)
        for bits in fields:
            for suffix in ["", "101010"]:
                decoder = Decompressor(bits + suffix)
                assert decoder.read_position(window) == number
                assert decoder.position == len(bits)


def test_direct_overlap():
    bits = literal("AC") + "11" + pointer(18, False, 1, 2)
    assert decompress(bits) == "AC" * 10


def test_forward_complement_overlap():
    bits = literal("ACGTA") + "11" + pointer(7, True, 2, 5)
    assert decompress(bits) == "ACGTA" + "TATATAT"


def test_forward_complement_not_reversal():
    bits = literal("ACGTA") + "11" + pointer(3, True, 6, 5)
    assert decompress(bits) == "ACGTATGC"


def test_consecutive_factors_and_literal_runs():
    bits = literal("AC") + encode_fibonacci(2)
    bits += pointer(3, False, 1, 2)
    bits += pointer(4, True, 2, 5)
    bits += literal("GT")
    assert decompress(bits) == "ACACA" + "TATA" + "GT"


def test_short_fibonacci_position_at_eof():
    seed = "ACGT" * 8
    bits = literal(seed) + "11" + encode_fibonacci(3) + "0" + "110"
    assert decompress(bits) == seed + "ACG"


@pytest.mark.parametrize("bits", [
    "0", "1", "2", "11", "01111", "11x1", "1111\n",
    literal("AC") + "11",
    literal("AC") + "11" + "11" + "0",
    literal("AC") + "11" + "11" + "0" + "0",
    literal("ACGTA") + "11" + "11" + "0" + "1111",
    literal("ACGTA") + "11" + "11" + "1" + "001",
    literal("ACGTA") + "11" + "11" + "0" + "11",
    literal("ACGTA") + "11" + "11" + "0" + "111",
    literal("A") + "11" + "11" + "0",
])
def test_malformed_streams(bits):
    with pytest.raises(ValueError):
        decompress(bits)


@pytest.mark.parametrize("height", [1, 2, 4, 11])
def test_compressor_round_trips(height):
    rng = random.Random(748)
    sequences = ["", "A", "ACGT", "A" * 1000, "AC" * 300,
                 "ACGT" * 200, "ACGTTGCA" * 100]
    sequences += ["".join(rng.choice("ACGT") for _ in range(length))
                  for length in list(range(1, 70)) + [128, 255, 512]]
    for sequence in sequences:
        assert decompress(compress(sequence, height)) == sequence


def test_import_has_no_file_io(monkeypatch):
    def unexpected_open(*args, **kwargs):
        raise AssertionError("Import must not open DNA files")
    with monkeypatch.context() as context:
        context.setattr(builtins, "open", unexpected_open)
        importlib.reload(decompressor)


def test_main(tmp_path, monkeypatch):
    monkeypatch.setattr(decompressor, "DNA_FILE_PATH", str(tmp_path) + "/")
    monkeypatch.setattr(decompressor, "DNA_FILE", "sample")
    monkeypatch.setattr(decompressor, "HEIGHT", 3)
    encoded = tmp_path / "sample_3_encoded.txt"
    decoded = tmp_path / "sample_3_decoded.txt"
    sequence = "ACGTTGCA" * 40
    encoded.write_text(compress(sequence, 3), encoding="utf-8")
    decoded.write_text("old contents", encoding="utf-8")
    decompressor.main()
    assert decoded.read_text(encoding="utf-8") == sequence
    encoded.write_text("1", encoding="utf-8")
    with pytest.raises(ValueError):
        decompressor.main()
    assert decoded.read_text(encoding="utf-8") == sequence
