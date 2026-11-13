"""Compress DNA using sparse Biocompress 1 matching and alternating output runs.

Only token starts are indexed, as in the original implementation. Internal trie
nodes retain their first occurrence; height-level leaves retain every occurrence.
The historical 'palindrome' token denotes a forward complement, not a reversal.
No output file is opened and no trie is constructed when this module is imported.
"""

from dataclasses import dataclass
import json
import os
import sys
import time
from typing import Dict, Final, List, Literal, Optional, TextIO

import psutil
from tqdm import tqdm

from AGCT_tree import Node
from config import COMPLEMENT_TABLE, CONTENT, DNA_FILE, DNA_FILE_PATH, HEIGHT


MatchKind = Literal["factor", "palindrome"]
SegmentKind = Literal["base", "factor", "palindrome"]
BASE_BITS: Final[Dict[str, str]] = {
    "A": "11", "C": "10", "T": "01", "G": "00",
}
BRANCHES: Final[Dict[str, str]] = {
    "A": "a_branch", "C": "c_branch", "T": "t_branch", "G": "g_branch",
}


@dataclass(frozen=True)
class Match:
    """A positive-length match with an absolute, zero-based earlier source."""

    source: int
    length: int
    kind: MatchKind


@dataclass(frozen=True)
class EncodedSegment:
    """Wire payload, token kind, and number of input bases consumed."""

    bits: str
    kind: SegmentKind
    length: int


def get_memory_usage() -> float:
    """Return current process resident memory in MiB, rounded to two decimals."""
    rss = int(psutil.Process(os.getpid()).memory_info().rss)
    return round(rss / (1024 ** 2), 2)


def encode_fibonacci(number: int) -> str:
    """Encode a positive integer, constructing O(log(number)) code bits linearly."""
    if number < 1:
        raise ValueError("Fibonacci codes require a positive integer")
    numbers: List[int] = [1, 2]
    while numbers[-1] <= number:
        numbers.append(numbers[-1] + numbers[-2])
    numbers.pop()
    bits: List[str] = []
    for value in reversed(numbers):
        if number >= value:
            bits.append("1")
            number -= value
        else:
            bits.append("0")
    bits.reverse()
    bits.append("1")
    return "".join(bits)


def encode_position(number: int, window: int) -> Optional[str]:
    """Return a position field readable by the existing decoder, or None.

    The decoder uses ceil(log2(window)) binary bits, including at power-of-two
    boundaries. Its Fibonacci branch also reads width + 1 bits unconditionally.
    Therefore short Fibonacci fields are unsafe at EOF; only ones whose
    terminator ends exactly at width are considered. Binary fields escape the
    first '11' with an extra '1', while Fibonacci fields append a '0'.
    """
    if number < 1 or window < 1:
        raise ValueError("Position and window must be positive")
    width = (window - 1).bit_length()
    binary = format(number, "b")
    best: Optional[str] = None
    if len(binary) <= width:
        binary = binary.zfill(width)
        index = binary.find("11")
        if index != -1:
            binary = binary[:index + 2] + "1" + binary[index + 2:]
        best = binary
    fibonacci = encode_fibonacci(number)
    if len(fibonacci) == width:
        fibonacci += "0"
        if best is None or len(fibonacci) <= len(best):
            best = fibonacci
    return best


class Compressor:
    """Own the input and an initially empty, lazily allocated matching trie.

    For n bases and height h, construction costs O(n), not O(4**h). Indexing
    allocates at most O(n*h) nodes (also bounded by the full trie size), plus
    O(n) leaf positions. Each query costs O(h + k*e), where k is the number
    of candidate occurrences and e bounds extension work per candidate.
    Candidate scanning remains a worst-case bottleneck on repetitive inputs;
    this is not a linear-time longest-match algorithm.
    """

    def __init__(self, content: str, height: int) -> None:
        if height < 1:
            raise ValueError("HEIGHT must be positive")
        if any(base not in BASE_BITS for base in content):
            raise ValueError("DNA must contain only uppercase A, C, G and T")
        self.content: str = content
        self.complement: str = content.translate(COMPLEMENT_TABLE)
        self.height: int = height
        self.length: int = len(content)
        self.tree: Node = Node(level=0)

    def _find_prefix(self, i: int, query: str) -> Optional[Node]:
        """Find the deepest previously indexed prefix without allocating slices."""
        node = self.tree
        last: Optional[Node] = None
        for offset in range(min(self.height, self.length - i)):
            child = node._child_for_base(query[i + offset])
            if child is None or not child.positions:
                break
            last = child
            node = child
        return last

    def _index_segment(self, i: int) -> None:
        """Index a token start, matching Node.create_positions retention rules."""
        node = self.tree
        for offset in range(min(self.height, self.length - i)):
            base = self.content[i + offset]
            child = node._child_for_base(base)
            if child is None:
                level = node.level + 1
                child = Node(level=level, is_leaf=level == self.height)
                setattr(node, BRANCHES[base], child)
            if child.is_leaf or not child.positions:
                child.positions.append(i)
            node = child

    def extended_search(self, i: int, source: int, query: str) -> int:
        """Return total length after a known height-base match, allowing overlap.

        Callers supply an indexed source < i and a full-length query string.
        Consequently source + length stays in bounds whenever i + length does.
        Overlapping direct/complement copies are supported by the decoder.
        """
        length = self.height
        limit = self.length - i
        while (
            length < limit
            and query[i + length] == self.content[source + length]
        ):
            length += 1
        return length

    def _find_match(self, i: int, kind: MatchKind) -> Optional[Match]:
        """Extend indexed occurrences, retaining the earliest source on ties."""
        query = self.content if kind == "factor" else self.complement
        node = self._find_prefix(i, query)
        if node is None:
            return None
        source = node.positions[0]
        length = node.level
        if length == self.height:
            limit = self.length - i
            for candidate in node.positions:
                if length == limit:
                    break
                candidate_length = self.extended_search(i, candidate, query)
                if candidate_length > length:
                    source = candidate
                    length = candidate_length
        return Match(source, length, kind)

    def longest_factor_or_palindrome(self, i: int) -> Optional[Match]:
        """Choose either available match, preferring direct factors on ties."""
        if not 0 <= i < self.length:
            raise IndexError("Match position is outside the DNA sequence")
        factor = self._find_match(i, "factor")
        if factor is not None and factor.length == self.length - i:
            return factor
        palindrome = self._find_match(i, "palindrome")
        if factor is None:
            return palindrome
        if palindrome is None or factor.length >= palindrome.length:
            return factor
        return palindrome

    def _literal(self, i: int, length: int) -> EncodedSegment:
        """Encode the actual input in O(length) time, without repeated concatenation."""
        bits = "".join(BASE_BITS[self.content[j]] for j in range(i, i + length))
        return EncodedSegment(bits, "base", length)

    def _encode_match(self, match: Match, i: int) -> EncodedSegment:
        """Serialize absolute direct sources or relative complement distances.

        Positions are one-based on the wire. A pointer must beat two bits per
        base; ties use literals. An unrepresentable position consumes only one
        literal so a later token can retry with a larger decoder window.
        """
        position = match.source if match.kind == "factor" else i - match.source
        position_bits = encode_position(position + 1, i)
        if position_bits is None:
            return self._literal(i, 1)
        kind_bit = "0" if match.kind == "factor" else "1"
        bits = encode_fibonacci(match.length) + kind_bit + position_bits
        if match.length * 2 <= len(bits):
            return self._literal(i, match.length)
        return EncodedSegment(bits, match.kind, match.length)

    def process(self, i: int) -> EncodedSegment:
        """Process the next token; call once per start, in increasing order.

        Lookup precedes insertion so a token cannot point to itself. The first
        token is always literal, establishing the decoder's alternating run order.
        """
        match = self.longest_factor_or_palindrome(i)
        self._index_segment(i)
        if match is None:
            return self._literal(i, 1)
        return self._encode_match(match, i)


class RunWriter:
    """Buffer one homogeneous run with an incremental base or pointer count.

    Literal runs count decoded bases; pointer runs count tokens regardless of
    pointer kind. Memory is O(the largest run's encoded size). Arbitrarily
    splitting runs would break the decoder's implicit literal/pointer alternation.
    """

    def __init__(self, output: TextIO) -> None:
        self.output: TextIO = output
        self.parts: List[str] = []
        self.is_base: bool = True
        self.count: int = 0

    def write_buffer(self) -> None:
        """Flush the current run, doing nothing for empty input/buffers."""
        if not self.parts:
            return
        self.output.write(encode_fibonacci(self.count))
        self.output.writelines(self.parts)
        self.parts.clear()
        self.count = 0

    def encode(self, processed: EncodedSegment) -> None:
        """Append a payload, flushing only when the run changes category."""
        is_base = processed.kind == "base"
        if self.parts and is_base != self.is_base:
            self.write_buffer()
        self.is_base = is_base
        self.parts.append(processed.bits)
        self.count += processed.length if is_base else 1


def main() -> None:
    """Compress configured DNA and emit the existing JSON timing/RSS fields.

    With lazy allocation, tree_creation_time measures initialization only;
    subsequent node allocation belongs to compressor_time. tree_memory remains
    an initialization-time process RSS snapshot, not a peak-memory measurement.
    """
    start_time = time.perf_counter()
    compressor = Compressor(CONTENT, HEIGHT)
    tree_created_time = time.perf_counter()
    tree_memory = get_memory_usage()
    output_path = f"{DNA_FILE_PATH}{DNA_FILE}_{HEIGHT}_encoded.txt"
    with open(output_path, "w", encoding="utf-8") as output:
        writer = RunWriter(output)
        position = 0
        with tqdm(
            total=compressor.length,
            desc="Compressing",
            unit="bytes",
            file=sys.stderr,
            leave=True,
            disable=not sys.stderr.isatty(),
        ) as pbar:
            while position < compressor.length:
                processed = compressor.process(position)
                writer.encode(processed)
                position += processed.length
                pbar.update(processed.length)
        writer.write_buffer()
    finished_time = time.perf_counter()
    print(json.dumps({
        "total_compression_time": round(finished_time - start_time, 3),
        "tree_creation_time": round(tree_created_time - start_time, 3),
        "compressor_time": round(finished_time - tree_created_time, 3),
        "tree_memory": tree_memory,
    }))


if __name__ == "__main__":
    main()
