"""Single-command Biocompress 1 pipeline (Python 3.9+).

Setup:
    python -m pip install -r requirements.txt
Run from the project directory (existing generated files require --force):
    python run_compressor.py --force
    python run_compressor.py data/another.fna --height 8

Default input is data/E_Coli.fna relative to this script, regardless of cwd.
Outputs beside it: E_Coli.txt, E_Coli_11.bin, E_Coli_11_decoded.txt,
and E_Coli_11_metrics.csv. The source FASTA is never modified.
Normalized/decoded files contain uppercase ACGT only, with no headers, wrapping,
or final newline. FASTA records are concatenated. Non-ACGT symbols are rejected
unless --drop-non-acgt explicitly permits their irreversible removal.

Binary container v1 (not the external dnazip format):
    8 bytes magic b'BC1BIN1\\0'
    8 bytes unsigned big-endian payload bit count
    MSB-first packed Biocompress bits; unused final low bits are zero
The header preserves the exact bit count so padding cannot become decoder input.
The existing decompressor.py CLI expects ASCII bits; this runner's read_binary
function unpacks the container for the existing Decompressor class instead.

CSV sizes are bytes; original_file_size_bytes means the normalized input, while
source_file_size_bytes includes the original FASTA metadata/wrapping. Compressed
size includes the container header. Timings are seconds. tree_creation_time is
Compressor initialization (validation, complement and empty root); lazy tree
population occurs during compression_time, which also includes binary packing
and output I/O. Neither timing includes preprocessing or decompression.

All outputs are staged and verified before publication. Each final replacement
is atomic, but publication of all four files is not a multi-file transaction.
Do not run concurrent pipelines targeting the same output filenames. The CSV
contains one successful run, not an append-only history; --force replaces it.
"""

import argparse
import csv
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import BinaryIO, Iterable, Sequence, Tuple

from config import HEIGHT
from preprocessor import preprocess


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "E_Coli.fna"
MAGIC = b"BC1BIN1\0"
HEADER = struct.Struct(">8sQ")
CSV_FIELDS = (
    "genome_name",
    "tree_height",
    "tree_creation_time_seconds",
    "compression_time_seconds",
    "total_compression_time_seconds",
    "original_file_size_bytes",
    "compressed_file_size_bytes",
    "source_file_size_bytes",
    "encoded_bit_count",
    "compressed_to_original_ratio",
    "dropped_non_acgt_symbols",
)


class PackedBitWriter:
    """The write/writelines sink used by RunWriter, packing bits incrementally.

    The caller owns the binary file and must call finish() before closing it.
    Only a short tail and one bounded conversion chunk are buffered here;
    RunWriter still buffers each homogeneous run according to its wire format.
    """

    def __init__(self, output: BinaryIO) -> None:
        self.output = output
        self.bit_count = 0
        self.tail = ""
        self.finished = False
        output.write(HEADER.pack(MAGIC, 0))

    def write(self, bits: str) -> int:
        if self.finished:
            raise ValueError("Cannot write after binary stream finalization")
        for start in range(0, len(bits), 8192):
            chunk = bits[start:start + 8192]
            if any(bit not in "01" for bit in chunk):
                raise ValueError("Compressor emitted a non-binary symbol")
            combined = self.tail + chunk
            boundary = len(combined) // 8 * 8
            if boundary:
                self.output.write(
                    int(combined[:boundary], 2).to_bytes(boundary // 8, "big")
                )
            self.tail = combined[boundary:]
            self.bit_count += len(chunk)
        return len(bits)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def finish(self) -> None:
        if self.finished:
            return
        if self.tail:
            self.output.write(bytes([int(self.tail.ljust(8, "0"), 2)]))
        self.output.seek(0)
        self.output.write(HEADER.pack(MAGIC, self.bit_count))
        self.output.flush()
        self.finished = True


def read_binary(path: Path) -> str:
    """Read this runner's binary container into the decoder's exact bit string."""
    with path.open("rb") as handle:
        header = handle.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("Truncated Biocompress binary header")
        magic, bit_count = HEADER.unpack(header)
        if magic != MAGIC:
            raise ValueError("Unsupported Biocompress binary format/version")
        expected_bytes = (bit_count + 7) // 8
        if os.fstat(handle.fileno()).st_size != HEADER.size + expected_bytes:
            raise ValueError("Binary payload size does not match its bit count")
        payload = handle.read()
    if len(payload) != expected_bytes:
        raise ValueError("Truncated Biocompress binary payload")
    padding = (-bit_count) % 8
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ValueError("Nonzero padding in Biocompress binary payload")
    # Bound the intermediate list overhead while assembling the decoder string.
    table = tuple(format(value, "08b") for value in range(256))
    chunks = [
        "".join(table[value] for value in payload[start:start + 65536])
        for start in range(0, len(payload), 65536)
    ]
    return "".join(chunks)[:bit_count]


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("height must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("height must be a positive integer")
    return number


def require_workspace_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Path must be inside the project workspace: {path}") from error
    return resolved


def output_paths(source: Path, height: int) -> Tuple[Path, Path, Path, Path]:
    normalized_name = (
        f"{source.stem}_preprocessed.txt"
        if source.suffix.lower() == ".txt" else f"{source.stem}.txt"
    )
    prefix = f"{source.stem}_{height}"
    return (
        source.with_name(normalized_name),
        source.with_name(prefix + ".bin"),
        source.with_name(prefix + "_decoded.txt"),
        source.with_name(source.stem + "_metrics.csv"),
    )


def check_outputs(source: Path, outputs: Sequence[Path], force: bool) -> None:
    for output in outputs:
        require_workspace_path(output)
        if output.is_symlink():
            raise ValueError(f"Refusing to replace an output symlink: {output}")
        if output == source or (output.exists() and output.samefile(source)):
            raise ValueError(f"Output would overwrite the source: {output}")
        if output.exists():
            if not output.is_file():
                raise ValueError(f"Output path is not a regular file: {output}")
            if not force:
                raise FileExistsError(
                    f"Output already exists: {output}\n"
                    "Use --force to replace generated files. The input is never replaced."
                )


def run_pipeline(args: argparse.Namespace) -> None:
    source = require_workspace_path(args.input)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    outputs = output_paths(source, args.height)
    check_outputs(source, outputs, args.force)

    # Lazy imports keep --help available before dependency installation and avoid
    # the external dnazip dependency and import-time writes in biocompress.py.
    try:
        from compressor import Compressor, RunWriter
        from decompressor import Decompressor
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError(
            f"Could not import compression dependencies: {error}\n"
            "Install dependencies using this Python interpreter:\n"
            f'  "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"'
        ) from error

    started = time.perf_counter()
    print(f"Input: {source}", flush=True)
    print(f"Tree height: {args.height}", flush=True)

    with tempfile.TemporaryDirectory(prefix=".biocompress-", dir=source.parent) as folder:
        staged = tuple(Path(folder) / output.name for output in outputs)

        print("[1/4] Preprocessing FASTA/DNA...", flush=True)
        result = preprocess(source, staged[0], drop_non_acgt=args.drop_non_acgt)
        content = result.content
        print(f"Normalized {len(content):,} bases.", flush=True)
        if result.records:
            print(
                f"Removed FASTA headers; concatenated {result.records:,} record(s).",
                flush=True,
            )
        if result.dropped_symbols:
            print(
                f"WARNING: removed {result.dropped_symbols:,} non-ACGT symbols. "
                "These symbols cannot be recovered from the compressed output.",
                file=sys.stderr, flush=True,
            )

        print("[2/4] Compressing to packed binary...", flush=True)
        initialization_started = time.perf_counter()
        compressor = Compressor(content, args.height)
        initialization_finished = time.perf_counter()
        with staged[1].open("wb") as encoded:
            packed = PackedBitWriter(encoded)
            # RunWriter needs only write(str) and writelines(Iterable[str]);
            # the adapter preserves its existing bitstream without algorithm changes.
            writer = RunWriter(packed)
            position = 0
            with tqdm(
                total=len(content), desc="Compressing", unit="bases",
                file=sys.stderr, disable=not sys.stderr.isatty(),
            ) as progress:
                while position < len(content):
                    segment = compressor.process(position)
                    if not 0 < segment.length <= len(content) - position:
                        raise RuntimeError("Compressor returned an invalid segment length")
                    writer.encode(segment)
                    position += segment.length
                    progress.update(segment.length)
            writer.write_buffer()
            packed.finish()
        compression_finished = time.perf_counter()
        bit_count = packed.bit_count
        # Release the matching trie before allocating the decoded genome.
        del compressor, writer, packed

        print("[3/4] Reading binary, decompressing and verifying...", flush=True)
        decoder = Decompressor(read_binary(staged[1]))
        with tqdm(
            total=len(decoder.bits), desc="Decompressing", unit="bits",
            file=sys.stderr, disable=not sys.stderr.isatty(),
        ) as progress:
            while decoder.position < len(decoder.bits):
                previous = decoder.position
                decoder.decode_run()
                progress.update(decoder.position - previous)
        decoded = "".join(decoder.output)
        del decoder
        if decoded != content:
            raise RuntimeError(
                "Round-trip verification failed: decoded DNA differs from the "
                "normalized input. Generated results were not published."
            )
        staged[2].write_text(decoded, encoding="ascii")
        del decoded

        print("[4/4] Recording metrics and publishing results...", flush=True)
        original_size = staged[0].stat().st_size
        compressed_size = staged[1].stat().st_size
        tree_time = initialization_finished - initialization_started
        compression_time = compression_finished - initialization_finished
        metrics = {
            "genome_name": source.stem,
            "tree_height": args.height,
            "tree_creation_time_seconds": f"{tree_time:.9f}",
            "compression_time_seconds": f"{compression_time:.9f}",
            "total_compression_time_seconds": f"{tree_time + compression_time:.9f}",
            "original_file_size_bytes": original_size,
            "compressed_file_size_bytes": compressed_size,
            "source_file_size_bytes": source.stat().st_size,
            "encoded_bit_count": bit_count,
            "compressed_to_original_ratio": f"{compressed_size / original_size:.9f}",
            "dropped_non_acgt_symbols": result.dropped_symbols,
        }
        with staged[3].open("w", newline="", encoding="utf-8") as handle:
            report = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            report.writeheader()
            report.writerow(metrics)

        # Recheck in case an output appeared during the run. Individual renames
        # are atomic; interruption here can still leave a partially updated set.
        check_outputs(source, outputs, args.force)
        for temporary, destination in zip(staged, outputs):
            os.replace(temporary, destination)

    print(f"Completed in {time.perf_counter() - started:.2f} seconds.")
    print(f"Verified: binary decodes to all {len(content):,} normalized bases exactly.")
    print(f"Tree initialization: {tree_time:.6f} s; compression: {compression_time:.6f} s")
    print(
        f"Normalized input: {original_size:,} bytes; binary: {compressed_size:,} bytes "
        f"({compressed_size / original_size:.2%} of normalized input)."
    )
    for label, output in zip(("Preprocessed", "Binary", "Decoded", "Metrics"), outputs):
        print(f"{label}: {output}")
    print("The source file is unchanged. Decoded output is normalized DNA, not original FASTA.")
    print("Tree creation time measures initialization; lazy tree growth is timed as compression.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess, compress to binary, decompress, verify, and report DNA metrics.",
        epilog=(
            "All outputs are saved beside the input within this workspace. "
            "FASTA metadata is omitted and records are concatenated. "
            "Existing outputs require --force, including data/E_Coli.txt. "
            "Dependencies: python -m pip install psutil tqdm"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT,
        help="FASTA (.fna/.fa/.fasta) or plain DNA; relative paths use the current directory",
    )
    parser.add_argument("--height", type=positive_integer, default=HEIGHT,
                        help="Matching trie height")
    parser.add_argument("--force", action="store_true",
                        help="Replace generated output files and CSV, never the input")
    parser.add_argument(
        "--drop-non-acgt", action="store_true",
        help="Explicitly discard ambiguous/non-ACGT symbols (lossy); otherwise reject them",
    )
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except MemoryError:
        print("Error: insufficient memory. Try a smaller --height or a smaller input.",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, IndexError, OverflowError, struct.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
