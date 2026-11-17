"""Run preprocessing, Biocompress 1 compression, and decompression together.

Setup (Python 3.9+):
    python -m pip install -r requirements.txt

From the project folder:
    python run_compressor.py
    python run_compressor.py --force
    python run_compressor.py data/another.fna --height 8

The default input is data/E_Coli.fna relative to this script, not the working
 directory. All outputs are saved beside the input. The encoded file contains
ASCII '0'/'1' characters in the project's existing format, NOT packed binary.
Verification compares decoded DNA against normalized DNA, not FASTA metadata.
No files outside this workspace or external dnazip modules are needed.
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Sequence, Tuple

from config import HEIGHT
from preprocessor import preprocess


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "E_Coli.fna"


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


def output_paths(source: Path, height: int) -> Tuple[Path, Path, Path]:
    # Plain .txt input is supported, but must never be its own normalized output.
    normalized_name = (
        f"{source.stem}_preprocessed.txt"
        if source.suffix.lower() == ".txt" else f"{source.stem}.txt"
    )
    prefix = f"{source.stem}_{height}"
    return (
        source.with_name(normalized_name),
        source.with_name(prefix + "_encoded.txt"),
        source.with_name(prefix + "_decoded.txt"),
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

    # These imports no longer load a configured DNA file. Keeping them here also
    # lets --help work before third-party dependencies have been installed.
    try:
        from compressor import Compressor, RunWriter
        from decompressor import Decompressor
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError(
            f"Could not import compression dependencies: {error}\n"
            "Install dependencies with the same Python interpreter:\n"
            f'  python -m pip install -r "{ROOT / "requirements.txt"}"'
        ) from error

    started = time.perf_counter()
    print(f"Input: {source}", flush=True)
    print(f"Tree height: {args.height}", flush=True)

    # Stage all results on the same filesystem. Compression/decompression or
    # validation failures leave existing outputs untouched. Final publication
    # replaces each file atomically, but is not a multi-file transaction.
    with tempfile.TemporaryDirectory(prefix=".biocompress-", dir=source.parent) as folder:
        staged = tuple(Path(folder) / output.name for output in outputs)

        print("[1/3] Preprocessing FASTA/DNA...", flush=True)
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
                file=sys.stderr,
                flush=True,
            )

        print("[2/3] Compressing...", flush=True)
        compressor = Compressor(content, args.height)
        with staged[1].open("w", encoding="ascii") as encoded:
            writer = RunWriter(encoded)
            position = 0
            with tqdm(
                total=len(content), desc="Compressing", unit="bases",
                file=sys.stderr, disable=not sys.stderr.isatty(),
            ) as progress:
                while position < len(content):
                    segment = compressor.process(position)
                    writer.encode(segment)
                    position += segment.length
                    progress.update(segment.length)
            writer.write_buffer()
        # Release the matching trie before allocating the decoded genome.
        del compressor

        print("[3/3] Decompressing and verifying...", flush=True)
        decoder = Decompressor(staged[1].read_text(encoding="ascii"))
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

        # Recheck before publication in case an output appeared during the run.
        # Do not run concurrent pipelines targeting the same output filenames.
        check_outputs(source, outputs, args.force)
        for temporary, destination in zip(staged, outputs):
            os.replace(temporary, destination)

    print(f"Completed in {time.perf_counter() - started:.2f} seconds.")
    print(f"Verified: decoded sequence exactly matches all {len(content):,} normalized bases.")
    for label, output in zip(("Preprocessed", "Encoded", "Decoded"), outputs):
        print(f"{label}: {output} ({output.stat().st_size:,} bytes)")
    print("The source file is unchanged. Encoded output is a text bitstream, not packed binary.")
    print("Decoded output contains normalized DNA only, without FASTA headers or line wrapping.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess, compress, decompress, and verify DNA in one command.",
        epilog=(
            "Outputs are saved beside the input within this project workspace. "
            "FASTA records are concatenated and metadata is omitted. "
            "Install dependencies with: python -m pip install psutil tqdm"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT,
        help="FASTA (.fna/.fa/.fasta) or plain DNA file; relative paths use the current directory",
    )
    parser.add_argument("--height", type=positive_integer, default=HEIGHT,
                        help="Matching trie height")
    parser.add_argument("--force", action="store_true",
                        help="Replace generated output files, never the input")
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
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
