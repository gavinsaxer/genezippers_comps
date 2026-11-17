"""Normalize FASTA or plain DNA without modifying the source file.

FASTA headers/comments and whitespace are omitted; records are concatenated.
Ambiguous bases are rejected unless dropping non-ACGT symbols is requested.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import List


DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "E_Coli.fna"


@dataclass(frozen=True)
class PreprocessResult:
    content: str
    dropped_symbols: int
    records: int


def read_sequence(source: Path, *, drop_non_acgt: bool = False) -> PreprocessResult:
    """Read sequence lines only; never treat FASTA descriptions as DNA."""
    chunks: List[str] = []
    dropped = 0
    records = 0
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                records += 1
                continue
            sequence = "".join(line.split()).upper()
            invalid = set(sequence) - set("ACGT")
            if invalid and not drop_non_acgt:
                symbols = ", ".join(repr(symbol) for symbol in sorted(invalid))
                raise ValueError(
                    f"{source.name}, line {line_number}: unsupported DNA symbols: "
                    f"{symbols}. Only A/C/G/T are supported. Use --drop-non-acgt "
                    "only if removing these symbols is acceptable."
                )
            if invalid:
                filtered = "".join(base for base in sequence if base in "ACGT")
                dropped += len(sequence) - len(filtered)
                sequence = filtered
            chunks.append(sequence)
    content = "".join(chunks)
    if not content:
        raise ValueError(f"No A/C/G/T sequence found in {source}")
    return PreprocessResult(content, dropped, records)


def preprocess(
    source: Path,
    output: Path,
    *,
    drop_non_acgt: bool = False,
    force: bool = False,
) -> PreprocessResult:
    """Validate before writing; replace output atomically and preserve source."""
    source = source.resolve()
    output = output.resolve()
    if source == output or (output.exists() and source.samefile(output)):
        raise ValueError("Preprocessed output must not overwrite the input file")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}; use --force")
    result = read_sequence(source, drop_non_acgt=drop_non_acgt)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=output.parent,
            prefix=".preprocess-", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(result.content)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--force", action="store_true", help="Replace existing .txt output")
    parser.add_argument("--drop-non-acgt", action="store_true",
                        help="Discard ambiguous/non-ACGT symbols (lossy)")
    args = parser.parse_args()
    source = args.input.resolve()
    output = source.with_suffix(".txt")
    try:
        result = preprocess(source, output, drop_non_acgt=args.drop_non_acgt,
                            force=args.force)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(result.content):,} bases to {output}")
    if result.dropped_symbols:
        print(f"WARNING: dropped {result.dropped_symbols:,} non-ACGT symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
