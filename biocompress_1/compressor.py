from AGCT_tree import create_tree, find_factor
from config import HEIGHT, DNA_FILE, CONTENT, DNA_FILE_PATH, COMPLEMENT_TABLE
from converter import base_to_binary, encode_factor, encode_fibonacci
from typing import Optional, List, Tuple, Union
from tqdm import tqdm
import json, sys, time, psutil, os


open(DNA_FILE_PATH + DNA_FILE + "_" + str(HEIGHT) + "_encoded.txt", "w").close()
output_file = open(DNA_FILE_PATH + DNA_FILE + "_" + str(HEIGHT) + "_encoded.txt", "a", encoding="utf-8")
start_time = time.time()
TREE = create_tree(HEIGHT)
tree_created_time = time.time()


def get_memory_usage() -> float:
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size: actual RAM used
    mem_mb = mem_bytes / (1024 ** 2)
    return round(mem_mb, 2)

tree_memory = get_memory_usage()


def extended_search(i: int, position: int, kind: str) -> int:
    """Extends the search for matching bases beyond the initial tree search."""
    i += HEIGHT
    position += HEIGHT
    add_length = 0
    if i > len(CONTENT) - 1:
        return add_length
    if kind == "factor":
        while i < len(CONTENT) and CONTENT[i] == CONTENT[position]:
            add_length += 1
            i += 1
            position += 1
    elif kind == "palindrome":
        while i < len(CONTENT) and CONTENT[i].translate(COMPLEMENT_TABLE) == CONTENT[position]:
            add_length += 1
            i += 1
            position += 1
    return add_length


def longest_factor_or_palindrome(i: int) -> Tuple[Optional[int], int, Optional[str]]:
    """Finds the longest factor or palindrome starting at a given position."""
    string = CONTENT[i:i + HEIGHT]
    palindrome = string.translate(COMPLEMENT_TABLE)
    factor_position = find_factor(string, TREE)

    if factor_position[1] == HEIGHT:
        add_length = 0
        position_number = 0
        position_number_temp = 0
        if factor_position[0]:
            for position in factor_position[0]:
                add_length_temp = extended_search(i, position, "factor")
                if add_length_temp > add_length:
                    add_length = add_length_temp
                    position_number = position_number_temp
                position_number_temp += 1
            factor_position = (factor_position[0][position_number], factor_position[1] + add_length)

    palindrome_position = find_factor(palindrome, TREE)
    if palindrome_position[1] == HEIGHT:
        add_length = 0
        position_number = 0
        position_number_temp = 0
        if palindrome_position[0]:
            for position in palindrome_position[0]:
                add_length_temp = extended_search(i, position, "palindrome")
                if add_length_temp > add_length:
                    add_length = add_length_temp
                    position_number = position_number_temp
                position_number_temp += 1
            palindrome_position = (i - palindrome_position[0][position_number], palindrome_position[1] + add_length)
    elif palindrome_position[1]:
        palindrome_position = (i - palindrome_position[0][0], palindrome_position[1])

    if factor_position[1] and palindrome_position[1]:
        if factor_position[1] >= palindrome_position[1]:
            return factor_position[0], factor_position[1], "factor"
        else:
            return palindrome_position[0], palindrome_position[1], "palindrome"
    return None, 0, None


def process(i: int) -> Tuple[Union[str, int], str, int]:
    """Processes a segment of the DNA sequence to find and encode the longest factor or palindrome."""
    segment = CONTENT[i:i + HEIGHT]
    longest_factor = longest_factor_or_palindrome(i)
    TREE.create_positions(segment, i)

    if longest_factor[0] is not None:
        return encode_factor(longest_factor, i)
    else:
        return base_to_binary(CONTENT[i]), "base", 1


def write_buffer(buffer: List[Tuple[Union[str, int], str, int]]) -> None:
    """Writes the encoded data buffer to the output file."""
    if buffer[0][1] == "base":
        length = sum(item[2] for item in buffer)
        output_file.write(encode_fibonacci(length))
    else:
        output_file.write(encode_fibonacci(len(buffer)))
    for item in buffer:
        output_file.write(item[0])


def encode(processed: Tuple[Union[str, int], str, int], buffer: List[Tuple[Union[str, int], str, int]]) -> List[Tuple[Union[str, int], str, int]]:
    """Manages the buffer of encoded data, writing to the output file when necessary."""
    if not buffer:
        return [processed]
    if (processed[1] == "base" and buffer[0][1] != "base") or (processed[1] != "base" and buffer[0][1] == "base"):
        write_buffer(buffer)
        buffer = []
    buffer.append(processed)
    return buffer


def main() -> None:
    """Main function to orchestrate the compression process."""
    length = len(CONTENT)
    position = 0
    buffer = []
    with tqdm(total=length, desc="Compressing", unit="bytes", file=sys.stderr, leave=True, disable=not sys.stderr.isatty()) as pbar:
        while position < len(CONTENT):
            processed = process(position)
            buffer = encode(processed, buffer)
            position += processed[2]
            pbar.update(processed[2])
    write_buffer(buffer)
    output_file.close()

    finished_time = time.time()
    print(json.dumps({
        "total_compression_time": round(finished_time - start_time, 3),
        "tree_creation_time": round(tree_created_time - start_time, 3),
        "compressor_time": round(finished_time - tree_created_time, 3),
        "tree_memory": tree_memory,
    }))


if __name__ == "__main__":
    main()
