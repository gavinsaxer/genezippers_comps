from config import DNA_FILE, DNA_FILE_PATH, HEIGHT, COMPLEMENT_TABLE
from tqdm import tqdm


PAIR_TO_BASE = ("G", "T", "C", "A")
COMPLEMENT_BASE = {base: base.translate(COMPLEMENT_TABLE) for base in PAIR_TO_BASE}


class Decompressor:
    def __init__(self, bits):
        self.bits = bits
        self.position = 0
        self.output = []
        self.literal_run = True

    def read_bit(self):
        if self.position >= len(self.bits):
            raise ValueError("Unexpected end of encoded input")
        bit = self.bits[self.position]
        if bit not in ("0", "1"):
            raise ValueError(f"Invalid bit at offset {self.position}")
        self.position += 1
        return 1 if bit == "1" else 0

    def read_number(self):
        value = 0
        current, following = 1, 2
        previous = 0
        while True:
            bit = self.read_bit()
            if bit and previous:
                return value
            if bit:
                value += current
            previous = bit
            current, following = following, current + following

    def read_position(self, window):
        if window < 2:
            raise ValueError("Pointer window has no usable position bits")
        width = (window - 1).bit_length()
        binary = 0
        fibonacci = 0
        current, following = 1, 2
        previous = 0
        for offset in range(width):
            bit = self.read_bit()
            binary = binary * 2 + bit
            if bit and previous:
                marker = self.read_bit()
                if marker == 0:
                    return fibonacci
                for _ in range(width - offset - 1):
                    binary = binary * 2 + self.read_bit()
                return binary
            if bit:
                fibonacci += current
            previous = bit
            current, following = following, current + following
        return binary

    def read_bases(self, count):
        if count > (len(self.bits) - self.position) // 2:
            raise ValueError("Truncated literal run")
        for _ in range(count):
            pair = self.read_bit() * 2 + self.read_bit()
            self.output.append(PAIR_TO_BASE[pair])

    def read_factors(self, count):
        for _ in range(count):
            length = self.read_number()
            complement = self.read_bit() == 1
            position = self.read_position(len(self.output))
            if position < 1:
                raise ValueError("Pointer positions must be positive")
            source = position - 1
            if complement:
                source = len(self.output) - source
            if not 0 <= source < len(self.output):
                raise ValueError("Pointer must reference earlier decoded bases")
            for offset in range(length):
                base = self.output[source + offset]
                self.output.append(COMPLEMENT_BASE[base] if complement else base)

    def decode_run(self):
        count = self.read_number()
        if self.literal_run:
            self.read_bases(count)
        else:
            self.read_factors(count)
        self.literal_run = not self.literal_run

    def decode(self):
        while self.position < len(self.bits):
            self.decode_run()
        return "".join(self.output)


def decompress(bits):
    return Decompressor(bits).decode()


def main():
    prefix = f"{DNA_FILE_PATH}{DNA_FILE}_{HEIGHT}"
    with open(prefix + "_encoded.txt", "r", encoding="utf-8") as source:
        decoder = Decompressor(source.read())
    with tqdm(total=len(decoder.bits), desc="Decompressing", unit="bits") as pbar:
        while decoder.position < len(decoder.bits):
            previous = decoder.position
            decoder.decode_run()
            pbar.update(decoder.position - previous)
    with open(prefix + "_decoded.txt", "w", encoding="utf-8") as output:
        output.write("".join(decoder.output))


if __name__ == "__main__":
    main()
