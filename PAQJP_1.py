import os
import sys
import math
import struct
import array
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum, IntEnum
import paq

# Constants
PROGNAME = "paq8f"
DEFAULT_OPTION = 9
MEM = 0x10000 << DEFAULT_OPTION

# Global variables
level = DEFAULT_OPTION
y = 0  # Last bit, 0 or 1
c0 = 1  # Last 0-7 bits of partial byte with leading 1 bit
c4 = 0  # Last 4 whole bytes, packed
bpos = 0  # bits in c0 (0 to 7)
pos = 0  # Number of input bytes in buf (not wrapped)

# Prime numbers for transformation
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]

class Filetype(Enum):
    DEFAULT = 0
    JPEG = 1
    EXE = 2
    TEXT = 3

class Mode(Enum):
    COMPRESS = 0
    DECOMPRESS = 1

# Helper classes
class String:
    def __init__(self, s: str = ""):
        self.data = bytearray(s.encode('utf-8'))
    
    def resize(self, new_size: int):
        if new_size > len(self.data):
            self.data += bytearray(new_size - len(self.data))
        else:
            self.data = self.data[:new_size]
    
    def size(self) -> int:
        return len(self.data)
    
    def c_str(self) -> str:
        return self.data.decode('utf-8')
    
    def __iadd__(self, s: str):
        self.data += s.encode('utf-8')
        return self
    
    def __getitem__(self, index: int) -> int:
        return self.data[index]
    
    def __setitem__(self, index: int, value: int):
        self.data[index] = value
    
    def __str__(self) -> str:
        return self.data.decode('utf-8')

class Array:
    def __init__(self, size: int = 0, initial_value: int = 0):
        self.data = array.array('B', [initial_value] * size)
    
    def resize(self, new_size: int):
        if new_size > len(self.data):
            self.data.extend([0] * (new_size - len(self.data)))
        else:
            self.data = self.data[:new_size]
    
    def size(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> int:
        return self.data[index]
    
    def __setitem__(self, index: int, value: int):
        self.data[index] = value
    
    def __len__(self) -> int:
        return len(self.data)

class Buf:
    def __init__(self, size: int = 0):
        self.size_ = size
        self.data = Array(size)
        self.pos = 0
    
    def setsize(self, size: int):
        if size > 0 and (size & (size - 1)) == 0:
            self.size_ = size
            self.data.resize(size)
    
    def __getitem__(self, index: int) -> int:
        return self.data[index & (self.size_ - 1)]
    
    def __call__(self, i: int) -> int:
        assert i > 0
        return self.data[(self.pos - i) & (self.size_ - 1)]
    
    def size(self) -> int:
        return self.size_

# Global buffer
buf = Buf()

# Transformation functions
def transform_with_prime_xor_every_3_bytes(data, repeat=50):
    transformed = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime / 2 / repeat))
        for _ in range(repeat):
            for i in range(0, len(transformed), 3):
                transformed[i] ^= xor_val
    return bytes(transformed)

def transform_with_pattern(data):
    # Simple reversible pattern: invert all bits
    return bytes([b ^ 0xFF for b in data])

# Helper functions
def quit(message: str = None):
    if message:
        print(message)
    sys.exit(1)

def equals(a: str, b: str) -> bool:
    return a.lower() == b.lower()

def ilog(x: int) -> int:
    if x < 0:
        return 0
    l = 0
    while x > 0:
        x >>= 1
        l += 1
    return l

def squash(d: int) -> int:
    t = [1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,
         1546,2047,2549,2994,3348,3607,3785,3901,3975,4022,
         4050,4068,4079,4085,4089,4092,4093,4094]
    if d > 2047:
        return 4095
    if d < -2047:
        return 0
    w = d & 127
    d = (d >> 7) + 16
    return (t[d] * (128 - w) + t[d + 1] * w + 64) >> 7

def stretch(p: int) -> int:
    t = Array(4096)
    pi = 0
    for x in range(-2047, 2048):
        i = squash(x)
        for j in range(pi, i + 1):
            t[j] = x
        pi = i + 1
    t[4095] = 2047
    return t[p]

def hash(*args: int) -> int:
    h = (args[0] * 200002979 + args[1] * 30005491 + 
         (args[2] if len(args) > 2 else 0xffffffff) * 50004239 + 
         (args[3] if len(args) > 3 else 0xffffffff) * 70004807 + 
         (args[4] if len(args) > 4 else 0xffffffff) * 110002499)
    return h ^ (h >> 9) ^ (args[0] >> 2) ^ (args[1] >> 3) ^ (
        (args[2] if len(args) > 2 else 0) >> 4) ^ (
        (args[3] if len(args) > 3 else 0) >> 5) ^ (
        (args[4] if len(args) > 4 else 0) >> 6)

# State table
class StateTable:
    def __init__(self):
        self.table = [
    [1, 2, 0, 0], [3, 5, 1, 0], [4, 6, 0, 1], [7, 10, 2, 0],
    # ... (rest of the state table entries)
    [100, 160, 0, 17], [161, 105, 18, 0], [162, 107, 17, 1], [108, 163, 1, 17],
    [110, 164, 0, 18], [165, 105, 19, 0], [166, 117, 18, 1], [118, 167, 1, 18],
    [110, 168, 0, 19], [169, 105, 20, 0], [170, 117, 19, 1], [118, 171, 1, 19],
    [110, 172, 0, 20], [173, 105, 21, 0], [174, 117, 20, 1], [118, 175, 1, 20],
    [110, 176, 0, 21], [177, 105, 22, 0], [178, 117, 21, 1], [118, 179, 1, 21],
    [110, 180, 0, 22], [181, 115, 23, 0], [182, 117, 22, 1], [118, 183, 1, 22],
    [120, 184, 0, 23], [185, 115, 24, 0], [186, 127, 23, 1], [128, 187, 1, 23],
    [120, 188, 0, 24], [189, 115, 25, 0], [190, 127, 24, 1], [128, 191, 1, 24],
    [120, 192, 0, 25], [193, 115, 26, 0], [194, 127, 25, 1], [128, 195, 1, 25],
    [120, 196, 0, 26], [197, 115, 27, 0], [198, 127, 26, 1], [128, 199, 1, 26],
    [120, 200, 0, 27], [201, 115, 28, 0], [202, 127, 27, 1], [128, 203, 1, 27],
    [120, 204, 0, 28], [205, 115, 29, 0], [206, 127, 28, 1], [128, 207, 1, 28],
    [120, 208, 0, 29], [209, 125, 30, 0], [210, 127, 29, 1], [128, 211, 1, 29],
    [130, 212, 0, 30], [213, 125, 31, 0], [214, 137, 30, 1], [138, 215, 1, 30],
    [130, 216, 0, 31], [217, 125, 32, 0], [218, 137, 31, 1], [138, 219, 1, 31],
    [130, 220, 0, 32], [221, 125, 33, 0], [222, 137, 32, 1], [138, 223, 1, 32],
    [130, 224, 0, 33], [225, 125, 34, 0], [226, 137, 33, 1], [138, 227, 1, 33],
    [130, 228, 0, 34], [229, 125, 35, 0], [230, 137, 34, 1], [138, 231, 1, 34],
    [130, 232, 0, 35], [233, 125, 36, 0], [234, 137, 35, 1], [138, 235, 1, 35],
    [130, 236, 0, 36], [237, 125, 37, 0], [238, 137, 36, 1], [138, 239, 1, 36],
    [130, 240, 0, 37], [241, 125, 38, 0], [242, 137, 37, 1], [138, 243, 1, 37],
    [130, 244, 0, 38], [245, 135, 39, 0], [246, 137, 38, 1], [138, 247, 1, 38],
    [140, 248, 0, 39], [249, 135, 40, 0], [250, 69, 39, 1], [80, 251, 1, 39],
    [140, 252, 0, 40], [249, 135, 41, 0], [250, 69, 40, 1], [80, 251, 1, 40],
    [140, 252, 0, 41]
]
    def nex(self, state: int, sel: int) -> int:
        return self.table[state][sel]

nex = StateTable()

# Context Map
class ContextMap:
    def __init__(self, size: int, contexts: int):
        self.size = size
        self.contexts = contexts
        self.table = [0] * (size * contexts)
        self.hash = [0] * contexts
    
    def set(self, cx: int, val: int):
        self.hash[cx] = val
    
    def get(self, cx: int) -> int:
        return self.table[(self.hash[cx] % self.size) * self.contexts + cx]
    
    def update(self, cx: int, bit: int):
        idx = (self.hash[cx] % self.size) * self.contexts + cx
        self.table[idx] = (self.table[idx] << 1) | bit

# Mixer
class Mixer:
    def __init__(self, inputs: int, contexts: int, rate: int = 8):
        self.inputs = inputs
        self.contexts = contexts
        self.rate = rate
        self.weights = [0] * (inputs * contexts)
        self.context = [0] * contexts
    
    def set(self, cx: int, val: int):
        self.context[cx] = val
    
    def update(self, bit: int):
        for cx in range(self.contexts):
            idx = self.context[cx] * self.inputs
            err = (bit << 12) - self.prediction
            for i in range(self.inputs):
                self.weights[idx + i] += (self.inputs[i] * err * self.rate) >> 16
    
    @property
    def prediction(self) -> int:
        total = 0
        for cx in range(self.contexts):
            idx = self.context[cx] * self.inputs
            for i in range(self.inputs):
                total += self.weights[idx + i] * self.inputs[i]
        return squash(total >> 8)

# Predictor
class Predictor:
    def __init__(self):
        self.pr = 2048  # Initial prediction (P(1) = 0.5)
        self.cm = ContextMap(MEM, 8)
        self.mixer = Mixer(256, 8)
        self.state = 0
    
    def p(self) -> int:
        return self.pr
    
    def update(self, bit: int):
        # Update context maps
        for cx in range(8):
            self.cm.update(cx, bit)
        
        # Update mixer
        self.mixer.update(bit)
        
        # Update prediction
        self.pr = self.mixer.prediction
        
        # Update state
        self.state = nex.nex(self.state, bit)

# Encoder
class Encoder:
    def __init__(self, mode: Mode, filename: str):
        self.mode = mode
        self.filename = filename
        self.file = open(filename, "wb+" if mode == Mode.COMPRESS else "rb+")
        self.x1 = 0
        self.x2 = 0xFFFFFFFF
        self.x = 0
        self.predictor = Predictor()
        
        if mode == Mode.DECOMPRESS and level > 0:
            # Read first 4 bytes for decompression
            self.x = struct.unpack('>I', self.file.read(4))[0]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def size(self) -> int:
        return os.path.getsize(self.filename)
    
    def flush(self):
        if self.mode == Mode.COMPRESS and level > 0:
            self.file.write(struct.pack('>B', self.x1 >> 24))
    
    def compress(self, c: int):
        if self.mode != Mode.COMPRESS:
            raise ValueError("Not in compress mode")
        
        if level == 0:
            self.file.write(bytes([c]))
        else:
            for i in range(7, -1, -1):
                self.code((c >> i) & 1)
    
    def decompress(self) -> int:
        if self.mode == Mode.COMPRESS:
            raise ValueError("Not in decompress mode")
        
        if level == 0:
            return ord(self.file.read(1))
        else:
            c = 0
            for _ in range(8):
                c += c + self.code()
            return c
    
    def code(self, bit: int = None) -> int:
        p = self.predictor.p()
        assert 0 <= p < 4096
        p += p < 2048
        
        xmid = self.x1 + ((self.x2 - self.x1) >> 12) * p + (((self.x2 - self.x1) & 0xFFF) * p >> 12)
        assert self.x1 <= xmid < self.x2
        
        if self.mode == Mode.DECOMPRESS:
            y = 1 if self.x <= xmid else 0
        else:
            y = bit
        
        if y:
            self.x2 = xmid
        else:
            self.x1 = xmid + 1
        
        self.predictor.update(y)
        
        while ((self.x1 ^ self.x2) & 0xFF000000) == 0:
            if self.mode == Mode.COMPRESS:
                self.file.write(struct.pack('>B', self.x2 >> 24))
            
            self.x1 <<= 8
            self.x2 = (self.x2 << 8) + 255
            
            if self.mode == Mode.DECOMPRESS:
                byte = self.file.read(1)
                if not byte:
                    byte = b'\x00'  # Handle EOF
                self.x = (self.x << 8) | ord(byte)
        
        return y

# PAQ Compressor
class PAQCompressor:
    def __init__(self):
        self.models = []
        self.mixer = Mixer(256, 8)
    
    def add_model(self, model):
        self.models.append(model)
    
    def compress(self, data: bytes) -> bytes:
        with contextlib.closing(Encoder(Mode.COMPRESS, "temp.paq")) as enc:
            for byte in data:
                enc.compress(byte)
            enc.flush()
        
        with open("temp.paq", "rb") as f:
            compressed = f.read()
        
        os.remove("temp.paq")
        return compressed
    
    def decompress(self, data: bytes) -> bytes:
        with open("temp.paq", "wb") as f:
            f.write(data)
        
        result = bytearray()
        with contextlib.closing(Encoder(Mode.DECOMPRESS, "temp.paq")) as enc:
            while True:
                try:
                    byte = enc.decompress()
                    result.append(byte)
                except:
                    break
        
        os.remove("temp.paq")
        return bytes(result)

# Smart Compressor with transformations
class SmartCompressor:
    def __init__(self):
        self.paq = PAQCompressor()
    
    def reversible_transform(self, data: bytes) -> bytes:
        return transform_with_prime_xor_every_3_bytes(data)
    
    def reverse_reversible_transform(self, data: bytes) -> bytes:
        return transform_with_prime_xor_every_3_bytes(data)
    
    def compress_with_best_method(self, data: bytes) -> bytes:
        # Try both transformation methods
        transformed_smart = self.reversible_transform(data)
        compressed_smart = self.paq.compress(transformed_smart)
        
        transformed_simple = transform_with_pattern(data)
        compressed_simple = self.paq.compress(transformed_simple)
        
        # Choose the best compression
        if len(compressed_smart) < len(compressed_simple):
            return b'\x01' + compressed_smart  # Marker for smart transform
        else:
            return b'\x02' + compressed_simple  # Marker for simple transform
    
    def decompress_with_best_method(self, data: bytes) -> bytes:
        if len(data) < 1:
            return b''
        
        method_marker = data[0]
        compressed_data = data[1:]
        
        decompressed = self.paq.decompress(compressed_data)
        
        if method_marker == 1:
            return self.reverse_reversible_transform(decompressed)
        elif method_marker == 2:
            return transform_with_pattern(decompressed)
        else:
            raise ValueError("Unknown compression method marker")

    
    def paq_compress(self, data):
        # Placeholder for actual PAQ compression
        return paq.compress(data)  # In a real implementation, this would use the PAQ algorithm
    
    def paq_decompress(self, data):
        # Placeholder for actual PAQ decompression
        return paq.decompress(data)  # In a real implementation, this would use the PAQ algorithm

# Models and compression classes would go here
# (Predictor, Encoder, etc. from the first code)



def main():
    print("Created by Jurijus Pacalovas.")
    action = input("Choose action - Compress (1) or Extract (2): ").strip()
    if action not in ("1", "2"):
        print("Invalid action. Exiting.")
        return

    input_file = input("Input file name: ").strip()
    output_file = input("Output file name: ").strip()

    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    sc = SmartCompressor()

    if action == "1":
        # Compression
        with open(input_file, "rb") as f:
            input_data = f.read()
        
        compressed = sc.compress_with_best_method(input_data)
        
        with open(output_file, "wb") as f_out:
            f_out.write(compressed)
        
        print(f"Compression successful. Output saved to {output_file}. Size: {len(compressed)} bytes")
    
    else:
        # Extraction
        with open(input_file, "rb") as f:
            input_data = f.read()
        
        try:
            decompressed = sc.decompress_with_best_method(input_data)
            
            with open(output_file, "wb") as f_out:
                f_out.write(decompressed)
            
            print(f"Decompression successful. Output saved to {output_file}.")
        except Exception as e:
            print("Error during decompression:", e)

if __name__ == "__main__":
    main()