import os
import sys
import math
import struct
import array
import paq
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum, IntEnum
import tempfile  # Added for safer temporary file handling

# Constants
PROGNAME = "paqjp_2"
DEFAULT_OPTION = 8
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

def transform_with_prime_xor_every_3_bytes(data, repeat=50):
    transformed = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime / 2 / repeat))
        for _ in range(repeat):
            for i in range(0, len(transformed), 3):
                transformed[i] ^= xor_val
    return bytes(transformed)

def transform_with_pattern(data):
    return bytes([b ^ 0xFF for b in data])

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
            [8, 12, 1, 1], [9, 13, 1, 1], [11, 14, 0, 2], [15, 19, 3, 0],
            [16, 23, 2, 1], [17, 24, 2, 1], [18, 25, 2, 1], [20, 27, 1, 2],
            [21, 28, 1, 2], [22, 29, 1, 2], [26, 30, 0, 3], [31, 33, 4, 0],
            [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1],
            [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2],
            [34, 37, 2, 2], [34, 37, 2, 2], [36, 39, 1, 3], [36, 39, 1, 3],
            [36, 39, 1, 3], [36, 39, 1, 3], [38, 40, 0, 4], [41, 43, 5, 0],
            [42, 45, 4, 1], [42, 45, 4, 1], [44, 47, 3, 2], [44, 47, 3, 2],
            [46, 49, 2, 3], [46, 49, 2, 3], [48, 51, 1, 4], [48, 51, 1, 4],
            [50, 52, 0, 5], [53, 43, 6, 0], [54, 57, 5, 1], [54, 57, 5, 1],
            [56, 59, 4, 2], [56, 59, 4, 2], [58, 61, 3, 3], [58, 61, 3, 3],
            [60, 63, 2, 4], [60, 63, 2, 4], [62, 65, 1, 5], [62, 65, 1, 5],
            [50, 66, 0, 6], [67, 55, 7, 0], [68, 57, 6, 1], [68, 57, 6, 1],
            [70, 73, 5, 2], [70, 73, 5, 2], [72, 75, 4, 3], [72, 75, 4, 3],
            [74, 77, 3, 4], [74, 77, 3, 4], [76, 79, 2, 5], [76, 79, 2, 5],
            [62, 81, 1, 6], [62, 81, 1, 6], [64, 82, 0, 7], [83, 69, 8, 0],
            [84, 71, 7, 1], [84, 71, 7, 1], [86, 73, 6, 2], [86, 73, 6, 2],
            [44, 59, 5, 3], [44, 59, 5, 3], [58, 61, 4, 4], [58, 61, 4, 4],
            [60, 49, 3, 5], [60, 49, 3, 5], [76, 89, 2, 6], [76, 89, 2, 6],
            [78, 91, 1, 7], [78, 91, 1, 7], [80, 92, 0, 8], [93, 69, 9, 0],
            [94, 87, 8, 1], [94, 87, 8, 1], [96, 45, 7, 2], [96, 45, 7, 2],
            [48, 99, 2, 7], [48, 99, 2, 7], [88, 101, 1, 8], [88, 101, 1, 8],
            [80, 102, 0, 9], [103, 69, 10, 0], [104, 87, 9, 1], [104, 87, 9, 1],
            [106, 57, 8, 2], [106, 57, 8, 2], [62, 109, 2, 8], [62, 109, 2, 8],
            [88, 111, 1, 9], [88, 111, 1, 9], [80, 112, 0, 10], [113, 85, 11, 0],
            [114, 87, 10, 1], [114, 87, 10, 1], [116, 57, 9, 2], [116, 57, 9, 2],
            [62, 119, 2, 9], [62, 119, 2, 9], [88, 121, 1, 10], [88, 121, 1, 10],
            [90, 122, 0, 11], [123, 85, 12, 0], [124, 97, 11, 1], [124, 97, 11, 1],
            [126, 57, 10, 2], [126, 57, 10, 2], [62, 129, 2, 10], [62, 129, 2, 10],
            [98, 131, 1, 11], [98, 131, 1, 11], [90, 132, 0, 12], [133, 85, 13, 0],
            [134, 97, 12, 1], [134, 97, 12, 1], [136, 57, 11, 2], [136, 57, 11, 2],
            [62, 139, 2, 11], [62, 139, 2, 11], [98, 141, 1, 12], [98, 141, 1, 12],
            [90, 142, 0, 13], [143, 95, 14, 0], [144, 97, 13, 1], [144, 97, 13, 1],
            [68, 57, 12, 2], [68, 57, 12, 2], [62, 81, 2, 12], [62, 81, 2, 12],
            [98, 147, 1, 13], [98, 147, 1, 13], [100, 148, 0, 14], [149, 95, 15, 0],
            [150, 107, 14, 1], [150, 107, 14, 1], [108, 151, 1, 14], [108, 151, 1, 14],
            [100, 152, 0, 15], [153, 95, 16, 0], [154, 107, 15, 1], [108, 155, 1, 15],
            [100, 156, 0, 16], [157, 95, 17, 0], [158, 107, 16, 1], [108, 159, 1, 16],
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

class Mixer:
    def __init__(self, inputs: int, contexts: int, rate: int = 8):
        self.inputs = inputs
        self.contexts = contexts
        self.rate = rate
        self.weights = [0] * (inputs * contexts)
        self.context = [0] * contexts
        self.input_values = [0] * inputs
    
    def set(self, cx: int, val: int):
        self.context[cx] = val
    
    def set_input(self, index: int, value: int):
        self.input_values[index] = value
    
    def update(self, bit: int):
        for cx in range(self.contexts):
            idx = self.context[cx] * self.inputs
            err = ((bit << 12) - self.prediction()) * self.rate >> 4
            for i in range(self.inputs):
                self.weights[idx + i] += (self.input_values[i] * err + (1 << 15)) >> 16
    
    def prediction(self) -> int:
        total = 0
        for cx in range(self.contexts):
            idx = self.context[cx] * self.inputs
            for i in range(self.inputs):
                total += self.weights[idx + i] * self.input_values[i]
        return squash(total >> 8)

class Predictor:
    def __init__(self):
        self.pr = 2048  # Initial prediction (P(1) = 0.5)
        self.cm = ContextMap(MEM, 8)
        self.mixer = Mixer(256, 8)
        self.state = 0
        self.byte_history = []
    
    def p(self) -> int:
        return self.pr
    
    def update(self, bit: int):
        global c0, c4, bpos, pos
        for cx in range(8):
            self.cm.update(cx, bit)
        
        for i in range(256):
            self.mixer.set_input(i, (self.cm.get(i >> 5) >> (7 - (i & 7))) & 1)
        
        self.pr = self.mixer.prediction()
        self.state = nex.nex(self.state, bit)
        
        if bpos == 7:
            self.byte_history.insert(0, c0)
            if len(self.byte_history) > 4:
                self.byte_history.pop()
            c0 = 1
            c4 = (c4 << 8) | self.byte_history[0]
            pos += 1
            bpos = 0
        else:
            bpos += 1
            c0 = (c0 << 1) | bit

class Encoder:
    def __init__(self, mode: Mode, filename: str):
        self.mode = mode
        self.filename = filename
        self.file = open(filename, "wb+" if mode == Mode.COMPRESS else "rb")
        self.x1 = 0
        self.x2 = 0xFFFFFFFF
        self.x = 0
        self.predictor = Predictor()
        
        if mode == Mode.DECOMPRESS and level > 0:
            bytes_read = self.file.read(4)
            if len(bytes_read) == 4:
                self.x = struct.unpack('>I', bytes_read)[0]
            else:
                self.x = 0
    
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
            if self.x1 > 0xFFFFFFFF:
                self.x1 &= 0xFFFFFFFF
            try:
                self.file.write(struct.pack('>I', self.x1))
            except struct.error as e:
                raise ValueError(f"Error in flush: {str(e)}")
    
    def compress(self, c: int):
        if self.mode != Mode.COMPRESS:
            raise ValueError("Not in compress mode")
        
        if level == 0:
            self.file.write(bytes([c & 0xFF]))
        else:
            for i in range(7, -1, -1):
                self.code((c >> i) & 1)
    
    def decompress(self) -> int:
        if self.mode != Mode.DECOMPRESS:
            raise ValueError("Not in decompress mode")
        
        if level == 0:
            byte = self.file.read(1)
            return ord(byte) if byte else -1
        else:
            c = 0
            for _ in range(8):
                bit = self.code()
                if bit == -1:
                    return -1
                c = (c << 1) | bit
            return c & 0xFF
    
    def code(self, bit: int = None) -> int:
        p = self.predictor.p()
        assert 0 <= p < 4096
        p += p < 2048
        
        xmid = self.x1 + ((self.x2 - self.x1) >> 12) * p + (((self.x2 - self.x1) & 0xFFF) * p >> 12)
        assert self.x1 <= xmid <= self.x2, f"x1={self.x1}, xmid={xmid}, x2={self.x2}"
        
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
                if self.x1 > 0xFFFFFFFF:
                    self.x1 &= 0xFFFFFFFF
                try:
                    self.file.write(struct.pack('>B', (self.x1 >> 24) & 0xFF))
                except struct.error as e:
                    raise ValueError(f"Error writing byte: {str(e)}")
            
            self.x1 <<= 8
            self.x2 = (self.x2 << 8) | 0xFF
            
            if self.mode == Mode.DECOMPRESS:
                byte = self.file.read(1)
                if not byte:
                    return -1
                self.x = (self.x << 8) | ord(byte)
        
        return y

class PAQCompressor:
    def __init__(self):
        self.models = []
        self.mixer = Mixer(256, 8)
    
    def add_model(self, model):
        self.models.append(model)
    
    def compress(self, data: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.paq') as temp_file:
            temp_filename = temp_file.name
        try:
            with Encoder(Mode.COMPRESS, temp_filename) as enc:
                for byte in data:
                    enc.compress(byte & 0xFF)
                enc.flush()
            
            with open(temp_filename, "rb") as f:
                compressed = f.read()
            return compressed
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    def decompress(self, data: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.paq') as temp_file:
            temp_filename = temp_file.name
        try:
            with open(temp_filename, "wb") as f:
                f.write(data)
            
            result = bytearray()
            with Encoder(Mode.DECOMPRESS, temp_filename) as enc:
                while True:
                    byte = enc.decompress()
                    if byte == -1:
                        break
                    result.append(byte & 0xFF)
            return bytes(result)
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

class SmartCompressor:
    def __init__(self):
        self.compressor = PAQCompressor()
    
    def reversible_transform(self, data):
        return transform_with_prime_xor_every_3_bytes(data)
    
    def reverse_reversible_transform(self, data):
        return transform_with_prime_xor_every_3_bytes(data)
    
    def compress_with_best_method(self, data):
        # Method 1: Prime XOR transformation + PAQ8F
        transformed_smart = self.reversible_transform(data)
        compressed_smart = self.compressor.compress(transformed_smart)
        
        # Method 2: Pattern transformation + PAQ8F
        transformed_simple = transform_with_pattern(data)
        compressed_simple = self.compressor.compress(transformed_simple)
        
        # Method 3: zlib compression
        try:
            compressed_zlib = paq.compress(data)  # Maximum compression
        except zlib.error as e:
            print(f"zlib compression error: {str(e)}")
            compressed_zlib = b''  # Fallback to empty bytes if zlib fails
        
        # Choose the smallest output
        compressed_methods = [
            (b'\x01', compressed_smart, "Prime XOR + PAQ8F"),
            (b'\x02', compressed_simple, "Pattern + PAQ8F"),
            (b'\x03', compressed_zlib, "zlib")
        ]
        method_marker, compressed_data, method_name = min(
            compressed_methods, key=lambda x: len(x[1]) if x[1] else float('inf')
        )
        
        print(f"Selected compression method: {method_name}")
        return method_marker + compressed_data
    
    def decompress_with_best_method(self, data):
        if len(data) < 1:
            return b''
        
        method_marker = data[0]
        compressed_data = data[1:]
        
        if method_marker == 1:
            decompressed = self.compressor.decompress(compressed_data)
            return self.reverse_reversible_transform(decompressed)
        elif method_marker == 2:
            decompressed = self.compressor.decompress(compressed_data)
            return transform_with_pattern(decompressed)
        elif method_marker == 3:
            try:
                return paq.decompress(compressed_data)
            except zlib.error as e:
                raise ValueError(f"zlib decompression error: {str(e)}")
        else:
            raise ValueError(f"Unknown compression method marker: {method_marker}")

def get_user_input():
    print("\nPAQJP_2 and paq Compressor/Decompressor")
    print("Created by Jurijus Pacalovas")
    print("=" * 40)
    
    while True:
        action = input("Choose operation:\n1. Compress\n2. Decompress\nEnter 1 or 2: ").strip()
        if action in ('1', '2'):
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    while True:
        input_file = input("Enter input file name: ").strip()
        if os.path.isfile(input_file):
            break
        print(f"Error: File '{input_file}' not found. Please try again.")
    
    output_file = input("Enter output file name: ").strip()
    
    return action, input_file, output_file

def main():
    action, input_file, output_file = get_user_input()
    
    sc = SmartCompressor()
    
    try:
        if action == '1':
            print("\nCompressing...")
            with open(input_file, "rb") as f:
                input_data = f.read()
            
            compressed = sc.compress_with_best_method(input_data)
            
            with open(output_file, "wb") as f_out:
                f_out.write(compressed)
            
            original_size = len(input_data)
            compressed_size = len(compressed)
            ratio = (compressed_size / original_size) * 100 if original_size > 0 else 0
            
            print("\nCompression successful!")
            print(f"Original size:   {original_size} bytes")
            print(f"Compressed size: {compressed_size} bytes")
            print(f"Compression ratio: {ratio:.2f}%")
            print(f"Output saved to: {output_file}")
        
        else:
            print("\nDecompressing...")
            with open(input_file, "rb") as f:
                input_data = f.read()
            
            decompressed = sc.decompress_with_best_method(input_data)
            
            with open(output_file, "wb") as f_out:
                f_out.write(decompressed)
            
            print("\nDecompression successful!")
            print(f"Decompressed size: {len(decompressed)} bytes")
            print(f"Output saved to: {output_file}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()