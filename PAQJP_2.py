import os
import sys
import math
import struct
import array
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum, IntEnum
import paq

# Constants
PROGNAME = "paqjp_2"
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

# Transformation functions (defined globally as in original code)
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

# State table (unchanged)
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

# Updated SmartCompressor class with new transformations
class SmartCompressor:
    def __init__(self):
        self.compressor = None
        self.PRIMES = PRIMES  # Use global PRIMES list
    
    def huffman_compress(self, data):
        return self.compress_with_best_method(data)
    
    def huffman_decompress(self, data):
        return self.decompress_with_best_method(data)
    
    # Transformation for Marker 01 (unchanged)
    def transform_with_prime_xor_every_3_bytes(self, data):
        return transform_with_prime_xor_every_3_bytes(data)
    
    def reverse_transform_with_prime_xor_every_3_bytes(self, data):
        return self.transform_with_prime_xor_every_3_bytes(data)  # Self-inverse
    
    # Transformation for Marker 02 (unchanged)
    def transform_with_pattern(self, data):
        return transform_with_pattern(data)
    
    def reverse_transform_with_pattern(self, data):
        return self.transform_with_pattern(data)  # Self-inverse
    
    # New Transformation for Marker 03
    def transform_03(self, data):
        # Compute total subtraction value from primes, similar to transform_with_prime_xor_every_3_bytes
        total_sub = sum(50 * (prime if prime == 2 else math.ceil(prime / 100)) for prime in self.PRIMES) % 256
        transformed = bytearray(data)
        for i in range(0, len(transformed), 3):
            transformed[i] = (transformed[i] - total_sub) % 256  # Subtract modulo 256
        return bytes(transformed)
    
    def reverse_transform_03(self, data):
        total_sub = sum(50 * (prime if prime == 2 else math.ceil(prime / 100)) for prime in self.PRIMES) % 256
        transformed = bytearray(data)
        for i in range(0, len(transformed), 3):
            transformed[i] = (transformed[i] + total_sub) % 256  # Add back modulo 256
        return bytes(transformed)
    
    # New Transformation for Marker 04
    def transform_04(self, data):
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = (transformed[i] - (i % 256)) % 256  # Subtract position mod 256
        return bytes(transformed)
    
    def reverse_transform_04(self, data):
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = (transformed[i] + (i % 256)) % 256  # Add back position mod 256
        return bytes(transformed)
    
    def compress_with_best_method(self, data):
        # Define all transformations with their markers
        transformations = [
            (1, self.transform_with_prime_xor_every_3_bytes),  # Marker 01
            (2, self.transform_with_pattern),                  # Marker 02
            (3, self.transform_03),                           # Marker 03
            (4, self.transform_04)                            # Marker 04
        ]
        best_compressed = None
        best_size = float('inf')
        best_marker = None
        
        # Try each transformation and select the smallest result
        for marker, transform in transformations:
            transformed = transform(data)
            compressed = self.paq_compress(transformed)
            size = len(compressed)
            if size < best_size:
                best_size = size
                best_compressed = compressed
                best_marker = marker
        return bytes([best_marker]) + best_compressed
    
    def decompress_with_best_method(self, data):
        if len(data) < 1:
            return b''
        
        method_marker = data[0]
        compressed_data = data[1:]
        decompressed = self.paq_decompress(compressed_data)
        
        # Map markers to reverse transformations
        reverse_transforms = {
            1: self.reverse_transform_with_prime_xor_every_3_bytes,
            2: self.reverse_transform_with_pattern,
            3: self.reverse_transform_03,
            4: self.reverse_transform_04
        }
        
        if method_marker in reverse_transforms:
            return reverse_transforms[method_marker](decompressed)
        else:
            raise ValueError(f"Unknown compression method marker: {method_marker}")
    
    def paq_compress(self, data):
        return paq.compress(data)  # Placeholder for PAQ compression
    
    def paq_decompress(self, data):
        return paq.decompress(data)  # Placeholder for PAQ decompression

# Main function (unchanged)
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
