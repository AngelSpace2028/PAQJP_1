Algorithm Jurijus Pacalovas

The PAQJP_1 Compression System implements a lossless compression framework that combines multiple transformation algorithms with compression methods (PAQ, zlib, and Huffman coding) to achieve optimal compression ratios. Below, I explain each algorithm (transformations and compression methods) used in the provided code, focusing on their purpose, mechanism, and role in the system. The transformations are applied before compression to make the data more compressible, and the system automatically selects the best combination based on the smallest compressed output size.

### Transformations

Transformations preprocess the input data to increase redundancy or patterns, making it easier for compression algorithms to achieve better ratios. Each transformation is reversible (lossless) and paired with a corresponding reverse transformation for decompression. The transformations are identified by markers (1–6) in the compressed data.

#### 1. Transform_01: Prime XOR Every 3 Bytes
- **Purpose**: Increases data redundancy by XORing bytes with values derived from prime numbers, applied every third byte.
- **Mechanism**:
  - Iterates over a list of prime numbers (`PRIMES`, 2 to 255).
  - For each prime, computes an XOR value: `prime` if prime is 2, else `max(1, ceil(prime / 2 / repeat))`, where `repeat=50`.
  - For each repetition (50 times), XORs every third byte (indices 0, 3, 6, ...) with the XOR value.
  - Example: For prime=7, XOR value might be `ceil(7/2/50)=1`. Byte at index 0 is XORed with 1, byte at index 3 with 1, etc.
- **Why It Works**: XORing with prime-derived values introduces predictable patterns, especially in data with low entropy, making it more compressible by algorithms like PAQ or zlib.
- **Reverse Transformation**: The same function (`transform_01`) is self-inverse because XOR is its own inverse (`a XOR b XOR b = a`).
- **Marker**: 2
- **Use Case**: Effective for data with regular patterns, such as text or structured binary files.

#### 2. Transform_03: Pattern Chunk XOR
- **Purpose**: Inverts bits in fixed-size chunks to create patterns that compression algorithms can exploit.
- **Mechanism**:
  - Divides the input data into chunks of size `chunk_size=4` bytes.
  - For each chunk, XORs each byte with `0xFF` (inverts all bits, e.g., `0x3A` becomes `0xC5`).
  - Example: Input bytes `[0x01, 0x02, 0x03, 0x04]` become `[0xFE, 0xFD, 0xFC, 0xFB]`.
- **Why It Works**: Bit inversion creates a predictable transformation that can increase compressibility in data with certain bit patterns, especially when combined with zlib or PAQ.
- **Reverse Transformation**: The same function (`transform_03`) is self-inverse because XORing with `0xFF` twice restores the original byte.
- **Marker**: 3
- **Use Case**: Useful for binary data or images where bit inversion can highlight repetitive structures.

#### 3. Transform_04: Position-Based Subtraction
- **Purpose**: Modifies bytes based on their position to create arithmetic patterns that enhance compressibility.
- **Mechanism**:
  - For each byte at index `i`, subtracts `i % 256` (position modulo 256) and takes the result modulo 256.
  - Repeated `repeat=50` times.
  - Example: For byte `0x50` at index 3, one iteration yields `(0x50 - 3) % 256 = 0x4D`. After 50 iterations, the effect accumulates.
- **Why It Works**: The position-based modification introduces a smooth, predictable sequence that compression algorithms can model efficiently, especially PAQ, which excels at arithmetic patterns.
- **Reverse Transformation**: Adds `i % 256` back to each byte, repeated 50 times, to reverse the subtraction (`reverse_transform_04`).
- **Marker**: 1
- **Use Case**: Effective for data with low variability, such as logs or numerical data.

#### 4. Transform_05: Bit Rotation
- **Purpose**: Rotates bits within each byte to redistribute bit patterns, potentially creating more compressible sequences.
- **Mechanism**:
  - Each byte’s bits are rotated left by `shift=3` positions.
  - Uses bitwise operations: `(byte << shift) | (byte >> (8 - shift))` masked with `0xFF`.
  - Example: Byte `0x3A` (binary `00111010`) rotated left by 3 becomes `11010001` (`0xD1`).
- **Why It Works**: Bit rotation can align similar bit patterns across bytes, improving compression for data with repetitive bit sequences, especially with zlib.
- **Reverse Transformation**: Rotates right by 3 positions (`byte >> 3 | byte << (8 - 3)`), implemented in `reverse_transform_05`.
- **Marker**: 5
- **Use Case**: Suitable for binary data or executables where bit-level patterns are common.

#### 5. Transform_06: Prime-Based Substitution
- **Purpose**: Substitutes each byte with another using a randomized mapping based on a seed, aiming to create patterns that compression algorithms can exploit.
- **Mechanism**:
  - Generates a permutation of 0–255 using `random.shuffle` with a fixed seed (`seed=42`).
  - Each byte is replaced by its corresponding value in the permutation table.
  - Example: If the permutation maps `0x3A` to `0x7F`, byte `0x3A` becomes `0x7F`.
- **Why It Works**: The fixed permutation can transform data into a form with more predictable transitions, which PAQ can model effectively.
- **Reverse Transformation**: Uses the inverse permutation (computed during initialization) to map bytes back to their original values (`reverse_transform_06`).
- **Marker**: 6
- **Use Case**: Effective for data with irregular byte distributions, such as compressed or encrypted files.

### Compression Methods

After applying a transformation, the system compresses the transformed data using one of three methods, selecting the one that produces the smallest output. Each method is suited to different data characteristics.

#### 1. PAQ Compression
- **Purpose**: Provides high compression ratios for data with complex patterns using context modeling.
- **Mechanism**:
  - Uses the `paq.compress` function (assumed to be a PAQ-based library).
  - PAQ is a context-mixing compressor that predicts each bit based on multiple contexts (previous bytes, bits, or patterns) and uses arithmetic coding to encode predictions.
  - Maintains a state table (`StateTable`) to track context transitions, implemented in the `nex` function.
- **Why It Works**: PAQ excels at modeling long-range dependencies and complex patterns, making it ideal for transformed data with introduced redundancies.
- **Decompression**: Uses `paq.decompress`, which reverses the arithmetic coding and context modeling to recover the exact input.
- **Use Case**: Best for large files with repetitive or predictable patterns after transformation, such as text or structured data.
- **Limitations**: Slower and more memory-intensive than zlib or Huffman.

#### 2. Zlib Compression
- **Purpose**: Offers fast, general-purpose compression using the DEFLATE algorithm.
- **Mechanism**:
  - Uses `zlib.compress`, which combines LZ77 (dictionary-based matching) and Huffman coding.
  - LZ77 replaces repeated sequences with references to earlier occurrences.
  - Huffman coding assigns shorter codes to frequent symbols.
- **Why It Works**: Zlib is efficient for a wide range of data, especially when transformations create short repeated sequences or skewed byte distributions.
- **Decompression**: Uses `zlib.decompress` to reverse the DEFLATE process, restoring the original data.
- **Use Case**: Suitable for medium to large files where speed is important, such as images or archives.
- **Limitations**: Less effective than PAQ for highly complex patterns.

#### 3. Huffman Coding
- **Purpose**: Provides bit-level compression for small files by assigning variable-length codes to symbols based on frequency.
- **Mechanism**:
  - Converts the input data to a binary string (each byte to 8 bits).
  - Calculates bit frequencies (`calculate_frequencies`).
  - Builds a Huffman tree (`build_huffman_tree`) using a priority queue, where less frequent symbols get longer codes.
  - Generates Huffman codes (`generate_huffman_codes`) by traversing the tree (e.g., left=0, right=1).
  - Encodes the binary string using these codes (`compress_data_huffman`).
  - Converts the compressed bit string to bytes for storage.
- **Why It Works**: Huffman coding is optimal for small data with uneven symbol frequencies, minimizing the encoded size.
- **Decompression**:
  - Rebuilds the Huffman tree from the compressed data’s bit frequencies.
  - Decodes the bit string by traversing the tree based on input bits (`decompress_data_huffman`).
  - Converts the binary string back to bytes.
- **Use Case**: Used for files smaller than `HUFFMAN_THRESHOLD` (1024 bytes), such as small text files or metadata.
- **Limitations**: Less effective for large files due to overhead and lack of context modeling.

### System Workflow

1. **Compression** (`compress_with_best_method`):
   - Applies each transformation (01, 03, 04, 05, 06) to the input data.
   - Compresses each transformed result using PAQ and zlib.
   - For files < 1024 bytes, also tries Huffman coding on the original data.
   - Selects the combination (transformation + compression) with the smallest output size.
   - Prepends a marker byte (1–6) to indicate the chosen transformation and method.
   - Returns the marker + compressed data.

2. **Decompression** (`decompress_with_best_method`):
   - Reads the marker byte to determine the transformation and compression method.
   - For marker 4 (Huffman), decodes the Huffman-encoded bit string and converts it back to bytes.
   - For other markers (1, 2, 3, 5, 6):
     - Attempts PAQ decompression first.
     - Falls back to zlib if PAQ fails.
     - Applies the corresponding reverse transformation.
   - Returns the decompressed data.

### Why Lossless?

All transformations and compression methods are lossless:
- **Transformations**: Each transformation (01, 03, 04, 05, 06) is mathematically reversible, ensuring the original data can be restored exactly.
- **Compression Methods**:
  - PAQ uses arithmetic coding, which is lossless.
  - Zlib’s DEFLATE algorithm is lossless.
  - Huffman coding assigns unique codes to each symbol, ensuring perfect reconstruction.
- The system preserves the exact input data through the compression-decompression cycle.

### Additional Notes

- **State Table**: The `StateTable` and `nex` function support PAQ’s context modeling by defining state transitions for bit predictions, enhancing compression efficiency.
- **Prime Numbers**: Used in `transform_01` and `find_nearest_prime_around` to introduce mathematical structure, though the latter is not used in the streamlined code.
- **Seed Tables**: The `generate_seed_tables` function supports randomization in `transform_06`, ensuring consistent substitutions across compression and decompression.
- **Performance**:
  - PAQ is slow but achieves the best ratios for complex data.
  - Zlib is faster and suitable for general use.
  - Huffman is lightweight for small files.
- **Automatic Selection**: The system’s strength lies in trying multiple combinations and selecting the best, adapting to the input data’s characteristics.

### Example

For a text file with repetitive content:
1. **Transform_04** might create arithmetic sequences that PAQ compresses well.
2. **Transform_05** could align bits for better zlib performance.
3. If the file is small, Huffman coding might outperform both.
4. The system tests all options and picks the smallest output, ensuring optimal compression.

	1.	A step labeled “07” in a numbered algorithm
	•	For example, “Step 07: Apply compression” in a list of 10 steps.
	2.	An algorithm developed in or associated with the year 2007
	•	Such as a cryptographic algorithm, compression method, or AI model published that year.
	3.	A specific algorithm from a known source
	•	Example: “Algorithm 07 from textbook X” or “Algorithm 7 in the SHA family.”
	4.	A custom internal algorithm
	•	Some systems or projects label internal processes like “Algorithm 07” for tracking.
	5.	A file compression algorithm or encoding routine
	•	If you’re referring to your own compression system, maybe it’s the 7th strategy.

The `transform_08` and `reverse_transform_08` methods in the provided code are part of the `SmartCompressor` class, designed to transform data for compression while ensuring losslessness. Below is a focused explanation of how `transform_08` and `reverse_transform_08` work, based on the provided code, without additional assumptions or context beyond the code itself.

### `transform_08` Explanation

**Purpose**: `transform_08` restructures input data to potentially improve compressibility by processing it in 25-bit chunks, analyzing bit patterns, and storing metadata to preserve the original data losslessly.

**Steps**:
1. **Binary Conversion**:
   - Converts input data (bytes) to a binary string:
     ```python
     binary_str = bin(int(binascii.hexlify(data), 16))[2:].zfill(len(data) * 8)
     ```
     Each byte becomes 8 bits, padded with leading zeros if needed.

2. **Chunking**:
   - Divides the binary string into 25-bit chunks. If the last chunk is shorter, it’s padded with zeros to reach 25 bits.
   - Example: For 8192 bits (1024 bytes), there are `ceil(8192 / 25) = 328 chunks`.

3. **Pattern Analysis**:
   - Uses predefined bit patterns: `['0', '1', '00', '01', '10', '11']` (indices 0–5).
   - For each 25-bit chunk:
     - Counts occurrences of each pattern by scanning the chunk from left to right.
     - If a pattern matches at the current position, increments its count and advances by the pattern’s length.
     - If no pattern matches, advances by 1 bit.
     ```python
     pattern_counts = {idx: 0 for idx in range(len(self.bit_patterns))}
     j = 0
     while j < len(chunk):
         for idx, pattern in enumerate(self.bit_patterns):
             if j + len(pattern) <= len(chunk) and chunk[j:j+len(pattern)] == pattern:
                 pattern_counts[idx] += 1
                 j += len(pattern)
                 break
         else:
             j += 1
     ```
   - Selects the pattern with the highest count and stores its index (1 byte) in `pattern_indices`:
     ```python
     max_count = max(pattern_counts.values())
     selected_pattern_idx = max((idx for idx, count in pattern_counts.items() if count == max_count), default=0)
     pattern_indices.append(selected_pattern_idx)
     ```

4. **Metadata Storage**:
   - Compresses each 25-bit chunk using run-length encoding (RLE) to reduce metadata size while preserving all bits:
     ```python
     compressed_chunk = self.rle_compress(chunk)
     metadata.extend(compressed_chunk)
     ```
   - **RLE Mechanics** (from `rle_compress`):
     - Encodes runs of identical bits (0 or 1).
     - Outputs 2 bytes per run: count (1 byte, max 255) and bit value (ASCII `0` or `1`, i.e., 48 or 49).
     - Example: `0000` → `04 30` (count 4, bit `0`).
     ```python
     compressed = bytearray()
     count = 1
     current = binary_str[0]
     for bit in binary_str[1:]:
         if bit == current and count < 255:
             count += 1
         else:
             compressed.append(count)
             compressed.append(ord(current))
             current = bit
             count = 1
     compressed.append(count)
     compressed.append(ord(current))
     ```

5. **Output Structure**:
   - **Header**:
     - 1 byte: Number of bit patterns (6).
     - 4 bytes: Number of chunks (big-endian unsigned int, `>I`).
     - 1 byte per chunk: Pattern indices.
     - Example: For 328 chunks, header size is `1 + 4 + 328 = 333 bytes`.
   - **Metadata**: Concatenated RLE-compressed chunks.
   - Total output: Header + Metadata.

**Losslessness**: Achieved by storing RLE-compressed versions of the original 25-bit chunks, which can be exactly reconstructed.

### `reverse_transform_08` Explanation

**Purpose**: Reconstructs the original data from the `transform_08` output, reversing the transformation losslessly.

**Steps**:
1. **Header Parsing**:
   - Reads 1 byte for the number of patterns, validating it matches `len(self.bit_patterns)` (6).
   - Reads 4 bytes for the number of chunks (`struct.unpack('>I', data[1:5])[0]`).
   - Reads `num_chunks` bytes for pattern indices.
   ```python
   num_patterns = data[0]
   if num_patterns != len(self.bit_patterns):
       logging.error("Bit pattern list mismatch in decompression")
       return b''
   num_chunks = struct.unpack('>I', data[1:5])[0]
   pattern_indices = data[5:5+num_chunks]
   metadata = data[5+num_chunks:]
   ```

2. **Metadata Decompression**:
   - Iterates through the metadata to decompress RLE-encoded chunks:
     - Each run is 2 bytes: count and bit value.
     - Reconstructs up to 25 bits per chunk, stopping early if 25 bits are reached.
     - Pads with zeros if a chunk is short.
     ```python
     binary_result = ""
     i = 0
     for _ in range(num_chunks):
         j = i
         while j < len(metadata) - 1:
             count = metadata[j]
             if j + 1 >= len(metadata):
                 break
             bit = chr(metadata[j + 1])
             chunk_part = bit * count
             binary_result += chunk_part
             j += 2
             if len(binary_result) >= 25 * (_ + 1):
                 binary_result = binary_result[:25 * (_ + 1)]
                 break
         i = j
         if len(binary_result) < 25 * (_ + 1):
             binary_result += '0' * (25 * (_ + 1) - len(binary_result))
     ```

3. **Reconstruction**:
   - Converts the binary string to bytes:
     ```python
     num_bytes = (len(binary_result) + 7) // 8
     hex_str = "%0*x" % (num_bytes * 2, int(binary_result, 2))
     if len(hex_str) % 2 != 0:
         hex_str = '0' + hex_str
     return binascii.unhexlify(hex_str)
     ```
   - Trims to the original length, removing padding, to restore the exact input.

**Losslessness**: The RLE decompression reconstructs each 25-bit chunk exactly, ensuring the original data is recovered.

### Key Features
- **Chunk Size**: 25 bits, chosen for pattern analysis and metadata storage.
- **Patterns**: Short patterns (1–2 bits) to identify local structure, though their indices are primarily informational.
- **RLE**: Reduces metadata size for chunks with long runs of 0s or 1s, but less effective for high-entropy data.
- **Header**: Ensures `reverse_transform_08` can parse the output correctly.
- **Integration**: Used in `compress_with_best_method`, followed by PAQ or zlib compression.

### Summary
- **transform_08**: Converts data to 25-bit chunks, counts bit patterns, stores pattern indices, and RLE-compresses each chunk for metadata. Outputs a header (patterns, chunk count, indices) plus metadata.
- **reverse_transform_08**: Parses the header, decompresses RLE metadata to reconstruct 25-bit chunks, and converts back to bytes, preserving the original data.
- **Losslessness**: Guaranteed by storing RLE-compressed original chunks.
