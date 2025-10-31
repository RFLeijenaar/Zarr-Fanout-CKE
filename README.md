# Zarr Fanout Chunk Key Encoding

A Python implementation of the [`fanout`](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/fanout) chunk key encoding for [**zarr-python**](https://github.com/zarr-developers/zarr-python).

## Overview

The *fanout* chunk key encoding converts chunk coordinates into a `/`-separated hierarchical path by splitting each coordinate into multiple nodes such that no node in the hierarchy exceeds a predefined maximum number of children. This is particularly useful for filesystems or other hierarchical stores that experience performance issues when directories contain many entries. The encoding ensures lexicographical ordering of chunk keys.

## Features

- **Hierarchical chunk organization**: Splits large coordinate values across multiple directory levels
- **Configurable fanout**: Control the maximum number of children per directory node (default: 1000)
- **Seamless zarr-python integration**: Works as a drop-in chunk key encoding for zarr arrays
- **Performance optimization**: Prevents filesystem performance degradation from directories with too many files

## Requirements

- Python ≥ 3.11
- zarr ≥ 3.1.3

## Installation

### Using pip

```bash
pip install git+https://github.com/RFLeijenaar/Zarr-Fanout-CKE.git
```

### Using uv

```bash
uv add git+https://github.com/RFLeijenaar/Zarr-Fanout-CKE.git
```

## Usage

```python
import zarr
import numpy as np
from zarr_fanout_cke import FanoutChunkKeyEncoding

# Create a zarr array with fanout chunk key encoding
encoding = FanoutChunkKeyEncoding(max_children=100)

arr = zarr.create_array(
    store="my_array.zarr",
    shape=(10000, 10000),
    chunks=(100, 100),
    dtype=np.float64,
    chunk_key_encoding=encoding
)

# Write some data
arr[0:100, 0:100] = np.random.random((100, 100))
```

## Configuration

The `FanoutChunkKeyEncoding` accepts the following parameter:

- **`max_children`** (int, default=1000): Maximum number of child entries allowed within a single directory node. Must be ≥ 100.

## How it works

The algorithm converts chunk coordinates into hierarchical paths:

1. Compute `decimal_len`, the number of digits in `max_children - 1`.
2. For each coordinate:
   * Split the coordinate into decimal chunks of length `decimal_len`, starting from the least significant digits.
   * Pad the leftmost chunk with zeros as needed so that it has exactly `decimal_len` digits.
   * Prepend the number of chunks of the coordinate before the sequence of chunks.
3. Concatenate all coordinate chunk sequences in order (from the lowest to highest dimension) and prepend `"c"` as the root.
4. Join all parts using `/` as a separator.

### Example

With `max_children = 1000` (decimal length = 3):

| Coordinates                  | Chunk key                                 |
| ---------------------------- | ----------------------------------------- |
| `()`                         | `c`                                       |
| `(0)`                        | `c/1/000`                                 |
| `(12,)`                      | `c/1/012`                                 |
| `(1234, 5, 6789012)`         | `c/2/001/234/1/005/3/006/789/012`         |

## Development

### Prerequisites

Ensure that [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed.

### Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/RFLeijenaar/Zarr-Fanout-CKE.git
cd Zarr-Fanout-CKE
uv sync
```

### Testing

Run the test suite:

```bash
uv run pytest
```

## License

This project follows the MIT License. See the [LICENSE](LICENSE) file for details.

## Related

- [Zarr extension specification for fanout chunk key encoding](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/fanout)
- [zarr-python](https://github.com/zarr-developers/zarr-python)