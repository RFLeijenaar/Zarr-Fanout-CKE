# Zarr Fanout Chunk Key Encoding

A Python implementation of the [`fanout`](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/fanout) chunk key encoding for [**zarr-python**](https://github.com/zarr-developers/zarr-python).

## Overview

The fanout chunk key encoding converts chunk coordinates into a `/`-separated hierarchical path by splitting each coordinate into multiple nodes such that no node in the hierarchy exceeds a predefined maximum number of children. This is particularly useful for filesystems or other hierarchical stores that experience performance issues when directories contain many entries.

## Features

- **Hierarchical chunk organization**: Splits large coordinate values across multiple directory levels
- **Configurable fanout**: Control the maximum number of children per directory node (default: 1001)
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

- **`max_children`** (int, default=1001): Maximum number of child entries allowed within a single directory node. Must be ≥ 3.

## How it works

The algorithm converts chunk coordinates into hierarchical paths:

1. For each coordinate dimension, create a dimension marker `d{dim}`
2. Express the coordinate in base `max_children - 1`
3. Create a subpath: `d{dim}/{digit0}/{digit1}/.../{digitN}`
4. Concatenate all dimension subpaths and append `/c` for the chunk file

### Example

With `max_children = 101` (effective base = 100):

| Coordinates        | Chunk key                    |
| ------------------ | ---------------------------- |
| `()`               | `c`                          |
| `(123,)`           | `d0/1/23/c`                  |
| `(1234, 5, 67890)` | `d0/12/34/d1/5/d2/6/78/90/c` |

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