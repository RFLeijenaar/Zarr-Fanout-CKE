import pytest
import numpy as np
import zarr
from pathlib import Path

from zarr_fanout_cke import FanoutChunkKeyEncoding


class TestFanoutChunkKeyEncoding:
    """Test suite for FanoutChunkKeyEncoding chunk key encoding."""

    def test_name_attribute(self):
        """Test that the name attribute is set correctly."""
        encoding = FanoutChunkKeyEncoding()
        assert encoding.name == "fanout"

    def test_default_max_children(self):
        """Test the default value of max_children."""
        encoding = FanoutChunkKeyEncoding()
        assert encoding.max_children == 1001

    def test_custom_max_children(self):
        """Test setting a custom max_children value."""
        encoding = FanoutChunkKeyEncoding(max_children=100)
        assert encoding.max_children == 100

    def test_frozen_dataclass(self):
        """Test that the dataclass is frozen (immutable)."""
        encoding = FanoutChunkKeyEncoding()
        with pytest.raises((AttributeError, Exception)):  # dataclass frozen error
            encoding.max_children = 500  # type: ignore

    def test_name_immutable(self):
        """Test that the name attribute cannot be modified."""
        encoding = FanoutChunkKeyEncoding()
        with pytest.raises((AttributeError, Exception)):  # name should be immutable
            encoding.name = "modified"  # type: ignore

    def test_name_cannot_be_set_in_init(self):
        """Test that name cannot be passed as an argument to __init__."""
        with pytest.raises(TypeError):  # name is not a dataclass field
            FanoutChunkKeyEncoding(name="custom_name")  # type: ignore

    def test_fanout_coord_small_values(self):
        """Test _fanout_coord method with small coordinate values."""
        encoding = FanoutChunkKeyEncoding(max_children=11)

        # Coordinate less than max_children (base = max_children - 1 = 10)
        assert encoding._fanout_coord(5) == ["5"]  # type: ignore
        assert encoding._fanout_coord(0) == ["0"]  # type: ignore
        assert encoding._fanout_coord(8) == ["8"]  # type: ignore

    def test_fanout_coord_equal_to_max_children(self):
        """Test _fanout_coord when coordinate equals max_children."""
        encoding = FanoutChunkKeyEncoding(max_children=10)

        # Coordinate exactly equal to max_children (triggers fanout)
        assert encoding._fanout_coord(10) == ["1", "1"]  # 10 = 1*9 + 1 # type:ignore

    def test_fanout_coord_larger_values(self):
        """Test _fanout_coord method with coordinates larger than max_children."""
        encoding = FanoutChunkKeyEncoding(max_children=9)

        # Single fanout level (base = 8)
        assert encoding._fanout_coord(7) == ["7"]  # type: ignore
        assert encoding._fanout_coord(8) == ["1", "0"]  # type: ignore
        assert encoding._fanout_coord(10) == ["1", "2"]  # type: ignore
        assert encoding._fanout_coord(16) == ["2", "0"]  # type: ignore

        # Multiple fanout levels
        assert encoding._fanout_coord(64) == ["1", "0", "0"]  # type: ignore
        assert encoding._fanout_coord(100) == ["1", "4", "4"]  # type: ignore
        assert encoding._fanout_coord(500) == ["7", "6", "4"]  # type: ignore

    def test_fanout_coord_different_max_children(self):
        """Test _fanout_coord with different max_children values."""
        # Test with default value (base = 1000)
        encoding_default = FanoutChunkKeyEncoding()  # max_children=1001
        assert encoding_default._fanout_coord(500) == ["500"]  # type: ignore
        assert encoding_default._fanout_coord(999) == ["999"]  # type: ignore
        assert encoding_default._fanout_coord(1000) == ["1", "0"]  # type: ignore
        assert encoding_default._fanout_coord(1500) == [  # type: ignore
            "1",
            "500",
        ]

        # Test with smaller value (base = 100)
        encoding_small = FanoutChunkKeyEncoding(max_children=101)
        assert encoding_small._fanout_coord(50) == ["50"]  # type: ignore
        assert encoding_small._fanout_coord(99) == ["99"]  # type: ignore
        assert encoding_small._fanout_coord(100) == ["1", "0"]  # type: ignore
        assert encoding_small._fanout_coord(250) == ["2", "50"]  # type: ignore


class TestFanoutChunkKeyEncodingEncodeChunkKey:
    """Test suite for the encode_chunk_key method."""

    def test_encode_1d_chunks_small_coords(self):
        """Test encoding 1D chunk coordinates with small values."""
        encoding = FanoutChunkKeyEncoding(max_children=10)

        # Single dimension, small coordinates (base = 9)
        assert encoding.encode_chunk_key((0,)) == "d0/0/c"
        assert encoding.encode_chunk_key((5,)) == "d0/5/c"
        assert encoding.encode_chunk_key((8,)) == "d0/8/c"

    def test_encode_1d_chunks_large_coords(self):
        """Test encoding 1D chunk coordinates with large values requiring fanout."""
        encoding = FanoutChunkKeyEncoding(max_children=10)

        # Single dimension, coordinates requiring fanout (base = 9)
        assert encoding.encode_chunk_key((9,)) == "d0/1/0/c"
        assert encoding.encode_chunk_key((10,)) == "d0/1/1/c"  # 10 = 1*9 + 1
        assert encoding.encode_chunk_key((18,)) == "d0/2/0/c"  # 18 = 2*9 + 0
        assert encoding.encode_chunk_key((90,)) == "d0/1/1/0/c"  # 90 = 1*9*9 + 1*9 + 0

    def test_encode_2d_chunks(self):
        """Test encoding 2D chunk coordinates."""
        encoding = FanoutChunkKeyEncoding(max_children=10)

        # 2D coordinates, mix of small and large values (base = 9)
        assert encoding.encode_chunk_key((0, 0)) == "d0/0/d1/0/c"
        assert encoding.encode_chunk_key((5, 7)) == "d0/5/d1/7/c"
        assert encoding.encode_chunk_key((9, 5)) == "d0/1/0/d1/5/c"
        assert encoding.encode_chunk_key((10, 5)) == "d0/1/1/d1/5/c"  # 10 = 1*9 + 1
        assert (
            encoding.encode_chunk_key((18, 27)) == "d0/2/0/d1/3/0/c"
        )  # 18 = 2*9 + 0, 27 = 3*9 + 0

    def test_encode_3d_chunks(self):
        """Test encoding 3D chunk coordinates."""
        encoding = FanoutChunkKeyEncoding(max_children=10)

        # 3D coordinates (base = 9)
        assert encoding.encode_chunk_key((0, 0, 0)) == "d0/0/d1/0/d2/0/c"
        assert encoding.encode_chunk_key((1, 2, 3)) == "d0/1/d1/2/d2/3/c"
        assert encoding.encode_chunk_key((8, 18, 27)) == "d0/8/d1/2/0/d2/3/0/c"

    def test_encode_empty_coords(self):
        """Test encoding empty chunk coordinates."""
        encoding = FanoutChunkKeyEncoding()
        assert encoding.encode_chunk_key(()) == "c"

    def test_encode_with_default_max_children(self):
        """Test encoding with the default max_children value."""
        encoding = FanoutChunkKeyEncoding()  # max_children=1001, base=1000

        # Values below threshold
        assert encoding.encode_chunk_key((500,)) == "d0/500/c"
        assert encoding.encode_chunk_key((999,)) == "d0/999/c"

        # Values at and above threshold
        assert encoding.encode_chunk_key((1000,)) == "d0/1/0/c"  # 1000 = 1*1000 + 0
        assert encoding.encode_chunk_key((1500,)) == "d0/1/500/c"  # 1500 = 1*1000 + 500
        assert encoding.encode_chunk_key((2500,)) == "d0/2/500/c"  # 2500 = 2*1000 + 500

    def test_encode_large_multi_dimensional(self):
        """Test encoding large coordinates in multiple dimensions."""
        encoding = FanoutChunkKeyEncoding(max_children=87)  # base = 86

        # Large multi-dimensional coordinates
        coords = (9999, 1933899)
        # 9999 = 1*86*86 + 30*86 + 23, 1933899 = 3*86*86*86 + 3*86*86 + 41*86 + 17
        result = encoding.encode_chunk_key(coords)
        expected = "d0/1/30/23/d1/3/3/41/17/c"
        assert result == expected


class TestFanoutChunkKeyEncodingMetadata:
    """Test metadata serialization and deserialization."""

    def test_to_dict_default(self):
        """Test serialization with default parameters."""
        encoding = FanoutChunkKeyEncoding()
        result = encoding.to_dict()

        expected = {"name": "fanout", "configuration": {"max_children": 1001}}
        assert result == expected

    def test_to_dict_custom(self):
        """Test serialization with custom parameters."""
        encoding = FanoutChunkKeyEncoding(max_children=500)
        result = encoding.to_dict()

        expected = {"name": "fanout", "configuration": {"max_children": 500}}
        assert result == expected

    def test_from_dict_full_config(self):
        """Test deserialization with full configuration."""
        data = {"name": "fanout", "configuration": {"max_children": 500}}

        encoding = FanoutChunkKeyEncoding.from_dict(data)  # type: ignore
        assert encoding.max_children == 500
        assert encoding.name == "fanout"

    def test_from_dict_minimal(self):
        """Test deserialization with minimal configuration."""
        data = {"name": "fanout"}

        encoding = FanoutChunkKeyEncoding.from_dict(data)  # type: ignore
        assert encoding.max_children == 1001  # default
        assert encoding.name == "fanout"

    def test_round_trip_serialization(self):
        """Test that serialization and deserialization are consistent."""
        original = FanoutChunkKeyEncoding(max_children=750)
        data = original.to_dict()
        restored = FanoutChunkKeyEncoding.from_dict(data)

        assert original == restored
        assert original.max_children == restored.max_children


class TestFanoutChunkKeyEncodingUniqueness:
    """Test the uniqueness of generated chunk keys over a complete grid."""

    def test_grid_uniqueness(self):
        """Test that all chunk keys are unique over a complete 2D grid."""
        encoding = FanoutChunkKeyEncoding(max_children=5)  # base = 4

        # Define grid size that covers two digits per dimension
        # With base=4, we need coordinates >= 16 to get two digits (16 = 4*4 + 0)
        grid_size = 20  # This ensures we test 0-19 in each dimension

        chunk_keys: set[str] = set()

        # Generate all coordinates in the grid
        for x in range(grid_size):
            for y in range(grid_size):
                coord = (x, y)
                key = encoding.encode_chunk_key(coord)

                # Ensure this key is unique
                assert key not in chunk_keys, (
                    f"Duplicate key '{key}' for coordinate {coord}"
                )
                chunk_keys.add(key)

        # Verify we generated the expected number of unique keys
        assert len(chunk_keys) == grid_size * grid_size

        # Verify all keys are strings and non-empty
        for key in chunk_keys:
            assert isinstance(key, str)
            assert len(key) > 0
            assert key.endswith("/c")  # All chunk keys should end with "/c"


class TestFanoutChunkKeyEncodingIntegration:
    """Integration tests using actual zarr arrays."""

    def test_zarr_array_roundtrip(self, tmp_path: Path):
        """Test creating zarr arrays, writing data, and reading it back."""
        store_path = tmp_path / "test_array.zarr"

        # Create encoding configuration
        encoding = FanoutChunkKeyEncoding(max_children=11)

        # Create a 2D zarr array with fanout chunk key encoding
        arr: zarr.Array = zarr.create_array(  # type: ignore
            store=store_path,
            shape=(100, 80),
            chunks=(10, 8),
            dtype=np.float32,
            chunk_key_encoding=encoding,
        )

        # Write some test data to specific chunks
        test_data_chunk_0_0 = np.arange(80, dtype=np.float32).reshape(10, 8)
        test_data_chunk_1_2 = np.full((10, 8), 42.5, dtype=np.float32)
        test_data_chunk_5_7 = np.random.random((10, 8)).astype(np.float32)

        # Write data to different chunks
        arr[0:10, 0:8] = test_data_chunk_0_0  # chunk (0, 0)
        arr[10:20, 16:24] = test_data_chunk_1_2  # chunk (1, 2)
        arr[50:60, 56:64] = test_data_chunk_5_7  # chunk (5, 7)

        # Close and reopen the array
        del arr

        # Reopen the array
        reopened_arr: zarr.Array = zarr.open_array(store_path, mode="r")  # type: ignore

        # Verify the data is still there and correct

        np.testing.assert_array_equal(reopened_arr[0:10, 0:8], test_data_chunk_0_0)
        np.testing.assert_array_equal(reopened_arr[10:20, 16:24], test_data_chunk_1_2)
        np.testing.assert_array_equal(reopened_arr[50:60, 56:64], test_data_chunk_5_7)

        # Verify that chunk keys follow the fanout pattern by checking the filesystem
        # Chunk (0, 0) should be at path "d0/0/d1/0/c"
        assert (store_path / "d0" / "0" / "d1" / "0" / "c").exists()

        # Chunk (1, 2) should be at path "d0/1/d1/2/c"
        assert (store_path / "d0" / "1" / "d1" / "2" / "c").exists()

        # Chunk (5, 7) should be at path "d0/5/d1/7/c"
        assert (store_path / "d0" / "5" / "d1" / "7" / "c").exists()
