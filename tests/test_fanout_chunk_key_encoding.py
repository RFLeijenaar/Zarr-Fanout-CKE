import pytest
import numpy as np
import zarr
from pathlib import Path
import warnings

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
        assert encoding.max_children == 1000

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

    def test_max_children_too_small(self):
        """Test that max_children cannot be less than 100."""
        with pytest.raises(ValueError, match="max_children must be at least 100."):
            FanoutChunkKeyEncoding(max_children=99)

    def test_max_children_flooring(self):
        """Test that max_children is floored to the nearest power of 10."""
        with pytest.warns(UserWarning, match=r"it will be floored to 100"):
            encoding = FanoutChunkKeyEncoding(max_children=150)
        assert encoding.max_children == 100

        with pytest.warns(UserWarning, match=r"it will be floored to 100"):
            encoding = FanoutChunkKeyEncoding(max_children=999)
        assert encoding.max_children == 100

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            encoding = FanoutChunkKeyEncoding(max_children=1000)
        assert encoding.max_children == 1000

        with pytest.warns(UserWarning, match=r"it will be floored to 1000"):
            encoding = FanoutChunkKeyEncoding(max_children=1234)
        assert encoding.max_children == 1000

    def test_fanout_coord_small_values(self):
        """Test _fanout_coord method with small coordinate values."""
        encoding = FanoutChunkKeyEncoding(max_children=100)

        assert encoding._fanout_coord(0) == ["00"]  # type: ignore
        assert encoding._fanout_coord(5) == ["05"]  # type: ignore
        assert encoding._fanout_coord(38) == ["38"]  # type: ignore

    def test_fanout_coord_equal_to_max_children(self):
        """Test _fanout_coord when coordinate equals max_children."""
        encoding = FanoutChunkKeyEncoding(max_children=100)

        # Coordinate exactly equal to max_children (triggers fanout)
        assert encoding._fanout_coord(100) == ["01", "00"]  # type: ignore

    def test_fanout_coord_larger_values(self):
        """Test _fanout_coord method with coordinates larger than max_children."""
        encoding = FanoutChunkKeyEncoding(max_children=100)

        # Single fanout level
        assert encoding._fanout_coord(123) == ["01", "23"]  # type: ignore
        assert encoding._fanout_coord(999) == ["09", "99"]  # type: ignore

        # Multiple fanout levels
        assert encoding._fanout_coord(12345) == ["01", "23", "45"]  # type: ignore
        assert encoding._fanout_coord(1234567) == ["01", "23", "45", "67"]  # type: ignore

    def test_fanout_coord_different_max_children(self):
        """Test _fanout_coord with different max_children values."""
        # Test with default value (max_children=1000)
        encoding_default = FanoutChunkKeyEncoding()
        assert encoding_default._fanout_coord(500) == ["500"]  # type: ignore
        assert encoding_default._fanout_coord(999) == ["999"]  # type: ignore
        assert encoding_default._fanout_coord(1000) == ["001", "000"]  # type: ignore
        assert encoding_default._fanout_coord(1500) == ["001", "500"]  # type: ignore
        assert encoding_default._fanout_coord(1234567) == ["001", "234", "567"]  # type: ignore

        # Test with smaller value (max_children=100)
        encoding_small = FanoutChunkKeyEncoding(max_children=100)
        assert encoding_small._fanout_coord(50) == ["50"]  # type: ignore
        assert encoding_small._fanout_coord(99) == ["99"]  # type: ignore
        assert encoding_small._fanout_coord(100) == ["01", "00"]  # type: ignore
        assert encoding_small._fanout_coord(250) == ["02", "50"]  # type: ignore
        assert encoding_small._fanout_coord(12345) == ["01", "23", "45"]  # type: ignore


class TestFanoutChunkKeyEncodingEncodeChunkKey:
    """Test suite for the encode_chunk_key method."""

    def test_encode_chunk_key_examples(self):
        """Test encoding based on the provided examples."""
        encoding = FanoutChunkKeyEncoding(max_children=1000)

        assert encoding.encode_chunk_key(()) == "c"
        assert encoding.encode_chunk_key((12,)) == "c/0/012"
        assert encoding.encode_chunk_key((1234, 5, 67890)) == "c/1/001/234/0/005/1/067/890"
        assert (
            encoding.encode_chunk_key((123, 3455678, 9123432435))
            == "c/0/123/2/003/455/678/3/009/123/432/435"
        )
        assert encoding.encode_chunk_key((1234, 0, 239395956)) == "c/1/001/234/0/000/2/239/395/956"
        assert (
            encoding.encode_chunk_key((0, 234235, 34, 3453456343456))
            == "c/0/000/1/234/235/0/034/4/003/453/456/343/456"
        )

    def test_encode_empty_coords(self):
        """Test encoding empty chunk coordinates."""
        encoding = FanoutChunkKeyEncoding()
        assert encoding.encode_chunk_key(()) == "c"


class TestFanoutChunkKeyEncodingMetadata:
    """Test metadata serialization and deserialization."""

    def test_to_dict_default(self):
        """Test serialization with default parameters."""
        encoding = FanoutChunkKeyEncoding()
        result = encoding.to_dict()

        expected = {"name": "fanout", "configuration": {"max_children": 1000}}
        assert result == expected

    def test_to_dict_custom(self):
        """Test serialization with custom parameters."""
        with pytest.warns(UserWarning, match=r"it will be floored to 10000"):
            encoding = FanoutChunkKeyEncoding(max_children=12345)
        result = encoding.to_dict()

        expected = {"name": "fanout", "configuration": {"max_children": 10000}}
        assert result == expected

    def test_from_dict_full_config(self):
        """Test deserialization with full configuration."""
        data = {"name": "fanout", "configuration": {"max_children": 321}}

        with pytest.warns(UserWarning, match=r"it will be floored to 100"):
            encoding = FanoutChunkKeyEncoding.from_dict(data)
        assert encoding.max_children == 100
        assert encoding.name == "fanout"

    def test_from_dict_minimal(self):
        """Test deserialization with minimal configuration."""
        data = {"name": "fanout"}

        encoding = FanoutChunkKeyEncoding.from_dict(data)
        assert encoding.max_children == 1000  # default
        assert encoding.name == "fanout"

    def test_round_trip_serialization(self):
        """Test that serialization and deserialization are consistent."""
        original = FanoutChunkKeyEncoding(max_children=1000)
        data = original.to_dict()
        restored = FanoutChunkKeyEncoding.from_dict(data)

        assert original == restored
        assert original.max_children == restored.max_children


class TestFanoutChunkKeyEncodingUniqueness:
    """Test the uniqueness of generated chunk keys over a complete grid."""

    def test_grid_uniqueness(self):
        """Test that all chunk keys are unique over a complete 2D grid."""
        encoding = FanoutChunkKeyEncoding(max_children=100)

        grid_size = 200

        chunk_keys: set[str] = set()

        # Generate all coordinates in the grid
        for x in range(grid_size):
            for y in range(grid_size):
                coord = (x, y)
                key = encoding.encode_chunk_key(coord)

                # Ensure this key is unique
                assert key not in chunk_keys, f"Duplicate key '{key}' for coordinate {coord}"
                chunk_keys.add(key)

        # Verify we generated the expected number of unique keys
        assert len(chunk_keys) == grid_size * grid_size

        # Verify all keys are strings and non-empty
        for key in chunk_keys:
            assert isinstance(key, str)
            assert len(key) > 0


class TestFanoutChunkKeyEncodingIntegration:
    """Integration tests using actual zarr arrays."""

    def test_zarr_array_roundtrip(self, tmp_path: Path):
        """Test creating zarr arrays, writing data, and reading it back."""
        store_path = tmp_path / "test_array.zarr"

        # Create encoding configuration
        encoding = FanoutChunkKeyEncoding(max_children=100)

        # Create a 2D zarr array with fanout chunk key encoding
        arr: zarr.Array = zarr.create(
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
        reopened_arr: zarr.Array = zarr.open_array(store_path, mode="r")

        # Verify the data is still there and correct

        np.testing.assert_array_equal(reopened_arr[0:10, 0:8], test_data_chunk_0_0)
        np.testing.assert_array_equal(reopened_arr[10:20, 16:24], test_data_chunk_1_2)
        np.testing.assert_array_equal(reopened_arr[50:60, 56:64], test_data_chunk_5_7)

        # Verify that chunk keys follow the fanout pattern by checking the filesystem
        # Chunk (0, 0) should be at path "c/0/00/0/00"
        assert (store_path / "c" / "0" / "00" / "0" / "00").exists()

        # Chunk (1, 2) should be at path "c/0/01/0/02"
        assert (store_path / "c" / "0" / "01" / "0" / "02").exists()

        # Chunk (5, 7) should be at path "c/0/05/0/07"
        assert (store_path / "c" / "0" / "05" / "0" / "07").exists()