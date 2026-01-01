from dataclasses import dataclass
from functools import cached_property
import warnings

from zarr.core.chunk_key_encodings import ChunkKeyEncoding


@dataclass(frozen=True)
class FanoutChunkKeyEncoding(ChunkKeyEncoding):
    name = "fanout"
    max_children: int = 1000

    def __post_init__(self) -> None:
        original_max_children = self.max_children
        if original_max_children < 100:
            raise ValueError("max_children must be at least 100.")

        # floor to a power of 10
        floored_max_children = 10 ** (len(str(original_max_children)) - 1)
        if original_max_children != floored_max_children:
            warnings.warn(
                (
                    f"max_children value {original_max_children} is not a power of 10; "
                    f"it will be floored to {floored_max_children}."
                ),
                UserWarning,
                stacklevel=2,
            )

        object.__setattr__(self, "max_children", floored_max_children)

    @cached_property
    def decimal_len(self) -> int:
        return len(str(self.max_children - 1))

    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        parts: list[str] = ["c"]
        for coord in chunk_coords:
            coords = self._fanout_coord(coord)
            parts.append(str(len(coords) - 1))
            parts.extend(coords)
        return "/".join(parts)

    def _fanout_coord(self, coord: int) -> list[str]:
        parts: list[str] = []
        if coord == 0:
            return ["0" * self.decimal_len]
        while coord > 0:
            parts.append(f"{coord % self.max_children:0{self.decimal_len}d}")
            coord //= self.max_children
        return parts[::-1]
