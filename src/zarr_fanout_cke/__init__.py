from dataclasses import dataclass

from zarr.core.chunk_key_encodings import ChunkKeyEncoding


@dataclass(frozen=True)
class FanoutChunkKeyEncoding(ChunkKeyEncoding):
    name = "fanout"
    max_children: int = 1001

    def _fanout_coord(self, coord: int) -> list[str]:
        # Break a coordinate into fanout parts
        parts: list[str] = []
        base = self.max_children - 1
        while coord >= base:
            coord, rem = divmod(coord, base)
            parts.append(str(rem))
        parts.append(str(coord))
        parts.reverse()
        return parts

    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        parts: list[str] = []
        for dim, coord in enumerate(chunk_coords):
            coords = self._fanout_coord(coord)
            coords = [f"d{dim}"] + coords
            parts.extend(coords)
        parts.append("c")
        return "/".join(parts)
