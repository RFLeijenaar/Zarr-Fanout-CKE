from zarr_fanout_cke import FanoutChunkKeyEncoding

coords = [
    (),
    (12,),
    (1234, 5, 67890),
    (123, 3455678, 9123432435),
    (1234, 0, 239395956),
]

cke = FanoutChunkKeyEncoding(max_children=1234)

for coord in coords:
    encoded = cke.encode_chunk_key(coord)
    print(f"Chunk coords: {coord} -> Encoded key: {encoded}")
