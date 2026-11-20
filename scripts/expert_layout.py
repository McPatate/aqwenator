"""Where a converted model's expert records are, for aq-replay: record size and count per
layer, then each layer's file and offset, in the engine's order (layers, then the draft head).

    python scripts/expert_layout.py <model dir> > trace.layout
"""
import glob, json, os, re, struct, sys

rows = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.safetensors"))):
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        meta = json.loads(h.read(n))
    for k, t in meta.items():
        if k.endswith(".mlp.experts.q4r"):
            m = re.search(r"layers\.(\d+)\.", k)
            rows[(k.startswith("mtp"), int(m[1]))] = (f, 8 + n + t["data_offsets"][0], t["shape"])
shape = next(iter(rows.values()))[2]
print(shape[1], shape[0])
for key in sorted(rows):
    print(rows[key][0], rows[key][1])
