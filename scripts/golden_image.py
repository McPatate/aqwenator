"""Reference for image input: one synthetic image in a chat prompt, run in transformers.

Writes: pixel_values, grid_thw, vision_embeds (tower output after the merger, one row per
image token), input_ids, position_ids [3, T], hidden.0 (embeddings with the image rows in
place), logits_last, greedy_ids, tf_logits.

    python -I golden_image.py --model Qwen/Qwen3.5-0.8B --size 448x320 --out img.safetensors
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor


def picture(w, h):
    """Deterministic test card: gradients, a grid and a few filled boxes."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h)], -1).astype(np.uint8)
    img[(x % 64 < 3) | (y % 64 < 3)] = 255
    for k, c in enumerate([(220, 30, 30), (30, 200, 60), (40, 60, 230)]):
        img[h // 4 : h // 2, (2 * k + 1) * w // 8 : (2 * k + 2) * w // 8] = c
    return Image.fromarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", default="448x320", help="WxH of the test image")
    p.add_argument("--greedy", type=int, default=32)
    args = p.parse_args()
    w, h = map(int, args.size.split("x"))

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    msgs = [{"role": "user", "content": [{"type": "image", "image": picture(w, h)}, {"type": "text", "text": "How many filled boxes are in the picture, and what are their colors?"}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
    ids = inp["input_ids"]

    t = {"input_ids": ids[0].to(torch.int32).cpu(), "pixel_values": inp["pixel_values"].float().cpu().contiguous(), "grid_thw": inp["image_grid_thw"].to(torch.int32).cpu()}
    with torch.no_grad():
        feats = model.get_image_features(inp["pixel_values"].to(model.dtype), inp["image_grid_thw"])
        ve = feats.pooler_output if hasattr(feats, "pooler_output") else feats
        ve = torch.cat(list(ve)) if isinstance(ve, (list, tuple)) else ve
        t["vision_embeds"] = ve.float().cpu().contiguous()
        kw = {k: v for k, v in inp.items()}
        pos, _ = model.model.get_rope_index(ids, mm_token_type_ids=inp.get("mm_token_type_ids"), image_grid_thw=inp["image_grid_thw"], attention_mask=inp["attention_mask"])
        t["position_ids"] = pos[:, 0].to(torch.int32).cpu().contiguous()
        out = model(**kw, output_hidden_states=True, use_cache=False)
        for i, hs in enumerate(out.hidden_states):
            t[f"hidden.{i}"] = hs[0].float().cpu().contiguous()
        t["image_rows"] = (ids[0] == model.config.image_token_id).nonzero()[:, 0].to(torch.int32).cpu()
        t["logits_last"] = out.logits[0, -1].float().cpu().contiguous()
        gen = model.generate(**kw, max_new_tokens=args.greedy, do_sample=False)
        greedy = gen[0, ids.shape[1] :]
        t["greedy_ids"] = greedy.to(torch.int32).cpu()
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    save_file(t, os.path.expanduser(args.out), metadata={"model": args.model, "image": args.size})
    n_img = int((ids[0] == model.config.image_token_id).sum())
    print("saved;", ids.shape[1], "prompt tokens,", n_img, "image tokens, grid", t["grid_thw"].tolist(), "vision_embeds", tuple(t["vision_embeds"].shape))
    print("positions around the image:", t["position_ids"][:, :8].tolist(), "... last", t["position_ids"][:, -3:].tolist())
    print("greedy:", repr(proc.tokenizer.decode(greedy)[:200]))


if __name__ == "__main__":
    main()
