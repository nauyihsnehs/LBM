import os
from PIL import Image
from tqdm import tqdm

root = "/mnt/data1/ssy/render_people/fill-light-dataset"
bad = []
for dirpath, _, filenames in os.walk(root):
    for fn in tqdm(filenames):
        if fn.lower().endswith("_rlb.png") or fn.lower().endswith("_rgb.png") or fn.lower().endswith("_alb.png"):
            p = os.path.join(dirpath, fn)
            try:
                with Image.open(p) as im:
                    im.load()  # 真解码
            except Exception as e:
                bad.append((p, repr(e)))
                print(p)

print("BAD COUNT:", len(bad))
for p,e in bad[:200]:
    print(p, e)