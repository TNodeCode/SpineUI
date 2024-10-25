import os
import json

root_dir = "data/MOT17" if not os.getenv("ROOT_DIR") else os.getenv("ROOT_DIR")

for split in ["train", "half-train", "half-val", "test"]:
    with open(f"{root_dir}/annotations/{split}_cocoformat.json", "r") as f:
        content = f.read()
        content = json.loads(content)
        content['categories'] = [dict(id=0, name="spine")]

    with open(f"{root_dir}/annotations/{split}_cocoformat_.json", "w+") as f:
        f.write(json.dumps(content, indent=4))