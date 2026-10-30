"""Split dataset_coco into train/valid/test by arrangement (A1..A10).

8 arrangements -> train, 1 -> valid, 1 -> test. The valid/test arrangements are
chosen so every object they contain also appears in the train arrangements.
"""
import argparse
import itertools
import json
import os
import re
import shutil
from collections import defaultdict

# e.g. images/f18ed72f__wp5_keys_p1_yellow_A6_id379.png
#      images/c018bddf__wp5_no-pointing_no_yellow_A7_id707.png
NAME_RE = re.compile(r"__wp\d+_(?P<obj>.+?)_(?:p\d+|no)_.+?_(?P<arr>A\d+)_id\d+")


def parse(file_name):
    m = NAME_RE.search(os.path.basename(file_name))
    if not m:
        raise ValueError(f"cannot parse arrangement/object from {file_name}")
    return m.group("arr"), m.group("obj")


def choose_split(objs_by_arr, val_arr=None, test_arr=None):
    arrs = sorted(objs_by_arr, key=lambda a: int(a[1:]))
    candidates = []
    for test, val in itertools.permutations(arrs, 2):
        if (test_arr and test != test_arr) or (val_arr and val != val_arr):
            continue
        train = [a for a in arrs if a not in (test, val)]
        train_objs = set().union(*(objs_by_arr[a] for a in train))
        if objs_by_arr[test] <= train_objs:
            # prefer splits where valid is also fully covered by train
            candidates.append((not objs_by_arr[val] <= train_objs, test, val, train))
    if not candidates:
        raise RuntimeError("no valid/test pair whose objects are all covered by train")
    _, test, val, train = min(candidates)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_coco")
    ap.add_argument("--dst", default="dataset")
    ap.add_argument("--val", help="force validation arrangement, e.g. A9")
    ap.add_argument("--test", help="force test arrangement, e.g. A10")
    args = ap.parse_args()

    coco = json.load(open(os.path.join(args.src, "result.json")))

    objs_by_arr = defaultdict(set)
    arr_of_image = {}
    for img in coco["images"]:
        arr, obj = parse(img["file_name"])
        arr_of_image[img["id"]] = arr
        if obj != "no-pointing":
            objs_by_arr[arr].add(obj)

    train, val, test = choose_split(objs_by_arr, args.val, args.test)
    split_of_arr = {a: "train" for a in train} | {val: "valid", test: "test"}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    for split in ("train", "valid", "test"):
        out_dir = os.path.join(args.dst, split)
        os.makedirs(out_dir, exist_ok=True)
        images, annotations = [], []
        for img in coco["images"]:
            if split_of_arr[arr_of_image[img["id"]]] != split:
                continue
            name = os.path.basename(img["file_name"])
            shutil.copy2(os.path.join(args.src, img["file_name"]), os.path.join(out_dir, name))
            images.append({**img, "file_name": name})
            annotations.extend(anns_by_image[img["id"]])
        out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
        out.update(images=images, annotations=annotations)
        with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
            json.dump(out, f)
        arrs = sorted((a for a, s in split_of_arr.items() if s == split), key=lambda a: int(a[1:]))
        objs = sorted(set().union(*(objs_by_arr[a] for a in arrs)))
        print(f"{split:5s}: {len(images):3d} images, {len(annotations):3d} annotations, "
              f"arrangements={arrs}, objects={objs}")


if __name__ == "__main__":
    main()
