import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    ego_kart = next((kart for kart in karts if kart["is_center_kart"]), None)
    captions = [
        f"There are {len(karts)} karts in the scene.",
        f"The track is {track_name}.",
    ]
    if ego_kart is None:
        return captions

    ego_name = ego_kart["kart_name"]
    ego_x, ego_y = ego_kart["center"]
    captions.insert(0, f"{ego_name} is the ego car.")
    for kart in karts:
        if kart["is_center_kart"]:
            continue

        kart_x, kart_y = kart["center"]
        name = kart["kart_name"]
        horizontal = "left" if kart_x < ego_x else "right"
        vertical = "front" if kart_y < ego_y else "back"
        captions.append(f"{name} is {horizontal} of the ego car.")
        if vertical == "front":
            captions.append(f"{name} is in front of the ego car.")
        else:
            captions.append(f"{name} is behind the ego car.")
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_dataset(
    output_json: str,
    data_dir: str = "data",
    split: str = "train",
):
    """
    Generate image-caption pairs for all training images.
    """
    if split != "train":
        raise ValueError("Generate captions only for the training split.")

    split_dir = Path(data_dir) / split
    all_caption_pairs = []
    for info_path in sorted(split_dir.glob("*_info.json")):
        with open(info_path) as f:
            info = json.load(f)
        frame_name = info_path.stem.replace("_info", "")
        for view_index in range(len(info["detections"])):
            image_name = f"{frame_name}_{view_index:02d}_im.jpg"
            image_path = split_dir / image_name
            if not image_path.exists():
                continue

            captions = generate_caption(str(info_path), view_index)
            for caption in captions:
                all_caption_pairs.append(
                    {
                        "image_file": f"{split}/{image_name}",
                        "caption": caption,
                    }
                )
    with open(output_json, "w") as f:
        json.dump(all_caption_pairs, f)
    print(f"Wrote {len(all_caption_pairs)} caption pairs to {output_json}")
    

"""
Usage Example: Visualize QA pairs for a specific file and view:
    python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire(
        {
            "check": check_caption,
            "generate": generate_dataset,
        }
    )


if __name__ == "__main__":
    main()
