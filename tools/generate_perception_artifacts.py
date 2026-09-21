from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _local_input(path: str, input_root: Path) -> Path:
    prefix = "benchmark/data/"
    if not path.startswith(prefix):
        raise ValueError(f"unexpected benchmark input path: {path}")
    return input_root / path.removeprefix(prefix)


def _output_path(arguments: dict[str, Any], output_root: Path) -> Path:
    relative = Path(arguments["output_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid output path: {relative}")
    path = output_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _generate_sam2(
    calls: list[dict[str, Any]],
    input_root: Path,
    output_root: Path,
    checkpoint: Path,
) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(checkpoint),
        device="cuda",
    )
    predictor = SAM2ImagePredictor(model)
    for call in calls:
        arguments = call["input"]
        image = np.asarray(
            Image.open(_local_input(arguments["input_image_path"], input_root)).convert("RGB")
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                box=np.asarray(arguments["bboxes"], dtype=np.float32),
                multimask_output=False,
            )
        combined = np.any(masks.reshape(-1, *masks.shape[-2:]), axis=0)
        destination = _output_path(arguments, output_root)
        Image.fromarray(combined.astype(np.uint8) * 255).save(destination)
        print(f"SAM2: {destination}")


def _generate_changeos(
    calls: list[dict[str, Any]], input_root: Path, output_root: Path
) -> None:
    import changeos
    import numpy as np
    import torch
    from PIL import Image

    model = changeos.from_name("changeos_r101")
    for call in calls:
        arguments = call["input"]
        pre = np.asarray(
            Image.open(_local_input(arguments["pre_image_path"], input_root)).convert("RGB")
        )
        post = np.asarray(
            Image.open(_local_input(arguments["post_image_path"], input_root)).convert("RGB")
        )
        with torch.inference_mode():
            localization, damage = model(pre, post)
        mask = localization if arguments["pre_image_path"] == arguments["post_image_path"] else damage > 1
        destination = _output_path(arguments, output_root)
        Image.fromarray(mask.astype(np.uint8) * 255).save(destination)
        print(f"ChangeOS: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the one-time SAM2 and ChangeOS cache artifacts on a GPU host."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    calls = [call for question in ground_truth for call in question["tool_calls"]]
    sam2_calls = [call for call in calls if call["name"] == "SAM2"]
    changeos_calls = [call for call in calls if call["name"] == "ChangeOS"]
    if len(sam2_calls) != 6 or len(changeos_calls) != 17:
        raise ValueError(
            f"expected 6 SAM2 and 17 ChangeOS calls; got "
            f"{len(sam2_calls)} and {len(changeos_calls)}"
        )

    _generate_sam2(sam2_calls, args.input_root, args.output_root, args.sam2_checkpoint)
    _generate_changeos(changeos_calls, args.input_root, args.output_root)


if __name__ == "__main__":
    main()
