"""Build a frozen observation patch from tool requests, never question answers."""

from __future__ import annotations

import argparse
import ast
import csv
import filecmp
import itertools
import json
import re
import sys
import gc
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.perception_prompts import condition_signature, normalize_prompt
from agent.perception_patch import equivalent_results

RUNS = [
    "earthbench_qwen36_27b_vllm020_native_20261029_212157",
    "earthbench_qwen35_9b_vllm020_native_20261030_100835",
    "earthbench_qwen36_27b_live_perception_20261030_220450",
]
TOOLS = {"MSCN", "RemoteCLIP", "SM3Det", "Strip_R_CNN", "RemoteSAM", "InstructSAM", "SAM2", "ChangeOS"}
DATASET = (ROOT / "benchmark/earthbench").resolve()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def input_path(value):
    if not isinstance(value, str):
        raise ValueError("Image path must be a string")
    match = re.search(r"(?:^|/)(data/question\d+/[^\n]+)$", value.replace("\\", "/"))
    if not match:
        raise ValueError("Input is not a released dataset image")
    relative = match[1]
    path = (DATASET / relative).resolve()
    if not path.is_relative_to(DATASET / "data") or not path.is_file():
        raise ValueError("Dataset image does not exist")
    return relative


def collect_requests():
    unique = {}
    invalid = []
    totals = {}
    for run in RUNS:
        rows = [json.loads(line) for line in (ROOT / "runs" / run / "result/results.jsonl").read_text().splitlines()]
        if len(rows) != 248:
            raise ValueError(f"Expected 248 trajectories: {run}")
        totals[run] = len(rows)
        for row in rows:
            for call in row.get("tool_calls", []):
                if call.get("name") not in TOOLS:
                    continue
                tool = call["name"].replace("Strip_R_CNN", "Strip-R-CNN")
                arguments = call.get("input", {})
                provenance = {"run": run, "question": str(row["question_id"])}
                try:
                    expected = ({"pre_image_path", "post_image_path", "output_path"} if tool == "ChangeOS" else
                                {"input_image_path", "bboxes", "output_path"} if tool == "SAM2" else
                                {"input_image_path"} if tool in {"MSCN", "RemoteCLIP"} else
                                {"input_image_path", "text_prompt"})
                    if set(arguments) != expected:
                        raise ValueError("Arguments do not match the tool schema")
                    paths = ([input_path(arguments["pre_image_path"]), input_path(arguments["post_image_path"])]
                             if tool == "ChangeOS" else [input_path(arguments["input_image_path"])])
                    prompt = arguments.get("text_prompt", "")
                    if not isinstance(prompt, str):
                        raise ValueError("text_prompt must be a string")
                    bbox = arguments.get("bboxes")
                    request = {"model": tool, "paths": paths, "text_prompt": prompt, "bbox": bbox}
                    key = json.dumps(request, sort_keys=True)
                    unique.setdefault(key, {**request, "sources": []})["sources"].append(provenance)
                except (ValueError, TypeError, KeyError) as error:
                    invalid.append({**provenance, "tool": tool, "input": arguments, "reason": str(error)})
    return list(unique.values()), invalid, totals


def released_rows():
    with (DATASET / "perception/observations.csv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    observations = []
    for index, row in enumerate(rows):
        raw = row["file_path"].strip()
        paths = [p.strip() for p in raw[1:-1].split(",")] if raw.startswith("(") else [raw]
        try:
            result = ast.literal_eval(row["result"])
        except (ValueError, SyntaxError):
            result = row["result"]
        observations.append({"id": f"released-{index}", "model": row["model"], "paths": paths,
                             "text_prompt": row["text_prompt"], "bbox": ast.literal_eval(row["bbox"]) if row["bbox"] else None,
                             "result": result, "provenance": {"source": "released-observation", "row": index}})
    return observations


def identical_paths(requests, rows):
    groups = defaultdict(list)
    paths = {path for item in requests + rows for path in item["paths"]}
    for path in sorted(paths):
        groups[(DATASET / path).stat().st_size].append(path)
    aliases = {}
    for same_size in groups.values():
        equivalents = []
        for path in same_size:
            for group in equivalents:
                if filecmp.cmp(DATASET / path, DATASET / group[0], shallow=False):
                    group.append(path)
                    break
            else:
                equivalents.append([path])
        for group in equivalents:
            for path in group:
                aliases[path] = group
    return aliases


def key(item):
    return (item["model"], tuple(item["paths"]), normalize_prompt(item["text_prompt"]),
            json.dumps(item["bbox"], sort_keys=True))


def audit(directory):
    requests, invalid, runs = collect_requests()
    rows = released_rows()
    equivalents = identical_paths(requests, rows)
    expanded = []
    for row in rows:
        for paths in itertools.product(*(equivalents[path] for path in row["paths"])):
            expanded.append({**row, "paths": list(paths)})
    known = {key(row) for row in expanded}
    for request in requests:
        request["released_hit"] = key(request) in known
    by_tool = {}
    for model in sorted({r["model"] for r in requests}):
        items = [r for r in requests if r["model"] == model]
        by_tool[model] = {"unique_requests": len(items), "released_hit": sum(r["released_hit"] for r in items)}
    report = {"runs": runs, "trajectories": sum(runs.values()), "requests": len(requests),
              "invalid_invocations": len(invalid), "by_tool": by_tool,
              "identical_image_groups": [group for path, group in equivalents.items() if len(group) > 1 and path == group[0]]}
    write(directory / "requests.json", requests)
    write(directory / "invalid_invocations.json", invalid)
    write(directory / "released_expanded.json", expanded)
    write(directory / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def proposals(directory):
    """Generate consensus DOTA detections; existing observations always win."""
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    torch.set_num_threads(4)
    requests = json.loads((directory / "requests.json").read_text())
    original = json.loads((directory / "released_expanded.json").read_text())
    original_keys = {key(row) for row in original}
    models_root = Path("/home/dijkstra/model/earthbench-cache-repair")
    models = [YOLO(models_root / filename) for filename in ["yolo11l-obb.pt", "yolo11x-obb.pt"]]
    from ultralytics.utils.metrics import batch_probiou
    from ultralytics.utils.nms import TorchNMS
    from ultralytics.utils.ops import xywhr2xyxyxyxy
    paths = sorted({r["paths"][0] for r in requests if r["model"] in {"SM3Det", "Strip-R-CNN", "InstructSAM", "RemoteSAM"} and not r["released_hit"]})
    detections = {}
    saved = directory / "detections.json"
    if saved.exists():
        detections = json.loads(saved.read_text())

    def iou(a, b):
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        return overlap / max(area_a + area_b - overlap, 1e-9)

    for index, path in enumerate(paths):
        if path in detections:
            continue
        image = Image.open(DATASET / path).convert("RGB")
        outputs = []
        for model in models:
            views = [(image, 0, 0)]
            if max(image.size) > 1280:
                for y in sorted(set(list(range(0, max(image.height-1024, 1), 768)) + [max(image.height-1024, 0)])):
                    for x in sorted(set(list(range(0, max(image.width-1024, 1), 768)) + [max(image.width-1024, 0)])):
                        views.append((image.crop((x, y, min(x+1024, image.width), min(y+1024, image.height))), x, y))
            oriented, scores, classes = [], [], []
            for view, x, y in views:
                result = model.predict(view, imgsz=1024, conf=0.35, iou=0.5, max_det=3000, device=0, verbose=False)[0]
                coordinates = result.obb.xywhr.clone()
                coordinates[:, 0] += x
                coordinates[:, 1] += y
                oriented.append(coordinates)
                scores.append(result.obb.conf)
                classes.append(result.obb.cls)
            oriented, scores, classes = torch.cat(oriented), torch.cat(scores), torch.cat(classes)
            keep = []
            for category in classes.unique():
                positions = torch.where(classes == category)[0]
                selected = TorchNMS.fast_nms(oriented[positions], scores[positions], 0.5, iou_func=batch_probiou)
                keep.extend(positions[selected].cpu().tolist())
            keep = torch.tensor(keep, device=oriented.device, dtype=torch.long)
            corners = xywhr2xyxyxyxy(oriented[keep])
            boxes = torch.cat((corners.amin(dim=1), corners.amax(dim=1)), dim=1).cpu().numpy()
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, image.width)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, image.height)
            outputs.append([{"category": model.names[int(c)], "score": float(s), "bbox": [round(float(v), 3) for v in box]}
                            for c, s, box in zip(classes[keep], scores[keep], boxes)])
        agreed = {}
        for category in models[0].names.values():
            left = [item for item in outputs[0] if item["category"] == category]
            right = [item for item in outputs[1] if item["category"] == category]
            if len(left) != len(right):
                continue
            remaining = list(right)
            match = True
            for item in left:
                best = max(range(len(remaining)), key=lambda i: iou(item["bbox"], remaining[i]["bbox"]))
                if iou(item["bbox"], remaining[best]["bbox"]) < 0.5:
                    match = False
                    break
                remaining.pop(best)
            if match:
                agreed[category] = [item["bbox"] for item in right]
        detections[path] = {"size": list(image.size), "models": outputs, "agreed": agreed}
        write(saved, detections)
        print(f"DOTA {index+1}/{len(paths)} {path} agreed={len(agreed)}/15", flush=True)
    observations = []
    for request in requests:
        if request["released_hit"] or request["paths"][0] not in detections:
            continue
        tool = request["model"]
        if tool not in {"SM3Det", "Strip-R-CNN", "InstructSAM", "RemoteSAM"}:
            continue
        prompt = normalize_prompt(request["text_prompt"])
        if tool == "Strip-R-CNN" and prompt != "ship":
            continue
        # Only complete category requests. A qualifier must never disappear.
        agreed = detections[request["paths"][0]]["agreed"]
        if prompt not in agreed:
            continue
        boxes = agreed[prompt]
        # Two detectors can miss the same small target. Absence is not evidence
        # strong enough to add a new negative observation to this benchmark.
        if not boxes:
            continue
        if tool == "RemoteSAM" and len(boxes) != 1:
            continue
        result = len(boxes) if tool == "InstructSAM" else boxes[0] if tool == "RemoteSAM" else boxes
        observations.append({**{k: request[k] for k in ["model", "paths", "text_prompt", "bbox"]},
                             "id": f"dota-{len(observations)}", "result": result,
                             "provenance": {"source": "model-consensus", "models": ["YOLO11l-OBB-DOTA", "YOLO11x-OBB-DOTA"],
                                            "confidence_threshold": 0.35, "agreement_iou": 0.5,
                                            "same_count": True, "image_size": 1024, "tile_overlap": 256,
                                            "original_tool_weights": False}})
    # Derive compatible observations from complete released detections, while
    # preserving provenance rather than claiming the original tool was run.
    for request in requests:
        if request["model"] not in {"InstructSAM", "RemoteSAM", "Strip-R-CNN"} or request["released_hit"]:
            continue
        prompt = normalize_prompt(request["text_prompt"])
        if request["model"] == "Strip-R-CNN" and prompt != "ship":
            continue
        candidates = [r for r in original if r["model"] == "SM3Det" and r["paths"] == request["paths"] and normalize_prompt(r["text_prompt"]) == prompt]
        if not candidates or any(r["result"] != candidates[0]["result"] for r in candidates):
            continue
        boxes = candidates[0]["result"]
        if not isinstance(boxes, list) or any(not isinstance(b, list) or len(b) != 4 for b in boxes):
            continue
        if request["model"] == "RemoteSAM" and len(boxes) != 1:
            continue
        result = len(boxes) if request["model"] == "InstructSAM" else boxes[0] if request["model"] == "RemoteSAM" else boxes
        observations = [r for r in observations if key(r) != key(request)]
        observations.append({**{k: request[k] for k in ["model", "paths", "text_prompt", "bbox"]},
                             "result": result,
                             "provenance": {"source": "derive-released-detections", "operation": "count" if request["model"] == "InstructSAM" else "unique-box" if request["model"] == "RemoteSAM" else "copy-ship-detections",
                                            "observation": candidates[0]["id"], "original_tool_weights": False}})
    for index, observation in enumerate(observations):
        observation["id"] = f"detection-{index}"
    write(directory / "detection_proposals.json", observations)
    print("Detection proposals:", len(observations), flush=True)


def semantic_aliases(directory):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    path = "/home/dijkstra/model/earthbench-cache-repair/nli-deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval().cuda()
    entailment = model.config.label2id["entailment"]
    requests = json.loads((directory / "requests.json").read_text())
    observations = json.loads((directory / "released_expanded.json").read_text())
    for filename in ["detection_proposals.json", "classification_proposals.json", "mask_proposals.json"]:
        if (directory / filename).exists():
            observations += json.loads((directory / filename).read_text())
    existing = {key(row) for row in observations}
    aliases, rejected = [], []
    for request in requests:
        if key(request) in existing or not request["text_prompt"]:
            continue
        candidates = [row for row in observations if row["model"] == request["model"] and row["paths"] == request["paths"] and row["bbox"] == request["bbox"]]
        matches = []
        for candidate in candidates:
            query_signature = condition_signature(request["text_prompt"])
            candidate_signature = condition_signature(candidate["text_prompt"])
            if query_signature["conditions"] != candidate_signature["conditions"]:
                continue
            if query_signature["categories"] and candidate_signature["categories"] and query_signature["categories"] != candidate_signature["categories"]:
                continue
            left, right = normalize_prompt(request["text_prompt"]), normalize_prompt(candidate["text_prompt"])
            generic = {"field", "court", "tank", "object", "sports", "sport", "sports field", "sports court", "sports infrastructure", "sports facility", "stadium", "baseball stadium", "sports complex", "playground", "white circle", "cylinder"}
            if left != right and (left in generic or right in generic):
                continue
            for a, b in [("leftmost", "westernmost"), ("rightmost", "easternmost"), ("topmost", "northernmost"), ("bottommost", "southernmost")]:
                left, right = left.replace(a, b), right.replace(a, b)
            inputs = tokenizer([f"The requested target is {left}.", f"The requested target is {right}."],
                               [f"The requested target is {right}.", f"The requested target is {left}."],
                               return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            with torch.inference_mode():
                probabilities = model(**inputs).logits.softmax(-1)[:, entailment].cpu().tolist()
            if min(probabilities) >= 0.98:
                matches.append((candidate, probabilities))
        if matches and all(c["result"] == matches[0][0]["result"] for c, _ in matches):
            candidate, probabilities = max(matches, key=lambda match: min(match[1]))
            aliases.append({**{k: request[k] for k in ["model", "paths", "text_prompt", "bbox"]},
                            "id": f"semantic-alias-{len(aliases)}", "result": candidate["result"],
                            "provenance": {"source": "offline-semantic-alias", "observation": candidate["id"],
                                           "original_prompt": candidate["text_prompt"], "nli_model": "cross-encoder/nli-deberta-v3-small",
                                           "bidirectional_entailment": probabilities}})
        else:
            rejected.append({"model": request["model"], "paths": request["paths"], "text_prompt": request["text_prompt"]})
    write(directory / "semantic_aliases.json", aliases)
    write(directory / "semantic_unmatched.json", rejected)
    print("Semantic aliases:", len(aliases), "unmatched:", len(rejected), flush=True)


def classifications(directory):
    import open_clip
    import torch
    from PIL import Image

    torch.set_num_threads(4)
    requests = [r for r in json.loads((directory / "requests.json").read_text()) if r["model"] in {"MSCN", "RemoteCLIP"} and not r["released_hit"]]
    aid = ["Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center", "Church", "Commercial", "DenseResidential", "Desert", "Farmland", "Forest", "Industrial", "Meadow", "MediumResidential", "Mountain", "Park", "Parking", "Playground", "Pond", "Port", "RailwayStation", "Resort", "River", "School", "SparseResidential", "Square", "Stadium", "StorageTanks", "Viaduct"]
    remote = ["Airport", "Beach", "Bridge", "Commercial", "Desert", "Farmland", "footballField", "Forest", "Industrial", "Meadow", "Mountain", "Park", "Parking", "Pond", "Port", "RailwayStation", "Residential", "River", "Viaduct"]
    predictions = {}
    configurations = [("ViT-L-14", "/home/dijkstra/model/earthbench-cache-repair/RemoteCLIP-ViT-L-14.pt"),
                      ("ViT-B-32", "/home/dijkstra/model/GeoRSCLIP/RS5M_ViT-B-32.pt")]
    for architecture, checkpoint in configurations:
        model, _, transform = open_clip.create_model_and_transforms(architecture, pretrained=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = {k.removeprefix("module."): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state)
        model.eval().cuda()
        tokenizer = open_clip.get_tokenizer(architecture)
        features = {}
        with torch.inference_mode():
            for tool, labels in [("MSCN", aid), ("RemoteCLIP", remote)]:
                vectors = []
                for label in labels:
                    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label).lower()
                    tokens = tokenizer([f"a satellite image of {text}.", f"an aerial photograph of {text}.", f"a remote sensing image of {text}."]).cuda()
                    vector = model.encode_text(tokens)
                    vector /= vector.norm(dim=-1, keepdim=True)
                    vector = vector.mean(0)
                    vectors.append(vector / vector.norm())
                features[tool] = torch.stack(vectors)
            for index, request in enumerate(requests):
                image = Image.open(DATASET / request["paths"][0]).convert("RGB")
                vector = model.encode_image(transform(image).unsqueeze(0).cuda())
                vector /= vector.norm(dim=-1, keepdim=True)
                probabilities = (100 * vector @ features[request["model"]].T).softmax(dim=-1)[0]
                scores, indices = probabilities.topk(5)
                labels = aid if request["model"] == "MSCN" else remote
                predictions.setdefault(str(index), []).append({"checkpoint": checkpoint, "predictions": [[labels[i], float(s)] for i, s in zip(indices.tolist(), scores.tolist())]})
        del model, state, features
        gc.collect()
        torch.cuda.empty_cache()
        print("Classified", len(requests), architecture, flush=True)
    proposals = []
    for index, request in enumerate(requests):
        evidence = predictions[str(index)]
        left, right = [e["predictions"] for e in evidence]
        if left[0][0] != right[0][0] or any(pred[0][1] - pred[1][1] < 0.1 for pred in [left, right]):
            continue
        top = left
        result = {"predicted_class": top[0][0], "confidence": top[0][1],
                  "top5_predictions": top if request["model"] == "MSCN" else
                  [{"rank": rank, "class": label, "probability": score} for rank, (label, score) in enumerate(top, 1)]}
        proposals.append({**{k: request[k] for k in ["model", "paths", "text_prompt", "bbox"]},
                          "id": f"scene-{len(proposals)}", "result": result,
                          "provenance": {"source": "model-consensus", "models": ["RemoteCLIP-ViT-L-14", "GeoRSCLIP-RS5M-ViT-B-32"],
                                         "minimum_top1_margin": 0.1, "original_tool_weights": request["model"] == "RemoteCLIP"}})
    write(directory / "classification_evidence.json", {"requests": requests, "predictions": predictions})
    write(directory / "classification_proposals.json", proposals)
    print("Classification proposals:", len(proposals), flush=True)


def masks(directory):
    import numpy as np
    import torch
    import changeos
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(4)
    requests = [r for r in json.loads((directory / "requests.json").read_text()) if r["model"] in {"ChangeOS", "SAM2"} and not r["released_hit"]]
    observations = []
    checkpoint = "/home/dijkstra/model/ChangeOS/changeos_r101.pt"
    model = changeos.ChangeOS(torch.jit.load(checkpoint, map_location="cpu").eval())
    for request in [r for r in requests if r["model"] == "ChangeOS"]:
        pre, post = [np.array(Image.open(DATASET / path).convert("RGB")) for path in request["paths"]]
        if pre.shape != post.shape:
            continue
        height, width = pre.shape[:2]
        selected = np.zeros((height, width), dtype=bool)
        # Full 1024-pixel tiles preserve native resolution and bound memory.
        for y in range(0, height, 1024):
            for x in range(0, width, 1024):
                h, w = min(1024, height-y), min(1024, width-x)
                first = np.pad(pre[y:y+h, x:x+w], ((0, (-h)%32), (0, (-w)%32), (0, 0)), mode="edge")
                second = np.pad(post[y:y+h, x:x+w], ((0, (-h)%32), (0, (-w)%32), (0, 0)), mode="edge")
                with torch.inference_mode():
                    localization, damage = model(first, second)
                mask = localization if request["paths"][0] == request["paths"][1] else damage > 1
                selected[y:y+h, x:x+w] = mask[:h, :w]
        identity = f"changeos-{len(observations)}"
        destination = directory / "artifacts" / f"{identity}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(selected.astype(np.uint8)*255).save(destination)
        observations.append({**{k: request[k] for k in ["model", "paths", "text_prompt", "bbox"]}, "id": identity,
                             "artifact": str(destination.relative_to(directory)), "result": None,
                             "provenance": {"source": "original-model-inference", "model": "ChangeOS-ResNet101", "tile_size": 1024, "original_tool_weights": True}})
        print("ChangeOS", request["paths"], int(selected.sum()), flush=True)
        write(directory / "mask_proposals.json", observations)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "/home/dijkstra/model/sam2.1_hiera_large/sam2.1_hiera_large.pt", device="cuda"))
    for request in [r for r in requests if r["model"] == "SAM2"]:
        image = np.array(Image.open(DATASET / request["paths"][0]).convert("RGB"))
        boxes = np.asarray(request["bbox"], dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or np.any(boxes[:, 2:] <= boxes[:, :2]):
            continue
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            predictions, _, _ = predictor.predict(box=boxes, multimask_output=False)
        selected = np.any(predictions.reshape(-1, *predictions.shape[-2:]), axis=0)
        identity = f"sam2-{len(observations)}"
        destination = directory / "artifacts" / f"{identity}.png"
        Image.fromarray(selected.astype(np.uint8)*255).save(destination)
        observations.append({**{k: request[k] for k in ["model", "paths", "text_prompt", "bbox"]}, "id": identity,
                             "artifact": str(destination.relative_to(directory)), "result": None,
                             "provenance": {"source": "original-model-inference", "model": "SAM2.1-Hiera-Large", "original_tool_weights": True}})
        print("SAM2", request["paths"], int(selected.sum()), flush=True)
        write(directory / "mask_proposals.json", observations)


def freeze(directory, destination):
    """Install only explicitly reviewed observations, with release precedence."""
    review = json.loads((directory / "review.json").read_text())
    released = released_rows()
    expanded = json.loads((directory / "released_expanded.json").read_text())
    baseline = {key(row): row for row in released}
    accepted = {}
    accepted_ids = {r["id"] for r in released}
    proposals = []
    for filename in ["detection_proposals.json", "classification_proposals.json",
                     "mask_proposals.json", "semantic_aliases.json"]:
        proposals += json.loads((directory / filename).read_text())
    if review["reviewed_observations"] != proposals:
        raise ValueError("Proposals changed after review; review their current contents again")
    ids = {row["id"] for row in proposals}
    if ids != set(review["decisions"]):
        raise ValueError("Every proposal requires an explicit review decision")
    candidates = [r for r in proposals if review["decisions"][r["id"]]["accept"]]

    def same_result(left, right):
        def value(row):
            return str(directory / row["artifact"]) if "artifact" in row else row["result"]
        return equivalent_results(value(left), value(right), DATASET)

    # Only identical image bytes may transfer a released observation to a new path.
    for index, row in enumerate(expanded):
        if key(row) in baseline:
            continue
        candidates.insert(0, {**row, "id": f"identical-{index}",
                              "provenance": {"source": "byte-identical-image",
                                             "observation": row["id"]}})
    for row in candidates:
        row = dict(row)
        identity = key(row)
        if row["provenance"]["source"] == "offline-semantic-alias":
            source = row["provenance"]["observation"]
            if source not in accepted_ids:
                raise ValueError(f"Alias depends on an unaccepted observation: {source}")
        if identity in baseline:
            if not same_result(row, baseline[identity]):
                raise ValueError(f"Proposal conflicts with release: {row['id']}")
            accepted_ids.add(row["id"])
            continue
        if identity in accepted:
            other = accepted[identity]
            if not same_result(row, other):
                raise ValueError(f"Conflicting proposals: {row['id']}, {other['id']}")
            accepted_ids.add(row["id"])
            continue
        accepted[identity] = row
        accepted_ids.add(row["id"])
    # Expand reviewed results over identical bytes too, preserving image order.
    equivalents = identical_paths(json.loads((directory / "requests.json").read_text()), released)
    for row in list(accepted.values()):
        for paths in itertools.product(*(equivalents[path] for path in row["paths"])):
            candidate = {**row, "paths": list(paths)}
            identity = key(candidate)
            other = baseline.get(identity, accepted.get(identity))
            if other is not None and not same_result(candidate, other):
                raise ValueError(f"Conflicting identical-image observations: {row['id']}, {other['id']}")
            if identity not in baseline and identity not in accepted:
                candidate["id"] = f"{row['id']}-image-alias-{len(accepted)}"
                accepted[identity] = candidate
    destination.mkdir(parents=True, exist_ok=True)
    for row in accepted.values():
        if "artifact" in row:
            source = directory / row["artifact"]
            target = destination / row["artifact"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    document = {"schema_version": 1, "dataset": "earthbench", "version": "trajectory-repair-v1",
                "source_runs": RUNS, "review": review,
                "policy": "Released results preserved; reviewed observations only; no answer-derived data; unknown requests remain errors.",
                "observations": list(accepted.values())}
    write(destination / "observations.json", document)
    print("Installed observations:", len(accepted), "at", destination, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "detections", "aliases", "classifications", "masks", "freeze"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.destination is None:
            parser.error("freeze requires --destination")
        freeze(args.directory, args.destination)
        return
    {"audit": audit, "detections": proposals, "aliases": semantic_aliases, "classifications": classifications, "masks": masks}[args.command](args.directory)


if __name__ == "__main__":
    main()
