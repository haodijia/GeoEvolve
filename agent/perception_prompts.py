"""Conservative prompt keys for offline perception observations."""

import re


CLASSES = {
    "plane": ("plane", "planes", "airplane", "airplanes", "aircraft", "aeroplane", "aeroplanes"),
    "ship": ("ship", "ships", "vessel", "vessels"),
    "storage tank": ("storage tank", "storage tanks"),
    "baseball diamond": ("baseball diamond", "baseball diamonds", "baseball field", "baseball fields"),
    "tennis court": ("tennis court", "tennis courts"),
    "basketball court": ("basketball court", "basketball courts", "basketball field", "basketball fields"),
    "ground track field": ("ground track field", "ground track fields", "running track", "running tracks", "athletics track", "athletics tracks", "track and field"),
    "soccer ball field": ("soccer ball field", "soccer ball fields", "soccer field", "soccer fields", "soccer pitch", "soccer pitches", "football field", "football fields", "football pitch", "football pitches"),
    "harbor": ("harbor", "harbors", "harbour", "harbours", "port", "ports"),
    "bridge": ("bridge", "bridges"),
    "large vehicle": ("large vehicle", "large vehicles"),
    "small vehicle": ("small vehicle", "small vehicles"),
    "vehicle": ("vehicle", "vehicles"),
    "helicopter": ("helicopter", "helicopters"),
    "roundabout": ("roundabout", "roundabouts", "traffic circle", "traffic circles"),
    "swimming pool": ("swimming pool", "swimming pools", "pool", "pools"),
    "building": ("building", "buildings"),
}
ALIASES = {alias: label for label, aliases in CLASSES.items() for alias in aliases}
PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True)) + r")\b")


def normalize_prompt(value: str) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    text = PATTERN.sub(lambda match: ALIASES[match[0]], text)
    text = re.sub(r"^(?:count (?:the )?(?:number of )?|detect (?:all )?|find |locate |search (?:for )?)", "", text)
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    for direction, adjective in (("west", "westernmost"), ("east", "easternmost"), ("north", "northernmost"), ("south", "southernmost")):
        text = re.sub(rf"\b(?:furthest|farthest) {direction}\b", adjective, text)
        text = re.sub(rf"\b(?:(?:located|situated) )?(?:on|in|at) (?:the )?{adjective} (?:side|position)\b", adjective, text)
    return " ".join(text.split())


def condition_signature(value: str) -> dict:
    """A veto for dangerous NLI false positives, not an equivalence decision."""
    text = normalize_prompt(value)
    categories = sorted(set(match[0] for match in PATTERN.finditer(text)))
    protected = re.findall(
        r"\b(?:westernmost|easternmost|northernmost|southernmost|west|east|north|south|"
        r"leftmost|rightmost|topmost|bottommost|left|right|above|below|between|near|"
        r"largest|smallest|biggest|small|large|big|white|black|red|green|blue|"
        r"first|second|third|fourth|fifth|last|not|no|without|except|ruined|destroyed|"
        r"damaged|undamaged|runway|river|individual|all|\d+)\b", text,
    )
    # Direction expressions are equivalent only under the benchmark's image axes.
    directions = {"leftmost": "westernmost", "rightmost": "easternmost", "topmost": "northernmost", "bottommost": "southernmost", "biggest": "largest"}
    return {"categories": categories, "conditions": sorted(directions.get(word, word) for word in protected)}
