from __future__ import annotations

import re
import sys


text = sys.stdin.read()

prediction_patterns = [
    r"(?:추론|예측)\s*[:=]\s*([0-9])",
    r"(?:prediction|predicted|predict|result|class|digit)\D{0,20}([0-9])",
    r"classified\s+as\D{0,20}([0-9])",
]

time_patterns = [
    r"Inference\s*time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    r"(?:latency|elapsed|time)\D{0,20}([0-9]+(?:\.[0-9]+)?)\s*ms",
]


def find_first(patterns: list[str], source: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


prediction = find_first(prediction_patterns, text)
elapsed_ms = find_first(time_patterns, text)

if prediction is None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"[0-9]", line):
            prediction = line
            break

print(f"{prediction or '?'} {elapsed_ms or '0.000'}")
