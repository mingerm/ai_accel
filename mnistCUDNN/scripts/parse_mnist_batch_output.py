from __future__ import annotations

import re
import sys


pattern = re.compile(
    r"^MNIST_RESULT\s+input=(?P<input>\S+)\s+prediction=(?P<prediction>[0-9?])\s+latency_ms=(?P<latency>[0-9]+(?:\.[0-9]+)?)"
)

for raw in sys.stdin:
    line = raw.strip()
    match = pattern.match(line)
    if not match:
        continue
    print(
        f"{match.group('input')}\t{match.group('prediction')}\t{match.group('latency')}"
    )
