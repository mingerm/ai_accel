from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_report(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def as_float(data: dict[str, str], key: str) -> float:
    try:
        return float(data.get(key, "0").split()[0])
    except ValueError:
        return 0.0


def profile_summary(path: str) -> tuple[str, dict[str, float], dict[str, float]]:
    if not path:
        return "", {}, {}
    csv_path = Path(path)
    if not csv_path.exists():
        return "", {}, {}

    environment = ""
    process_ms: dict[str, float] = defaultdict(float)
    op_ms: dict[str, float] = defaultdict(float)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_type = row.get("record_type", "")
            section = row.get("section", "")
            value = float(row.get("value_ms", "0") or 0)
            if record_type == "environment" and not environment:
                environment = row.get("detail", "")
            elif record_type == "process":
                process_ms[section] += value
            elif record_type == "op":
                op_ms[section] += value
            elif record_type == "image":
                op_ms["image_latency_sum"] += value
                detail = row.get("detail", "")
                for item in detail.split(";"):
                    if "=" not in item:
                        continue
                    key, raw_value = item.split("=", 1)
                    if key == "file_io_preprocess_ms":
                        try:
                            op_ms["file_io_preprocess"] += float(raw_value)
                        except ValueError:
                            pass
    return environment, dict(process_ms), dict(op_ms)


def write_section_times(out: list[str], title: str, times: dict[str, float]) -> None:
    out.append("")
    out.append(title)
    for key in sorted(times):
        out.append(f"- {key}: {times[key]:.6f} ms")


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: compare_mnist_profiles.py <baseline_report> <optimized_report> <output_report>",
            file=sys.stderr,
        )
        return 2

    baseline_report = Path(sys.argv[1])
    optimized_report = Path(sys.argv[2])
    output_report = Path(sys.argv[3])

    baseline = parse_report(baseline_report)
    optimized = parse_report(optimized_report)
    baseline_avg = as_float(baseline, "average_latency_ms")
    optimized_avg = as_float(optimized, "average_latency_ms")
    baseline_total = as_float(baseline, "total_inference_time_ms")
    optimized_total = as_float(optimized, "total_inference_time_ms")
    baseline_acc = as_float(baseline, "accuracy")
    optimized_acc = as_float(optimized, "accuracy")

    speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0.0
    total_speedup = baseline_total / optimized_total if optimized_total > 0 else 0.0

    base_env, base_process, base_ops = profile_summary(baseline.get("mnist_profile_csv", ""))
    opt_env, opt_process, opt_ops = profile_summary(optimized.get("mnist_profile_csv", ""))

    lines: list[str] = []
    lines.append("# MNIST Pipeline Optimization Report")
    lines.append("")
    lines.append(f"execution_environment: {opt_env or base_env or optimized.get('execution_environment', '')}")
    lines.append("modified_files: mnistCUDNN.cpp, ocr.py, scripts/run_pipeline.sh, scripts/parse_mnist_batch_output.py")
    lines.append("modified_functions: network_t::classify_example, network_t::convoluteForward, network_t::resize, main")
    lines.append(
        "change_summary: Added CUDA-event profiling, reusable GPU buffers/workspace, per-shape convolution algorithm cache, and batch PGM inference."
    )
    lines.append("")
    lines.append("| metric | baseline | optimized | change |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| total inference time ms | {baseline_total:.3f} | {optimized_total:.3f} | {total_speedup:.3f}x |")
    lines.append(f"| average latency ms/image | {baseline_avg:.3f} | {optimized_avg:.3f} | {speedup:.3f}x |")
    lines.append(f"| accuracy | {baseline_acc:.6f} | {optimized_acc:.6f} | {optimized_acc - baseline_acc:+.6f} |")
    lines.append("")
    lines.append("Digit accuracy")
    for key in sorted(k for k in optimized if k.startswith("digit_") and k.endswith("_accuracy")):
        lines.append(f"- {key}: baseline={baseline.get(key, 'n/a')}, optimized={optimized.get(key, 'n/a')}")

    write_section_times(lines, "Baseline process timings", base_process)
    write_section_times(lines, "Optimized process timings", opt_process)
    write_section_times(lines, "Baseline CUDA/op timings", base_ops)
    write_section_times(lines, "Optimized CUDA/op timings", opt_ops)

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
