"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks on
NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

This runner is designed to work on macOS/Linux development machines without
requiring a local flashinfer-bench installation. Packing and benchmarking are
performed inside the Modal container.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def _load_config() -> Dict[str, Dict[str, str]]:
    """Load config.toml using tomllib when available, with a small fallback parser."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Python 3.11+ path
    try:
        import tomllib  # type: ignore

        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config
    except ModuleNotFoundError:
        pass

    # Minimal fallback parser for this repository's simple config structure.
    config = {"solution": {}, "build": {}}
    section = None
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line or section not in config:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        config[section][key] = value
    return config


def _resolve_definition_name(raw_definition: str, available_definitions: Iterable[str]) -> str:
    """Resolve shorthand track names to concrete definition names."""
    available = sorted(set(available_definitions))
    if raw_definition in available:
        return raw_definition

    alias_rules = {
        "fused_moe": lambda n: n.startswith("moe_"),
        "sparse_attention": lambda n: "sparse_attention" in n,
        "gated_delta_net": lambda n: n.startswith("gdn_"),
    }
    matcher = alias_rules.get(raw_definition)
    if matcher is None:
        raise ValueError(
            f"Definition '{raw_definition}' not found in trace set. "
            f"Available definitions: {', '.join(available)}"
        )

    candidates = [name for name in available if matcher(name)]
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"Definition alias '{raw_definition}' did not match any dataset definition. "
            f"Available definitions: {', '.join(available)}"
        )

    raise ValueError(
        f"Definition alias '{raw_definition}' is ambiguous. "
        f"Please set config.toml [solution].definition to one of: {', '.join(candidates)}"
    )


def _normalize_entry_point(language: str, entry_point: str) -> str:
    """Normalize legacy entry point names to file-qualified format."""
    if "::" in entry_point:
        return entry_point

    if language == "triton":
        return f"kernel.py::{entry_point}"
    if language == "cuda":
        return f"kernel.cu::{entry_point}"

    raise ValueError(f"Unsupported language in config.toml: {language}")


def _build_solution_payload() -> Dict[str, Any]:
    """Collect config and source files for remote packing."""
    config = _load_config()
    solution_cfg = config.get("solution", {})
    build_cfg = config.get("build", {})

    required_solution = ("name", "definition", "author")
    required_build = ("language", "entry_point")

    missing = [k for k in required_solution if not solution_cfg.get(k)]
    missing += [k for k in required_build if not build_cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required config fields in config.toml: {', '.join(sorted(missing))}")

    language = build_cfg["language"]
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language in config.toml: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    sources: List[Dict[str, str]] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(source_dir).as_posix()
            sources.append({"path": rel_path, "content": path.read_text()})

    if not sources:
        raise ValueError(f"No source files found in: {source_dir}")

    normalized_entry_point = _normalize_entry_point(
        build_cfg["language"],
        build_cfg["entry_point"],
    )

    return {
        "solution": {
            "name": solution_cfg["name"],
            "definition": solution_cfg["definition"],
            "author": solution_cfg["author"],
        },
        "build": {
            "language": build_cfg["language"],
            "entry_point": normalized_entry_point,
        },
        "sources": sources,
    }


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution_payload: Dict[str, Any], config: Dict[str, int] = None) -> Dict[str, Any]:
    """Run benchmark on Modal B200 and return results."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        config = BenchmarkConfig(**config)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    requested_definition = solution_payload["solution"]["definition"]
    resolved_definition = _resolve_definition_name(
        requested_definition,
        trace_set.definitions.keys(),
    )

    with TemporaryDirectory() as tmp_dir:
        source_dir = Path(tmp_dir) / "solution_src"
        source_dir.mkdir(parents=True, exist_ok=True)

        for source in solution_payload["sources"]:
            dst = source_dir / source["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(source["content"])

        build_spec = BuildSpec(
            language=solution_payload["build"]["language"],
            target_hardware=["cuda"],
            entry_point=solution_payload["build"]["entry_point"],
        )
        solution = pack_solution_from_files(
            path=str(source_dir),
            spec=build_spec,
            name=solution_payload["solution"]["name"],
            definition=resolved_definition,
            author=solution_payload["solution"]["author"],
        )

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Collect source files and run benchmark on Modal."""
    print("Collecting solution from source files...")
    solution_payload = _build_solution_payload()
    solution_meta = solution_payload["solution"]
    print(f"Prepared: {solution_meta['name']} ({solution_meta['definition']})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution_payload)

    if not results:
        print("No results returned!")
        return

    print_results(results)
