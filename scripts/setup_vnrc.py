"""Fetch pinned sources and build VNRC with assertions enabled; no system installs."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/huawei-csl/mlsys-2026-graph-scheduling-vnrc-submission.git"
REVISION = "1f19c2ffeae394ab7c5f1d92c06795debf667758"
JSON_VERSION = "v3.11.3"
JSON_SHA256 = "9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6"


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--backend-only", action="store_true",
                        help="build the RL shared library without the upstream solver and its 48 test programs")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = ROOT / "external/vnrc"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", URL, str(source))
        run("git", "-C", str(source), "checkout", "--detach", REVISION)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected VNRC {REVISION}, found {revision}; leaving checkout untouched")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        raise SystemExit("VNRC checkout is modified; use a clean pinned checkout")
    deps = ROOT / "external/deps"
    header = deps / "nlohmann/json.hpp"
    header.parent.mkdir(parents=True, exist_ok=True)
    for filename, remote in ((header, "single_include/nlohmann/json.hpp"),
                             (header.parent / "LICENSE.MIT", "LICENSE.MIT")):
        if not filename.exists():
            url = f"https://raw.githubusercontent.com/nlohmann/json/{JSON_VERSION}/{remote}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if filename == header and hashlib.sha256(content).hexdigest() != JSON_SHA256:
                raise SystemExit("Downloaded JSON header checksum mismatch")
            filename.write_bytes(content)
    if hashlib.sha256(header.read_bytes()).hexdigest() != JSON_SHA256:
        raise SystemExit("JSON header checksum mismatch")
    if not args.backend_only:
        run("cmake", "-S", str(source), "-B", str(ROOT / "build/vnrc"),
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -UNDEBUG",
            f'-DCMAKE_CXX_FLAGS=-I"{deps}"')
        run("cmake", "--build", str(ROOT / "build/vnrc"), f"-j{args.jobs}")
    run("cmake", "-S", str(ROOT / "evaluator/native"), "-B", str(ROOT / "build/evaluator"),
        f"-DVNRC_SOURCE={source}", f"-DJSON_INCLUDE={deps}")
    run("cmake", "--build", str(ROOT / "build/evaluator"), f"-j{args.jobs}")


if __name__ == "__main__":
    main()
