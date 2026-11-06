"""Package the local multi-source dataset and original LoRA adapter for private PC transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(adapter: Path, output: Path) -> Path:
    data_dir = ROOT / "dataset_multisource"
    run_config = adapter.parent / "run_config.json"
    required = [
        data_dir / "fraudlm_sft_train.json",
        data_dir / "fraudlm_sft_val.json",
        data_dir / "fraudlm_sft_test.json",
        data_dir / "prediction_store.jsonl",
        data_dir / "build_manifest.json",
        adapter,
        run_config,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing private input files:\n- " + "\n- ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"files": {}}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in required[:5]:
            arcname = Path("dataset_multisource") / path.name
            archive.write(path, arcname)
            manifest["files"][str(arcname)] = sha256(path)
        for path in (adapter, run_config):
            arcname = Path("checkpoint/base_adapter/xl_r16_v1") / path.name
            archive.write(path, arcname)
            manifest["files"][str(arcname)] = sha256(path)
        archive.writestr("PRIVATE_INPUTS_MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"Created {output} ({output.stat().st_size / 1024**2:.1f} MB)")
    print("Copy this ZIP to the PC and extract it in the cloned repository root.")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path,
                        help="path to the original xl_r16_v1/lora_adapter.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "checkpoint" / "fraudlm-v2-5060-private-inputs.zip")
    args = parser.parse_args(argv)
    package(args.adapter.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
