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


def package(adapter: Path, output: Path, data_dir: Path | None = None, evaluation_inputs=False) -> Path:
    data_dir = data_dir or ROOT / "dataset_final_v2"
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
    evaluation_files = [ROOT/'dataset_multisource/fraudlm_sft_train.json',
                        ROOT/'checkpoint/xl_r16_v1/lora_adapter.pt', ROOT/'checkpoint/xl_r16_v1/run_config.json'] if evaluation_inputs else []
    for path in evaluation_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Preserving existing private archive: {output}; choose another --output')
    manifest = {"files": {}}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in required[:5]:
            arcname = Path(data_dir.name) / path.name
            archive.write(path, arcname)
            manifest["files"][str(arcname)] = sha256(path)
        for path in (adapter, run_config):
            arcname = Path('checkpoint') / adapter.parent.name / path.name
            archive.write(path, arcname)
            manifest["files"][str(arcname)] = sha256(path)
        for path in evaluation_files:
            arcname = path.relative_to(ROOT)
            archive.write(path, arcname)
            manifest['files'][str(arcname)] = sha256(path)
        archive.writestr("PRIVATE_INPUTS_MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"Created {output} ({output.stat().st_size / 1024**2:.1f} MB)")
    print("Copy this ZIP to the PC and extract it in the cloned repository root.")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path,
                        help="path to the adapter used to initialise the next run")
    parser.add_argument('--data-dir', type=Path, default=ROOT/'dataset_final_v2')
    parser.add_argument('--evaluation-inputs', action='store_true')
    parser.add_argument("--output", type=Path,
                        default=ROOT / "checkpoint" / "fraudlm-final-models-private-inputs.zip")
    args = parser.parse_args(argv)
    package(args.adapter.resolve(), args.output.resolve(), args.data_dir.resolve(), args.evaluation_inputs)


if __name__ == "__main__":
    main()
