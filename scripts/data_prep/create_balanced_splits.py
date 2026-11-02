"""Split unique clips first, then balance train and validation independently.

Historical CSV files are never overwritten. Train future models with explicit
validation_data, using models/M0/runtime.py train --dataset OUT_DIR.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from tools.dataset import DEFAULT, build_partitions, local_path, read_csv, write_partition_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT/'manifest.csv')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT/'particiones')
    parser.add_argument('--target', type=int, default=600, help='Total examples per class across train and validation')
    parser.add_argument('--validation-fraction', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = local_path(args.out_dir)
    manifest = local_path(args.manifest)
    rows = read_csv(manifest)
    parts = build_partitions(rows, target=args.target, validation_fraction=args.validation_fraction, seed=args.seed)
    if not args.dry_run:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Refusing to overwrite an existing partition version')
        output.mkdir(parents=True, exist_ok=True)
        write_partition_files(output, rows, parts)
        (output/'estado.json').write_text(json.dumps(dict(seed=args.seed,
            validation_fraction=args.validation_fraction, target_per_class=args.target), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(rows={name:len(values) for name,values in parts.items()},
                          shared_clips=0, shared_content_hashes=0, dry_run=args.dry_run)))


if __name__ == '__main__':
    main()
