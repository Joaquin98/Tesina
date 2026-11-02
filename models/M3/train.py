"""Entrenar M3 con las particiones actuales y caché local reutilizable."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from models.common.legacy_training import training_main
    return training_main('M3', argv)


if __name__ == '__main__':
    main()
