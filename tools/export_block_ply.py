"""Export an atomic paged checkpoint as a standard SH1 Gaussian PLY."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gaussian_block_store import GaussianBlockStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    store = GaussianBlockStore.open(args.checkpoint)
    output = Path(args.output) if args.output else store.root / f"scene_step{store.metadata['iteration']}.ply"
    if output.exists():
        raise FileExistsError(output)
    print(store.export_ply(output))


if __name__ == '__main__':
    main()
