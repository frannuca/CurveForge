#!/usr/bin/env python3
"""
Read a CSV with columns sample,short,long,diff and plot sample, short, long and diff.
Usage: python3 plot_cma.py <csv-path> <output-png>
If paths are omitted the script will look for 'signal_cma.csv' in the current dir and
output '../../docs/images/signal_cma.png'.
"""
import sys
from pathlib import Path

import csv

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print("ERROR: matplotlib not available:", e)
    sys.exit(2)


def read_csv(path):
    samples = []
    short = []
    longv = []
    diff = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(float(row['sample']))
            short.append(float(row['short']))
            longv.append(float(row['long']))
            diff.append(float(row['diff']))
    return samples, short, longv, diff


def plot(samples, short, longv, diff, outpath):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(samples, short, label='short (EMA)')
    ax1.plot(samples, longv, label='long (EMA)')
    ax1.set_ylabel('value')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(samples, diff, label='short - long')
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_ylabel('diff')
    ax2.set_xlabel('step')
    ax2.grid(True)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    print(f"WROTE {outpath}")


def main(argv):
    csv_path = Path(argv[1]) if len(argv) > 1 else Path('signal_cma.csv')
    out_path = Path(argv[2]) if len(argv) > 2 else Path('../../docs/images/signal_cma.png')

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        sys.exit(3)

    samples, short, longv, diff = read_csv(csv_path)
    plot(samples, short, longv, diff, out_path)


if __name__ == '__main__':
    main(sys.argv)
