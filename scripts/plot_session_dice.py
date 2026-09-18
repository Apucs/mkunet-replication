"""Bar chart of test DICE per SugarBeets recording session (growth stage).

Reads the table produced by eval_by_session.py and renders a report figure:
date-ordered bars in one hue, CKA_weeds set apart in gray, and a dashed
reference line at the in-domain ClinicDB score for comparison.

Usage:
    python scripts/plot_session_dice.py eval_by_session_run4.txt figures/session_dice.png
"""
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLINICDB_REF = 0.9353   # in-domain 5-run mean, ClinicDB
BAR_BLUE = '#3D6FB6'
BAR_GRAY = '#9AA3AD'
INK = '#333940'
MUTED = '#6B7280'


def parse(path):
    rows = []
    for line in open(path):
        m = re.match(r'(CKA_\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)', line.strip())
        if m:
            rows.append((m.group(1), int(m.group(2)), float(m.group(3))))
    dated = [r for r in rows if r[0] != 'CKA_weeds']
    weeds = [r for r in rows if r[0] == 'CKA_weeds']
    dated.sort(key=lambda r: r[0])
    return dated + weeds


def label_of(session):
    m = re.match(r'CKA_(\d{6})', session)
    if m:
        return datetime.strptime(m.group(1), '%y%m%d').strftime('%b %d')
    return 'weeds\n(Jun)'


def main():
    table_path, out_path = sys.argv[1], sys.argv[2]
    rows = parse(table_path)
    labels = [label_of(s) for s, _, _ in rows]
    dice = [d for _, _, d in rows]
    ns = [n for _, n, _ in rows]
    colors = [BAR_GRAY if s == 'CKA_weeds' else BAR_BLUE for s, _, _ in rows]

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    x = range(len(rows))
    ax.bar(x, dice, color=colors, width=0.72, zorder=3)

    ax.axhline(CLINICDB_REF, color=INK, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.text(-0.45, CLINICDB_REF + 0.012, f'ClinicDB (in-domain) {CLINICDB_REF:.2f}',
            ha='left', va='bottom', fontsize=8.5, color=INK)

    # selective direct labels: first, last dated session, weeds
    for i in (0, len(rows) - 2, len(rows) - 1):
        ax.text(i, dice[i] + 0.012, f'{dice[i]:.2f}', ha='center', va='bottom',
                fontsize=8, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right', color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Test DICE', fontsize=9.5, color=INK)
    ax.set_title('MK-UNet on SugarBeets2016: accuracy tracks crop growth stage',
                 fontsize=11, color=INK, pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis='y', color='#E3E6EA', lw=0.7, zorder=0)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#C9CED4')

    # sample sizes inside the bars, at the baseline
    for i, n in enumerate(ns):
        ax.text(i, 0.015, str(n), ha='center', va='bottom', fontsize=6.5,
                color='white', zorder=4)
    ax.text(-0.45, 0.015, 'n =', ha='right', va='bottom', fontsize=6.5, color=MUTED)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
