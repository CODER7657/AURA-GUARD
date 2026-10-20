"""Charts and the results table from results/backtest_summary.json.

    python scripts/plot_results.py

Reads numbers only, so it runs in a second on a laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "persistence": "Persistence (next = now)",
    "same_hour_yesterday": "Same hour yesterday",
    "climatology": "Climatology",
    "linear": "Ridge on lags",
    "gradient_boosting": "Gradient boosting",
}
COLORS = {
    "persistence": "#9aa0a6",
    "same_hour_yesterday": "#c7b299",
    "climatology": "#b5c7d3",
    "linear": "#6c8ebf",
    "gradient_boosting": "#1d3557",
}


def main() -> None:
    d = json.loads((ROOT / "results" / "backtest_summary.json").read_text(encoding="utf-8"))
    pooled = {(r["horizon"], r["model"]): r for r in d["pooled"]}
    horizons = sorted({h for h, _ in pooled})

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in NAMES:
        ax.plot(horizons, [pooled[(h, m)]["mae"] for h in horizons], marker="o", label=NAMES[m],
                color=COLORS[m], linewidth=2.4 if m == "gradient_boosting" else 1.4)
    ax.set_xticks(horizons, [f"{h} h" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Mean absolute error (ppb NO$_2$)")
    ax.set_title("Walk-forward backtest, 20 EPA monitors, Jul–Dec 2024", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "mae_by_horizon.png", dpi=150)
    plt.close(fig)

    lines = [
        "| Horizon | Model | MAE (ppb) | RMSE (ppb) | R² | vs persistence | 90% interval coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in horizons:
        for m in NAMES:
            r = pooled[(h, m)]
            lines.append(
                f"| {h} h | {NAMES[m]} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.3f} | "
                f"{r['skill_vs_persistence']:+.0%} | {r['coverage']:.1%} |"
            )
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
