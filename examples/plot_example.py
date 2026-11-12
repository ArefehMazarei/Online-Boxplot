"""Plot the online whiskers against the expanding-batch whiskers (needs matplotlib)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from onlineboxplot import OnlineBoxplot, fences as fn

np.random.seed(0)
x = np.random.gamma(0.3, 10, 10000)
WARMUP, AB = 100, dict(a=2.14, b=2.24)

det = OnlineBoxplot("qsm", n_bins="std", k=55, floor_bins=0.0, **AB).fit(x[:WARMUP])
on_lo, on_up, flagged = [det.lower], [det.upper], []
for t in range(WARMUP, len(x)):
    if det.update(x[t]):
        flagged.append(t)
    on_lo.append(det.lower); on_up.append(det.upper)
t_axis = np.arange(WARMUP, len(x) + 1)

ba = np.array([fn.qsm(*np.percentile(x[:t], [25, 50, 75]), w=3, **AB) for t in t_axis])

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(x, color="0.7", lw=0.5, label="data")
ax.plot(t_axis, on_up, "k:", lw=1.6, label="online whiskers")
ax.plot(t_axis, on_lo, "k:", lw=1.6)
ax.plot(t_axis, ba[:, 1], "r--", lw=1.0, label="expanding-batch whiskers")
ax.plot(t_axis, ba[:, 0], "r--", lw=1.0)
ax.scatter(flagged, x[flagged], s=14, c="k", zorder=3, label=f"online outliers ({len(flagged)})")
ax.set_xlabel("time"); ax.set_ylabel("value")
ax.set_title("Right-skewed data, Gamma(0.3, scale 10): online vs expanding-batch whiskers")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.26), ncol=2, fontsize=9)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "docs/example.png"
fig.savefig(out, dpi=130)
print("saved", out)
