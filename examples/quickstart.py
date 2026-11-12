"""Smallest useful example: find outliers in a stream-like series."""
import numpy as np

from onlineboxplot import detect

rng = np.random.default_rng(0)
series = rng.gamma(0.3, 10, 5000)       # strongly right-skewed data
series[[2000, 3500]] = [90, 120]        # two injected spikes

# Online boxplot (the paper's method). The first 200 points only build the
# first histogram; later points are tested one at a time.
outliers = detect(series, warmup=200)
print(f"{len(outliers)} outliers in {len(series) - 200} tested points")
for o in sorted(outliers, key=lambda o: -o.score)[:5]:
    print(f"index {o.index:>5}  value {o.value:7.2f}  score {o.score:6.2f}  "
          f"whiskers [{o.lower:.2f}, {o.upper:.2f}]")
