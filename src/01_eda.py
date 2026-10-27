"""Step 1: dataset description + EDA figures."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

df = load_raw()
summary = {
    "rows": int(df.shape[0]), "columns": int(df.shape[1]),
    "positive_rate": float(df[TARGET].mean()),
    "missing": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
    "duplicates": int(df.duplicated().sum()),
}
save_json(summary, "eda_summary.json")
print(summary)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
df[TARGET].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4C78A8", "#E45756"])
ax[0].set_title("Class balance (0 = no CHD, 1 = CHD in 10y)"); ax[0].set_xticklabels(["0", "1"], rotation=0)
miss = df.isna().sum(); miss = miss[miss > 0].sort_values()
miss.plot.barh(ax=ax[1], color="#F58518"); ax[1].set_title("Missing values per column")
df.groupby(pd.cut(df.age, [30, 40, 50, 60, 70]), observed=True)[TARGET].mean().plot.bar(ax=ax[2], color="#54A24B")
ax[2].set_title("CHD rate by age group"); ax[2].set_xlabel("")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_overview.png"), dpi=140)

plt.figure(figsize=(7, 5.5))
c = df.corr(numeric_only=True)
plt.imshow(c, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(c)), c.columns, rotation=90, fontsize=7); plt.yticks(range(len(c)), c.columns, fontsize=7)
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_corr.png"), dpi=140)
