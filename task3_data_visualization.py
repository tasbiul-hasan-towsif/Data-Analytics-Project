"""
CodeAlpha - Data Analytics Internship
TASK 3: Data Visualization

Builds a set of clear, story-driven charts plus a one-page dashboard using
Matplotlib and Seaborn on the 'tips' (restaurant) dataset.

Story: "How can a restaurant increase revenue and tips?"

Install:  pip install pandas numpy matplotlib seaborn
Run:      python task3_data_visualization.py    (charts saved in ./visuals)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "visuals"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
plt.rcParams["figure.dpi"] = 110

# ------------------------------------------------------------------ DATA
df = sns.load_dataset("tips")
df["tip_pct"] = df["tip"] / df["total_bill"] * 100
df["day"] = pd.Categorical(df["day"], ["Thur", "Fri", "Sat", "Sun"], ordered=True)
print(df.head(), "\n")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["total_bill"], bins=25, kde=True, color="#4C72B0", ax=ax)
ax.axvline(df["total_bill"].mean(), color="red", ls="--", label=f"Mean = ${df['total_bill'].mean():.2f}")
ax.set(title="Most Bills Fall Between $10 and $25", xlabel="Total Bill ($)", ylabel="Number of Tables")
ax.legend()
save(fig, "01_bill_distribution.png")

# ------------------------------------------------------------------ 2. Revenue by day
rev = df.groupby("day", observed=True)["total_bill"].sum().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
bars = sns.barplot(data=rev, x="day", y="total_bill", ax=ax)
for b in bars.patches:
    ax.annotate(f"${b.get_height():,.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontweight="bold")
ax.set(title="Weekends Drive the Most Revenue", xlabel="", ylabel="Total Revenue ($)")
save(fig, "02_revenue_by_day.png")

# ------------------------------------------------------------------ 3. Bill vs tip
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="total_bill", y="tip", hue="time", size="size", sizes=(30, 220), alpha=.7, ax=ax)
sns.regplot(data=df, x="total_bill", y="tip", scatter=False, color="black", line_kws={"lw": 1.5}, ax=ax)
ax.set(title="Bigger Bills -> Bigger Tips (bubble size = party size)", xlabel="Total Bill ($)", ylabel="Tip ($)")
save(fig, "03_bill_vs_tip.png")

# ------------------------------------------------------------------ 4. Tip % by group
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="day", y="tip_pct", hue="sex", ax=axes[0])
axes[0].set(title="Tip % by Day and Gender", xlabel="", ylabel="Tip (% of bill)")
sns.violinplot(data=df, x="smoker", y="tip_pct", hue="time", split=True, inner="quart", ax=axes[1])
axes[1].set(title="Tip % : Smokers vs Non-smokers, Lunch vs Dinner", xlabel="Smoker", ylabel="")
save(fig, "04_tip_pct_groups.png")

# ------------------------------------------------------------------ 5. Heatmap
pivot = df.pivot_table(index="day", columns="time", values="total_bill", aggfunc="mean", observed=True)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5, cbar_kws={"label": "Avg bill ($)"}, ax=ax)
ax.set(title="Average Bill by Day & Meal Time", xlabel="", ylabel="")
save(fig, "05_heatmap_day_time.png")

# ------------------------------------------------------------------ 6. Party size
size_stats = df.groupby("size").agg(avg_bill=("total_bill", "mean"), tables=("size", "count")).reset_index()
fig, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(data=size_stats, x="size", y="tables", color="#8fb8de", ax=ax1)
ax1.set(xlabel="Party Size", ylabel="Number of Tables", title="Tables of 2 Dominate, but Large Parties Spend More")
ax2 = ax1.twinx()
ax2.plot(range(len(size_stats)), size_stats["avg_bill"], color="crimson", marker="o", lw=2.5)
ax2.set_ylabel("Average Bill ($)", color="crimson")
ax2.grid(False)
save(fig, "06_party_size.png")

# ------------------------------------------------------------------ 7. Pairplot & correlation
g = sns.pairplot(df[["total_bill", "tip", "size", "tip_pct", "time"]], hue="time", corner=True, height=2.2)
g.figure.suptitle("Relationships Between Numeric Variables", y=1.02)
g.figure.savefig(os.path.join(OUT_DIR, "07_pairplot.png"), dpi=130, bbox_inches="tight")
plt.close(g.figure)

# ------------------------------------------------------------------ 8. DASHBOARD
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, height_ratios=[0.5, 2, 2])

# KPI row
kpis = [("Total Revenue", f"${df.total_bill.sum():,.0f}"),
        ("Avg Bill", f"${df.total_bill.mean():.2f}"),
        ("Avg Tip %", f"{df.tip_pct.mean():.1f}%"),
        ("Total Tables", f"{len(df)}")]
for i, (label, value) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.text(.5, .62, value, ha="center", fontsize=26, fontweight="bold", color="#2c3e50")
    ax.text(.5, .15, label, ha="center", fontsize=13, color="gray")

ax = fig.add_subplot(gs[1, :2])
sns.barplot(data=rev, x="day", y="total_bill", ax=ax)
ax.set(title="Revenue by Day", xlabel="", ylabel="$")

ax = fig.add_subplot(gs[1, 2:])
sns.scatterplot(data=df, x="total_bill", y="tip", hue="time", alpha=.7, ax=ax)
ax.set(title="Bill vs Tip", xlabel="Total Bill ($)", ylabel="Tip ($)")

ax = fig.add_subplot(gs[2, :2])
sns.boxplot(data=df, x="day", y="tip_pct", hue="sex", ax=ax)
ax.set(title="Tip % by Day", xlabel="", ylabel="%")

ax = fig.add_subplot(gs[2, 2:])
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar=False, ax=ax)
ax.set(title="Avg Bill: Day x Meal", xlabel="", ylabel="")

fig.suptitle("Restaurant Performance Dashboard", fontsize=20, fontweight="bold")
save(fig, "08_dashboard.png")

# ------------------------------------------------------------------ INSIGHTS
print("KEY INSIGHTS")
print(f" - Weekend (Sat+Sun) brings {rev[rev.day.isin(['Sat','Sun'])].total_bill.sum() / rev.total_bill.sum():.0%} of revenue.")
print(f" - Average tip is {df.tip_pct.mean():.1f}% of the bill; it does not rise strongly with bill size.")
print(f" - Correlation bill vs tip: {df['total_bill'].corr(df['tip']):.2f}")
print(f" - Dinner bills average ${df[df.time=='Dinner'].total_bill.mean():.2f} vs lunch ${df[df.time=='Lunch'].total_bill.mean():.2f}.")
print(f"\nAll charts saved in ./{OUT_DIR}/")
