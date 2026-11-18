import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [r for r in csv.DictReader(open("results.csv")) if r["test"]=="prefix" and r["metric"]=="ttft_p50_s"]
d = {r["tag"]: float(r["value"])*1000 for r in rows}
order = [t for t in ["prefix-off","prefix-on"] if t in d]
fig, ax = plt.subplots(figsize=(5,4.5))
bars = ax.bar(order, [d[t] for t in order], color=["#d62728","#2ca02c"])
ax.set_ylabel("TTFT p50 (ms) — menor es mejor"); ax.set_title("Prefix caching: time-to-first-token")
for b,t in zip(bars,order): ax.text(b.get_x()+b.get_width()/2, d[t], f"{d[t]:.0f} ms", ha="center", va="bottom")
fig.tight_layout(); fig.savefig("prefix_ttft.png", dpi=130); print("-> prefix_ttft.png")