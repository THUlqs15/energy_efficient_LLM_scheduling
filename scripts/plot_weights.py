"""Plot w_ttft and w_tpot convergence over iterations."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "results/demo/iter_custom.log"
OUT = "results/demo/weight_convergence.png"

iters, wt, wp = [], [], []
with open(LOG) as f:
    for line in f:
        r = json.loads(line)
        iters.append(r["iter"])
        wt.append(r["w_ttft"])
        wp.append(r["w_tpot"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(iters, wt, linewidth=0.8, color="#2563eb")
ax1.set_ylabel("w_ttft", fontsize=13)
ax1.set_title("Weight Convergence over Iterations", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.annotate(f"final = {wt[-1]:.2f}", xy=(iters[-1], wt[-1]),
             xytext=(-120, 20), textcoords="offset points",
             fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

ax2.plot(iters, wp, linewidth=0.8, color="#dc2626")
ax2.set_ylabel("w_tpot", fontsize=13)
ax2.set_xlabel("Iteration", fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.annotate(f"final = {wp[-1]:.4f}", xy=(iters[-1], wp[-1]),
             xytext=(-120, 20), textcoords="offset points",
             fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
