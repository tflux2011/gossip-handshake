"""Quick script to display publication results."""
import json

d = "./results/publication"

t1 = json.load(open(f"{d}/table1_router_comparison.json"))
print("=" * 78)
print("TABLE 1: Router Comparison")
print("=" * 78)
print(f"{'Router':<28} {'Agro':>8} {'Vet':>8} {'Overall':>9} {'Route Acc':>11}")
print("-" * 78)
for k in ["keyword", "cosine"]:
    r = t1[k]
    acc = r.get("routing_accuracy_pct", "-")
    print(f"{r['label']:<28} {r['agro_pct']:>7.1f}% {r['vet_pct']:>7.1f}% "
          f"{r['overall_pct']:>8.1f}% {acc:>10}%")

t2 = json.load(open(f"{d}/table2_variance.json"))
print("\n" + "=" * 78)
print("TABLE 2: 3-Run Variance (mean +/- std)")
print("=" * 78)
fmt = f"{'Configuration':<24} {'Agronomy':>14} {'Veterinary':>14} {'Overall':>14}"
print(fmt)
print("-" * 78)
for label, s in t2.items():
    a = f"{s['agro_mean']:.1f}+/-{s['agro_std']:.1f}%"
    v = f"{s['vet_mean']:.1f}+/-{s['vet_std']:.1f}%"
    o = f"{s['overall_mean']:.1f}+/-{s['overall_std']:.1f}%"
    print(f"{label:<24} {a:>14} {v:>14} {o:>14}")

t3 = json.load(open(f"{d}/table3_density_ablation.json"))
print("\n" + "=" * 78)
print("TABLE 3: TIES Merge Density Ablation")
print("=" * 78)
print(f"{'Density':<16} {'Agronomy':>10} {'Veterinary':>12} {'Overall':>10}")
print("-" * 78)
for k in sorted(t3.keys()):
    r = t3[k]
    print(f"{r['label']:<16} {r['agro_pct']:>9.1f}% "
          f"{r['vet_pct']:>11.1f}% {r['overall_pct']:>9.1f}%")
print("=" * 78)

meta = json.load(open(f"{d}/all_results.json"))["metadata"]
print(f"\nDuration: {meta['duration_seconds']/60:.1f} minutes")
print(f"Model: {meta['base_model']}")
print(f"Torch: {meta['torch_version']}")
