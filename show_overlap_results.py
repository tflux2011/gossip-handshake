import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/publication_overlap"
d = json.load(open(path + "/all_results.json"))

print("=== TABLE 1: Router Comparison ===")
for k in ["keyword", "cosine"]:
    r = d["table1_router_comparison"][k]
    print("  %-24s overall=%5.1f  routing=%s" %
          (r["label"], r["overall_pct"], r.get("routing_accuracy_pct", "--")))

print("\n=== TABLE 2: Variance ===")
for label, s in d["table2_variance"].items():
    print("  %-24s overall=%5.1f+/-%4.1f" %
          (label, s["overall_mean"], s["overall_std"]))

print("\n=== TABLE 3: Density Ablation ===")
for k in sorted(d["table3_density_ablation"].keys()):
    r = d["table3_density_ablation"][k]
    print("  %-16s aeco=%5.1f vet=%5.1f irrig=%5.1f sres=%5.1f aqua=%5.1f overall=%5.1f" % (
        r["label"], r.get("aeco_pct", 0), r.get("vet_pct", 0),
        r.get("irrig_pct", 0), r.get("sres_pct", 0), r.get("aqua_pct", 0),
        r.get("overall_pct", 0)))

print("\n=== TABLE 4: Naive vs TIES ===")
for k in ["linear", "ties"]:
    r = d["table4_naive_merge"][k]
    print("  %-24s overall=%5.1f" % (r["label"], r["overall_pct"]))
