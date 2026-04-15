"""Evaluate classical and DL agents on the same N seeds, write a CSV. Stub."""
# import argparse, csv, time
# from classical.run import run as run_classical
# from dl.infer import run as run_dl
#
# def main(n_seeds: int):
#     rows = []
#     for seed in range(n_seeds):
#         for name, fn in [("classical", run_classical), ("dl", run_dl)]:
#             t0 = time.time()
#             ret = fn(seed, render=False)
#             rows.append({"seed": seed, "agent": name, "return": ret, "wall_s": time.time() - t0})
#     with open("eval/results.csv", "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
#
# if __name__ == "__main__":
#     p = argparse.ArgumentParser(); p.add_argument("--n-seeds", type=int, default=50)
#     main(p.parse_args().n_seeds)
