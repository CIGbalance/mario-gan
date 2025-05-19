from scipy.stats import qmc
from typing import List
from mario_gan_evaluator import *
import pandas as pd
import argparse

def lhc(dim:int, n:int):
    # Between -1 and 1, 10 dimensions
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n)
    sample = qmc.scale(sample, -1, 1)
    return sample

def collect_data(f_list: List[int],
                 dim:int, n:int,
                 sim: int):
    """
    Collect data from a list of functions.
    """
    data = []
    samples = lhc(dim, n)
    for sample in samples:
        print(sample)
        x = {f"x_{i}": sample[i] for i in range(dim)}
        for i in range(sim):
            tmp_dat = {"i": i}
            tmp_dat.update(x)
            for f in f_list:
                tmp_dat[f"f_{f}"]= evaluate_mario_gan("mario-gan", f, 1, list(sample), 1)[0]
            data.append(tmp_dat)
    return data

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Collect data from a list of functions.")
    parser.add_argument("--dim", type=int, default=10, help="Number of dimensions")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--sim", type=int, default=30, help="Number of simulations")
    parser.add_argument("--f_list", type=int, nargs='+', default=[11, 17], help="List of functions to evaluate")
    return parser.parse_args()


def main():
    """
    Main function to collect data and save it to a CSV file.
    """
    args = parse_args()
    params = {
        "dim": args.dim,
        "n": args.n,
        "sim": args.sim,
        "f_list": args.f_list
    }
    # Collect data
    data = collect_data(**params)
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Save to CSV
    params_name = "dim_{}_n_{}_sim_{}".format(params["dim"], params["n"], params["sim"])
    df.to_csv(f"data_{params_name}.csv", index=False)

if __name__ == "__main__":
    main()
