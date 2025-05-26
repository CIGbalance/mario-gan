from scipy.stats import qmc
from typing import List
from mario_gan_evaluator import *
import pandas as pd
import argparse
import time
import os
import numpy as np


def lhc(dim: int, n: int):
    # Between -1 and 1, 10 dimensions
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n)
    sample = qmc.scale(sample, -1, 1)
    return sample


def run_mario_gan(sample, f_list, sim):
    """
    Run the Mario GAN evaluator for a given set of parameters.
    """
    data_list = []
    x = {f"x_{i}": sample[i] for i in range(len(sample))}
    for i in range(sim):
        tmp_dat = {"i": i}
        tmp_dat.update(x)
        for f in f_list:
            start = time.time()
            tmp_dat[f"f_{f}"] = evaluate_mario_gan("mario-gan", f, 1, list(sample), 1)[
                0
            ]
            end = time.time()
            tmp_dat[f"t_{f}"] = end - start
        data_list.append(tmp_dat)
    return pd.DataFrame(data_list)


def collect_data(
    f_list: List[int], dim: int, n: int, sim: int, n_neighbours: int, noise_scale: float
):
    """
    Collect data from a list of functions.
    """
    params_name = "dim_{}_n_{}_sim_{}".format(dim, n, sim)
    output_file = f"data_{params_name}.csv"
    samples = lhc(dim, n)
    order = random.sample(range(n), n)
    for i in order:
        sample = samples[i]
        # Run the Mario GAN evaluator for each sample
        data = run_mario_gan(list(sample), f_list, sim)
        data["sample"] = i
        # Save the data to a CSV file
        data.to_csv(
            output_file, mode="a", header=not os.path.exists(output_file), index=False
        )
        # Add neighbours
        for j in range(n_neighbours):
            neighbour = np.random.normal(loc=sample, scale=noise_scale)
            data = run_mario_gan(list(neighbour), f_list, sim)
            data["sample"] = i
            # Save the data to a CSV file
            data.to_csv(
                output_file,
                mode="a",
                header=not os.path.exists(output_file),
                index=False,
            )
    return output_file


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect data from a list of functions."
    )
    parser.add_argument("--dim", type=int, default=10, help="Number of dimensions")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--sim", type=int, default=30, help="Number of simulations")
    parser.add_argument(
        "--f_list",
        type=int,
        nargs="+",
        default=[11, 17, 13, 19],
        help="List of functions to evaluate",
    )
    parser.add_argument(
        "--n_neighbours", type=int, default=10, help="Number of neighbours to add"
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.01,
        help="Scale of the noise to add to the samples to define neighbours",
    )
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
        "f_list": args.f_list,
        "n_neighbours": args.n_neighbours,
        "noise_scale": args.noise_scale,
    }
    # Collect data
    data_file = collect_data(**params)
    print(f"Data collected and saved to {data_file}")


if __name__ == "__main__":
    main()
