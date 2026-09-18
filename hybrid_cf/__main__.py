"""CLI: python -m hybrid_cf --fold 1 [--k 2 5 10] [--plot]"""
import argparse

from .evaluate import mae_by_k

DEFAULT_K = [2, 5, 10, 15, 20, 25, 30, 35, 40]


def plot(results):
    import matplotlib.pyplot as plt

    ks = list(results)
    plt.bar([str(k) for k in ks], list(results.values()), color="green")
    plt.ylim(0.73, 0.89)
    plt.xlabel("Number of Neighbours")
    plt.ylabel("MAE")
    plt.title("The Result of MAE by adjusting the value of k-neighbors")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate item-based hybrid similarity CF on MovieLens 100k.")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K, help="neighbour counts")
    parser.add_argument("--plot", action="store_true", help="show MAE bar chart")
    args = parser.parse_args()

    results = mae_by_k(args.fold, args.k)
    for k, mae in results.items():
        print(f"k={k:<3} MAE={mae:.6f}")
    if args.plot:
        plot(results)


if __name__ == "__main__":
    main()
