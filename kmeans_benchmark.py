import sys
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kmeans_ker import kmeans_seq, kmeans_par


def test_algos(ds_size, ds_features, ds_clust, thread_list):
    # gen data
    data, _ = make_blobs(n_samples=ds_size, n_features=ds_features, centers=ds_clust, random_state=42)

    # run seq
    start = time.time()
    kmeans_seq(data, ds_clust)
    seq_time = time.time() - start
    print(f"seq time - {seq_time:.2f} seconds")

    # run par for diff threads
    par_times = []
    speedups = []
    for th in thread_list:
        start = time.time()
        kmeans_par(data, ds_clust, n_jobs=th)
        end = time.time()

        par_time = end - start
        par_times.append(par_time)
        speedups.append(seq_time / par_time)
        print(f"{th} jobs - {par_time:.2f} seconds, speedup: {seq_time / par_time:.2f}")

    # make graphs
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].plot(thread_list, speedups, marker='o')
    axes[0].set_title('K-means algo Speedup')
    axes[0].set_xlabel('Threads')
    axes[0].set_xticks(thread_list)
    axes[0].set_ylabel('Speedup')
    axes[0].grid(True)

    axes[1].plot(thread_list, par_times, marker='o', label='Parallel time')
    axes[1].axhline(y=seq_time, color='red', linestyle='--', label='Sequentive time')
    axes[1].set_title('K-means algo time measurments')
    axes[1].set_xlabel('Threads')
    axes[1].set_xticks(thread_list)
    axes[1].set_ylabel('Time')
    axes[1].grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs_{ds_size}_{ds_features}_{ds_clust}.png")
    


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Input params: dataset size, number of features, number of clusters')
        exit()

    threads = [1,2,4,6,8,12]
    size = int(sys.argv[1])
    features = int(sys.argv[2])
    clusters = int(sys.argv[3])
    print(f'Benchmark with size {size}, {features} features and {clusters} clusters')
    test_algos(size, features, clusters, threads)
