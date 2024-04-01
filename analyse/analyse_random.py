import pandas as pd
import matplotlib.pyplot as plt

# Load the new dataset
data_3 = pd.read_csv('out/data/test.txt', header=None)

# Define a function to plot histograms with different bucket sizes
def plot_histograms(data, bucket_counts):
    fig, axs = plt.subplots(len(bucket_counts), 1, figsize=(10, 20))

    for i, bucket_count in enumerate(bucket_counts):
        axs[i].hist(data[0], bins=bucket_count, color='skyblue', edgecolor='black')
        axs[i].set_title(f'Bucket count {bucket_count}')
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    plt.savefig('out/data/rng_plot.png', bbox_inches='tight')

# Call the function with specified bucket sizes
plot_histograms(data_3, [100, 50, 20, 10])