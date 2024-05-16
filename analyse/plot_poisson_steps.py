import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "poisson_timesteps"

# Read the CSV file into a DataFrame
df = pd.read_csv("out/data/" + filename + ".csv")
filter_criteria = ((df["func"] == "Naive") & (df["block size"] == 256)) | \
                  ((df["func"] == "Dynamic") & (df["block size"] == 1024)) | \
                  ((df["func"] == "CPU Sync") & (df["block size"] == 1024)) | \
                  ((df["func"] == "Dynamic Old") & (df["block size"] == 1024))
df = df[filter_criteria]

plt.figure(figsize=(12, 8))
palette = {
    "Dynamic": "green",
    "CPU Sync": "blue",
    "Naive": "red",
    "Dynamic Old": "orange"
}
sns.lineplot(data=df, x="iterations", y="time", hue="func", markers=True, dashes=False, palette=palette)
#plt.xscale('log')
#plt.yscale('log')
plt.title("Poisson steps vs. Time Across Functions and Block Sizes")
plt.xlabel("Poisson steps")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.legend(title='Function & Block Size')

# Save the plot as pic_cc_short.png
plt.savefig("out/visualization/" + filename + ".png")