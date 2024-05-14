import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "pic_cc_short"
#filename = "pic_cc_long"

# Read the CSV file into a DataFrame
df = pd.read_csv("out/data/" + filename + ".csv")
df = df[df['func'] != 'Naive']
df = df[df['collision chance'] <= 0.1]

plt.figure(figsize=(12, 8))
palette = {
    "Dynamic": "green",
    "CPU Sync": "blue",
    "Naive": "red",
    "Dynamic Old": "orange"
}
sns.lineplot(data=df, x="collision chance", y="time", hue="func", style="block size", markers=True, dashes=False, palette=palette)
plt.xscale('log')
plt.yscale('log')
plt.title("Collision Chance vs. Time Across Functions and Block Sizes")
plt.xlabel("Collision Chance (log scale)")
plt.ylabel("Time (ms) (log scale)")
plt.legend(title='Function & Block Size')

# Save the plot as pic_cc_short.png
plt.savefig("out/visualization/" + filename + "_no_naive_cut.png")