import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "poisson_timesteps"

# Read the CSV file into a DataFrame
df = pd.read_csv("out/data/" + filename + ".csv")
#df = df[df['func'] != 'Naive']
df = df[df['iterations'] == 10]

plt.figure(figsize=(12, 8))
palette = {
    "Dynamic": "green",
    "CPU Sync": "blue",
    "Naive": "red",
    "Dynamic Old": "orange"
}
sns.lineplot(data=df, x="block size", y="time", hue="func", markers=True, dashes=False, palette=palette)
#plt.xscale('log')
#plt.yscale('log')
plt.title("Block size vs. Time Across Functions")
plt.xlabel("Block size")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.xticks([128, 256, 512, 1024])
plt.legend(title='Function')
plt.tight_layout()

# Save the plot as pic_cc_short.png
plt.savefig("out/visualization/pic_block.png")