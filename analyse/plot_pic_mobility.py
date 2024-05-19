import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "mobility_timesteps"

# Read the CSV file into a DataFrame
df = pd.read_csv("out/data/" + filename + ".csv")
filter_criteria = ((df["func"] == "Naive") & (df["block size"] == 256)) | \
                  ((df["func"] == "Dynamic") & (df["block size"] == 1024)) | \
                  ((df["func"] == "CPU Sync") & (df["block size"] == 1024)) | \
                  ((df["func"] == "Dynamic Old") & (df["block size"] == 1024))
df = df[filter_criteria]
df = df[df['func'] == "Dynamic"]
df = df[df['mobility steps'] <= 100]

plt.figure(figsize=(12, 8))
palette = {
    "Dynamic": "green",
    "CPU Sync": "blue",
    "Naive": "red",
    "Dynamic Old": "orange"
}
sns.lineplot(data=df, x="mobility steps", y="time", hue="func", markers=True, dashes=False, palette=palette)
#plt.xscale('log')
#plt.yscale('log')
plt.title("Mobility Timesteps vs. Time Across Functions")
plt.xlabel("Mobility Timesteps")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.legend(title='Function')

# Save the plot as pic_cc_short.png
plt.savefig("out/visualization/" + filename + "_test.png")