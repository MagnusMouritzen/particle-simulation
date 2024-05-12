import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#filename = "pic_cc_short"
filename = "pic_cc_long"

# Read the CSV file into a DataFrame
df = pd.read_csv("out/data/" + filename + ".csv")
df = df[df['func'] != 'Naive']

plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x="collision chance", y="time", hue="func", style="block size", markers=True, dashes=False)
plt.xscale('log')
plt.yscale('log')
plt.title("Collision Chance vs. Time Across Functions and Block Sizes")
plt.xlabel("Collision Chance (log scale)")
plt.ylabel("Time (seconds)")
plt.legend(title='Function & Block Size')

# Save the plot as pic_cc_short.png
plt.savefig("out/visualization/" + filename + "_no_naive_double.png")