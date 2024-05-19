import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "mobility_timesteps"
x = "mobility steps"
y_label = "Mobility Steps"

# Read the CSV file into a DataFrame
df = pd.read_csv("out/data/" + filename + ".csv")
filter_criteria = ((df["func"] == "Naive") & (df["block size"] == 256))
df = df[filter_criteria]
df = df[df[x] >= 100]

plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x=x, y="particles added", markers=True, dashes=False)

#plt.xscale('log')
#plt.yscale('log')
plt.title(y_label + " vs. Split Collisions")
plt.xlabel(y_label)
plt.ylabel("Split Collisions")
plt.grid(True)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5)

# Save the plot as pic_cc_short.png
plt.savefig("out/visualization/" + filename + "_added_second.png")