import pandas as pd
# Seaborn isn't strictly necessary, but I think it makes plotting a bit easier, especially with more complex plots
import seaborn as sns
import matplotlib.pyplot as plt

print("Hello There!")

# Get all files in 'out' folder that start with 'data' and end with '.csv'
import os
files = os.listdir('out')
files = [f for f in files if f.startswith('data') and f.endswith('.csv')]

# Print the files
print(files)

# Read all files into a single dataframe
df = pd.concat([pd.read_csv('out/' + f) for f in files])

# Remove rows with NaN values
df = df.dropna()

# Print the first 5 rows of the dataframe
print(df.head())

# Plot the data as a lineplot grouped by 'func'
# sns.lineplot(data=df, x='block size', y='time', hue='func', errorbar=("pi", 95))
sns.lineplot(data=df, x='block size', y='time', hue='func')
plt.legend(title='Function')

# Define the size of the plot in inches
plt.gcf().set_size_inches(8, 5)

# Use log scale for y-axis
# plt.xscale("log")
plt.yscale("log")

# Set the title of the plot
plt.title('Figure title')
# Set the x-axis label
plt.xlabel('n')
# Set the y-axis label
plt.ylabel('time (ms)')
# Have the legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Use major and minor grid lines
plt.grid(which="major", linestyle='-', alpha=0.5)
plt.grid(which="minor", linestyle='-', alpha=0.2)
plt.minorticks_on()

# Can also show file instead of saving it
# plt.show()

# Save the plot to a file
plt.savefig('out/plot.png', bbox_inches='tight')