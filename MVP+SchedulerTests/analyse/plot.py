import pandas as pd
# Seaborn isn't strictly necessary, but I think it makes plotting a bit easier, especially with more complex plots
import seaborn as sns
import matplotlib.pyplot as plt

print("Hello There!")

# Get all files in 'out' folder that start with 'data' and end with '.csv'
import os
files = os.listdir('out')
files = [f for f in files if f.startswith('data_max_t_no') and f.endswith('.csv')]

# Print the files
print(files)

# Read all files into a single dataframe
df = pd.concat([pd.read_csv('out/' + f) for f in files])

# Remove rows with NaN values
df = df.dropna()

# Print the first 5 rows of the dataframe
print(df.head())

# filtered_df = df[df['func'] == 'Static']
# filtered_df = filtered_df[filtered_df['iterations'] == 10000]
# filtered_df = filtered_df[filtered_df['split chance'] == 0.01]

df['label'] = df['func'] + ' (' + df['block size'].astype(str) + ')'

# Plot the data as a lineplot grouped by 'func'
# sns.lineplot(data=df, x='block size', y='time', hue='func', errorbar=("pi", 95))

x = 'iterations'

sns.lineplot(data=df, x=x, y='time', hue='label', palette=[
# # Row 1 - Red, Blue, Green, Yellow (Darker than the previous lightest)
    "#FF9999",  # Darker Light Red
    "#9999FF",  # Darker Light Blue
    "#99FF99",  # Darker Light Green
    "#FFFF99",  # Darker Light Yellow
    
    # Row 2 - Red, Blue, Green, Yellow (Medium-Light)
    "#FF5050",  # Medium-Light Red
    "#5071F0",  # Medium-Light Blue
    "#33CC33",  # Medium-Light Green
    "#FFD700",  # Medium-Light Yellow
    
    # Row 3 - Red, Blue, Green, Yellow (Medium-Dark)
    "#CC0000",  # Medium-Dark Red
    "#0000CC",  # Medium-Dark Blue
    "#008B00",  # Medium-Dark Green
    "#FFC400",  # Medium-Dark Yellow
    
    # Row 4 - Red, Blue, Green, Yellow (Darker)
    "#8B0000",  # Darker Red
    "#00008B",  # Darker Blue
    "#006400",  # Darker Green
    "#CCAC00"   # Darker Yellow
])
plt.legend(title='Function')

# Define the size of the plot in inches
plt.gcf().set_size_inches(8, 5)

# Use log scale for y-axis
plt.xscale("log")
# plt.xticks([128, 256, 512, 1024], labels=['128', '256', '512', '1024'])
plt.yscale("log")

# Set the title of the plot
plt.title('Time as a result of #iterations')
# Set the x-axis label
plt.xlabel(x)
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