import json
import numpy as np

# Load the JSON file
with open('/home/kenchan/research/SignCL/examples/GFSLT-VLP/out/0630_GF_SignCL/sdr_values_2.json', 'r') as file:
    sdr_values = json.load(file)

# Filter out entries with a value of 0
filtered_items = [(label, value) for label, value in sdr_values.items() if value != 0]

# Sort the filtered SDR values
sorted_items = sorted(filtered_items, key=lambda item: item[1])

# Define the number of bins
num_bins = 9

# Calculate the number of elements per bin
elements_per_bin = len(sorted_items) // num_bins

# Initialize bins
binned_labels = {f'bin_{i+1}': [] for i in range(num_bins)}
bin_averages = {f'bin_{i+1}': 0 for i in range(num_bins)}

# Separate labels into bins and calculate the average value for each bin
for i in range(num_bins):
    start_index = i * elements_per_bin
    if i == num_bins - 1:  # Last bin takes the remaining elements
        end_index = len(sorted_items)
    else:
        end_index = (i + 1) * elements_per_bin
    bin_values = [value for label, value in sorted_items[start_index:end_index]]
    bin_averages[f'bin_{i+1}'] = np.mean(bin_values)
    for label, value in sorted_items[start_index:end_index]:
        binned_labels[f'bin_{i+1}'].append(label)

# Print the results
for bin_label, labels in binned_labels.items():
    print(f"{bin_label}: {sorted(labels)}")
    print(f"Average value in {bin_label}: {bin_averages[bin_label]}")
    
total_average = np.mean([value for label, value in sorted_items])
print(f"Total average value: {total_average}")