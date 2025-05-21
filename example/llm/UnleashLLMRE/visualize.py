import json
import matplotlib.pyplot as plt
import os

# Change this path to wherever your file is actually located
file_path = os.path.join("generated", "relation_frequencies.json")

# Load relation frequencies
with open(file_path, "r") as f:
    relation_counts = json.load(f)

# Remove 'no_relation'
filtered_counts = {k: v for k, v in relation_counts.items() if k != "no_relation"}

# Identify low-frequency relations (≤ 50)
low_freq_threshold = 50
low_freq_counts = {k: v for k, v in filtered_counts.items() if v <= low_freq_threshold}

# Sort both sets
sorted_all = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
sorted_low = sorted(low_freq_counts.items(), key=lambda x: x[1], reverse=True)

relations_all, counts_all = zip(*sorted_all)
relations_low, counts_low = zip(*sorted_low)

# Create single plot for all relations
plt.figure(figsize=(14, 6))
plt.bar(relations_all, counts_all)
plt.xticks(rotation=90)
plt.title("Relation Frequency Distribution in TACRED (Excl. no_relation)")
plt.xlabel("Relation Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
