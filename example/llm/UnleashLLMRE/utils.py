import json
import matplotlib.pyplot as plt
import os


# Function to count relation frequencies in a TACRED-style JSON file
def count_relation_frequencies(input_file, freq_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    rel_counts = {}
    for example in data:
        rel = example.get("relation", "no_relation")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1

    with open(freq_file, "w") as f:
        json.dump(rel_counts, f, indent=2)

    return rel_counts


def count_relation_percentages(input_file, output_file=None):
    with open(input_file, "r") as f:
        data = json.load(f)
    total = len(data)
    print("total relations:", total)
    rel_counts = {}
    for example in data:
        rel = example.get("relation", "no_relation")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    rel_percentages = {rel: round((count / total) * 100, 4) for rel, count in rel_counts.items()}

    if output_file:
        with open(output_file, "w") as f:
            json.dump(rel_percentages, f, indent=2)

    return rel_percentages


def plot(freq_file):
    with open(freq_file, "r") as f:
        relation_counts = json.load(f)

    # Remove 'no_relation'
    # filtered_counts = {k: v for k, v in relation_counts.items() if k != "no_relation"}
    # All relations
    filtered_counts = {k: v for k, v in relation_counts.items()}

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
    for i, val in enumerate(counts_all):
        plt.text(i, val + 1, str(val), ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=90)
    # plt.title("Relation Frequency Distribution in TACRED (Excl. no_relation)")
    plt.title("Relation Frequency Distribution in Re-TACRED")
    plt.xlabel("Relation Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # set data path to count relations and plot
    tacred_path = "./data/train_skewed.json"
    frequency_path = "./generated/relation_frequencies.json"

    # count_relation_frequencies(tacred_path, frequency_path)
    # plot(frequency_path)

    percentages = count_relation_percentages(tacred_path)
    for rel, pct in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
        print(f"{rel}: {pct}%")
