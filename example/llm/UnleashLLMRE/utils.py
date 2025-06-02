import json
import difflib
import matplotlib.pyplot as plt
import os
import spacy
import uuid
import pandas as pd
import random
import math
import re

nlp = spacy.load("en_core_web_sm")


# convert the LLM generated DA outputs to the data format of the original TACRED, TACREV or Re-TACRED
def convert_generated_to_tac(input_path, model):
    print("generating new data examples:")
    doc = os.path.basename(input_path).replace('.json', '')
    converted = []
    with open(input_path) as fin:
        for i, line in enumerate(fin):
            data = json.loads(line)
            text = data["text"]
            tokens = [token.text for token in nlp(text)]

            try:
                subj_tokens = [token.text for token in nlp(data["subj"].strip())]
                obj_tokens = [token.text for token in nlp(data["obj"].strip())]

                subj_start = next(i for i in range(len(tokens)) if tokens[i:i + len(subj_tokens)] == subj_tokens)
                subj_end = subj_start + len(subj_tokens) - 1

                obj_start = next(i for i in range(len(tokens)) if tokens[i:i + len(obj_tokens)] == obj_tokens)
                obj_end = obj_start + len(obj_tokens) - 1

                converted.append({
                    "id": uuid.uuid4().hex[:20],
                    "modelid": model,
                    "relation": data["relation"],
                    "token": tokens,
                    "subj_start": subj_start,
                    "subj_end": subj_end,
                    "obj_start": obj_start,
                    "obj_end": obj_end,
                    "subj_type": data["subj_type"],
                    "obj_type": data["obj_type"]
                })

            except StopIteration:
                print(f"Skipping line {i}: couldn't locate subj/obj in tokens.")
                print("DEBUG: Tokens:", tokens)
                print("DEBUG: Subj tokens:", subj_tokens)
                print("DEBUG: Obj tokens:", obj_tokens)
                print("DEBUG: Subj string:", data["subj"])
                print("DEBUG: Obj string:", data["obj"])

    new_converted = [ex for ex in converted]
    filename = os.path.basename(input_path).replace("generated", "train").replace(".json", f"_{len(converted)}.json")
    output_path = os.path.join("./tacred/skewed/", filename)
    with open(output_path, 'w') as fout:
        json.dump(new_converted, fout, indent=2)
    print(f"Saved {len(new_converted)} new converted examples to {output_path}")
    count_relation_stats(output_path)


def merge_datasets(paths, limit=None):
    print("Merging datasets from multiple files:")
    seen_ids = set()
    merged = []

    for path in paths:
        with open(path) as f:
            try:
                data_list = json.load(f)  # Expecting a JSON array
                if limit is not None:
                    data_list = data_list[:limit]
                print(f"{len(data_list)} examples loaded from {path}")

                for data in data_list:
                    if data['id'] not in seen_ids:
                        merged.append(data)
                        seen_ids.add(data['id'])
            except Exception as e:
                print(f"Error reading {path}: {e}")

    merged_file = path.replace('.json', f'_merged_{len(merged)}.json')
    with open(merged_file, 'w') as fout:
        json.dump(merged, fout, indent=2)
    print(f"Merged {len(merged)} examples into {merged_file}")
    count_relation_stats(merged_file)


def extract_relations_to_csv(json_path, csv_output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    relation_set = set()
    for item in data:
        relation_set.add((item['subj_type'], item['obj_type'], item['relation']))

    relation_list = sorted(list(relation_set))
    relation_df = pd.DataFrame(relation_list, columns=["head_type", "tail_type", "relation"])
    relation_df['index'] = range(len(relation_df))
    relation_df = relation_df[['head_type', 'tail_type', 'relation', 'index']]
    relation_df.to_csv(csv_output_path, index=False)
    print(f"Extracted {len(relation_df)} relations to {csv_output_path}")


# Function to convert TACRED-format data to ICL prompts
def convert_tacred_back_to_icl_prompts(json_path, output_path, format_type="text"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts = []
    for item in data:
        text = ' '.join(item['token'])
        head = item['token'][item['subj_start']:item['subj_end'] + 1]
        tail = item['token'][item['obj_start']:item['obj_end'] + 1]
        head_entity = ' '.join(head)
        tail_entity = ' '.join(tail)
        head_type = item['subj_type']
        tail_type = item['obj_type']
        relation = item['relation']

        if format_type == "text":
            prompt = f"There are candidate relations: [RELATION List].\n" \
                     f"Context: {text}. The relation between ({head_type}) '{head_entity}' and ({tail_type}) '{tail_entity}' in the context is {relation}."
        elif format_type == "instruct":
            prompt = f"Given a context, a pair of head and tail entities in the context, decide the relationship between the head and tail entities from candidate relations: [RELATION List].\n" \
                     f"Context: {text}. The relation between ({head_type}) '{head_entity}' and ({tail_type}) '{tail_entity}' in the context is {relation}."
        elif format_type == "text_schema":
            prompt = f"Relation: {relation}.\n" \
                     f"Context: {text}.\n" \
                     f"Head Type: {head_type}. Head Entity: {head_entity}.\n" \
                     f"Tail Type: {tail_type}. Tail Entity: {tail_entity}.\n" \
                     f"Generate more samples like above for the relation '{relation}'."
        elif format_type == "instruct_schema":
            prompt = f"Given a relation, context, and entity annotations, generate more examples for the relation '{relation}'.\n" \
                     f"Relation: {relation}.\n" \
                     f"Context: {text}.\n" \
                     f"Head Type: {head_type}. Head Entity: {head_entity}.\n" \
                     f"Tail Type: {tail_type}. Tail Entity: {tail_entity}."
        else:
            raise ValueError(
                "Unsupported format_type. Choose from 'text', 'instruct', 'text_schema', or 'instruct_schema'.")

        prompts.append(prompt)

    with open(output_path, 'w') as fout:
        json.dump(prompts, fout, indent=2)

    print(f"Saved {len(prompts)} prompts to {output_path}")


# Function to count relation stats like label frequency, percentage, types etc. in a TACRED-style JSON file
def count_relation_stats(input_file, sort_by_count=True, sample_num=None, sample_method="seq", out_file=None):
    print(f"{input_file} dataset stats:")
    with open(input_file, "r") as f:
        data = json.load(f)

    # decide sampling method and the number of examples to sample if not sampling the whole dataset
    if sample_method != "seq":
        random.shuffle(data)
    if sample_num is not None:
        data = data[:sample_num]

    # count relation examples and types
    total = len(data)
    print("total relation examples:", total)
    unique_relations = set(item['relation'] for item in data)
    num_types = len(unique_relations)
    print("total relation types (including no_relation):", num_types)

    # count frequency and percentages
    print("Relation type stats:")
    rel_counts = {}
    for example in data:
        rel = example.get("relation", "no_relation")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    rel_percentages = {rel: round((count / total) * 100, 6) for rel, count in rel_counts.items()}
    sorted_rel_counts = dict(sorted(rel_counts.items(), key=lambda x: x[1], reverse=True))
    sorted_rel_percentages = dict(sorted(rel_percentages.items(), key=lambda x: x[1], reverse=True))

    for rel in sorted_rel_counts:
        print(f"{rel}: {rel_counts[rel]}, {rel_percentages[rel]}%")

    if out_file:
        # other useful stats about the dataset
        freqs = list(rel_counts.values())
        relation_min = min(freqs)
        relation_max = max(freqs)
        relation_avg = round(sum(freqs) / len(freqs), 2)
        relation_avg_percentage = round((relation_avg / total) * 100, 6)
        relation_stddev = round((sum((x - relation_avg) ** 2 for x in freqs) / len(freqs)) ** 0.5, 2)

        # long_tail_cutoff = 50  # underrepresented relations
        total_w_rel = sum(count for rel, count in rel_counts.items() if rel != "no_relation")
        long_tail_cutoff = total_w_rel * 0.005  # 0.5% of total examples excluding no_relation
        print("Long tail cutoff point:", long_tail_cutoff)
        long_tail_relations = [rel for rel, count in rel_counts.items() if count <= long_tail_cutoff]

        total_count = sum(freqs)
        probs = [count / total_count for count in freqs]
        entropy = round(-sum(p * math.log(p, 2) for p in probs if p > 0), 6)  # entropy: lower means more skewed

        with open(out_file, "w") as f:
            json.dump({
                "num_examples": total,
                "num_relation_types": num_types,
                "relation_min": relation_min,
                "relation_max": relation_max,
                "relation_avg": relation_avg,
                "relation_avg_percentage": relation_avg_percentage,
                "relation_stddev": relation_stddev,
                "relation_distribution_entropy": entropy,
                "long_tail_relations": long_tail_relations,
                "relation_frequencies": sorted_rel_counts if sort_by_count else dict(sorted(rel_counts.items(), key=lambda x: x[0])),
                "relation_percentages": sorted_rel_percentages if sort_by_count else dict(sorted(rel_percentages.items(), key=lambda x: x[0]))
            }, f, indent=2)

    return rel_counts, rel_percentages


# compute number of examples to generate for each relation matching the relation distribution of the original training data(e.g. tacred, retacred)
def relations_gen_count(tot_gen, dataset):
    if dataset == "tacrev":
        dataset = "tacred"  # tacrev and tacred share the same training data
    with open(f"./generated/relation_stats_{dataset}.json", "r") as freq_file:
        relation_stats = json.load(freq_file)

    label_distribution = {
        r: relation_stats['relation_percentages'][r] / 100
        for r in relation_stats['relation_percentages']
    }
    print("Label distribution:\n", label_distribution)

    generation_count = {r: round(label_distribution[r] * tot_gen) for r in label_distribution}
    # Ensure at least one example per relation
    for r in generation_count:
        if generation_count[r] == 0:
            generation_count[r] = 1
    print("Generation count:\n", sum(generation_count.values()), generation_count)

    return generation_count


# plot frequencies of relaiton labels
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


# pick a subset of examples from the original dataset to form a new skewed dataset that still preserves the same relation distribution as the original
def generate_skewed_dataset(orig_paths, total_gen, exclude_no_relation=False):
    for orig_path in orig_paths:  # loop through input files if multiple
        with open(orig_path, 'r') as f:
            data = json.load(f)
        if exclude_no_relation:  # for generating a even distribution (non-skewed)
            data = [ex for ex in data if ex['relation'] != 'no_relation']

        label_to_examples = {}
        for item in data:
            rel = item['relation']
            if rel not in label_to_examples:
                label_to_examples[rel] = []
            label_to_examples[rel].append(item)

        # calculate the distribition of the original dataset
        rel_freq, _ = count_relation_stats(orig_path)  # get the distribution counts of the original
        total_orig = sum(rel_freq.values())
        rel_distribution = {rel: count / total_orig for rel, count in rel_freq.items()}
        # Calculate how many examples to sample for each relation
        gen_counts = {rel: round(rel_distribution[rel] * total_gen) for rel in rel_distribution}

        # Ensure at least one example per relation
        for rel in gen_counts:
            if gen_counts[rel] == 0:
                gen_counts[rel] = 1

        # Adjust in case rounding/adding makes total less/more than total_gen
        diff = total_gen - sum(gen_counts.values())
        if diff != 0:
            sorted_rels = sorted(gen_counts.items(), key=lambda x: -x[1])
            # starting from the most frequent relations to minimizes distortion of the original distribution
            for i in range(abs(diff)):
                rel = sorted_rels[i % len(sorted_rels)][0]
                if diff > 0:
                    gen_counts[rel] += 1
                elif gen_counts[rel] > 1:  # make sure not to subtract if relation only has a single example
                    gen_counts[rel] -= 1
        # print("Adjusted generation counts for the skewed dataset:\n", gen_counts)

        skewed_dataset = []
        for rel, count in gen_counts.items():
            examples = label_to_examples.get(rel, [])
            if len(examples) < count:
                sampled = examples  # not enough examples, take all
                print(f"Warning: only {len(examples)} examples available for {rel}, requested {count}")
            else:
                sampled = random.sample(examples, count)  # random sampling
            skewed_dataset.extend(sampled)  # add each example individually to ensure JSON format

        out_path = orig_path.replace('.json', f'_{total_gen}.json')
        with open(out_path, 'w') as f:
            json.dump(skewed_dataset, f, indent=2)
        # print(f"Saved skewed dataset of {len(skewed_dataset)} examples to {out_path}")
        count_relation_stats(out_path)


if __name__ == '__main__':
    # DA file paths
    gen_path = "./generated/tacred/train_20250601_2211.json"  # TACRED, TACREV, or Re-TACRED dataset format
    tac_path = "./tacred/train_20000.json"
    gpt4o, gpt45preview, gpt41, gpt4o0806, gpt4omini, o4mini, o3mini, gpt41mini, gpt41nano = (
        "./tacred/skewed/train_gpt4o_multi_1000.json", "./tacred/skewed/train_gpt45preview_multi_1000.json", "to be generated",
        "to be generated", "./tacred/skewed/train_gpt4omini_multi_1000.json", "./tacred/skewed/train_o4mini_multi_1000.json",
        "./tacred/skewed/train_o3mini_multi_1000.json", "./tacred/skewed/train_gpt41mini_multi_1000.json", "./tacred/skewed/train_gpt41nano_multi_1000.json")
    multi_models = [gpt4o, gpt4omini, o4mini, gpt41mini, gpt41nano, o3mini, gpt45preview]  # for multiGPTs: 1000 examples from each model

    # convert_generated_to_tac(gen_path, "gpt-4o-mini-2024-07-18")  # remenber to change model id for differet models
    # generate_skewed_dataset([tac_path], total_gen=35161)
    # merge_datasets(["./tacred/skewed/train_gpt4omini_merged_10000.json",tac_path], limit=None)
    count_relation_stats(gen_path, sort_by_count=True, sample_num=None, sample_method="seq", out_file="./generated/relation_stats.json")  # count stats of the dataset
    # relations_gen_count(15000, "tacred")
    # plot("./generated/relation_stats.json")  # plot relation frequencies

    # icl file paths
    relations_icl_path = "./data/relation_icl.csv"
    icl_prompts_path = "./data/generated_prompts.json"

    # count_relation_types(tac_gen_path)
    # extract_relations_to_csv(tac_gen_path, relations_icl_path)
    # convert_tacred_back_to_icl_prompts(tac_gen_path, icl_prompts_path, format_type="text")
