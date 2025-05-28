from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
from tqdm import tqdm
import argparse
import os
from datetime import datetime
from huggingface_hub import login
import openai
from utils import relations_gen_count
from processDA import normalize_fragmented_fields, construct_relation, convert_token, entity_types
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-ak', type=str, required=True)
    parser.add_argument('--demo_path', '-dp', type=str, required=True, help="The directory of demonstration data.")
    parser.add_argument('--output_dir', type=str, required=True, help="The output directory of generated data.")
    parser.add_argument('--dataset', type=str, required=True, choices=["tacred", "tacrev", "retacred"])
    parser.add_argument('--k', type=int, default=3, help="k-shot demonstrations")
    parser.add_argument('--timestamp_output', action='store_true',
                        help="If set, generate a new output file with timestamp each time")
    args = parser.parse_args()

    openai.api_key = args.api_key
    input_file = args.demo_path
    datasetname = args.dataset

    if args.timestamp_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = os.path.join(args.output_dir, f"generated_{timestamp}.json")
    else:
        output_file = os.path.join(args.output_dir, "generated.json")

    data = []
    label_list = {}
    with open(input_file, 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    for line in data:
        rel = line['relation']
        if rel not in label_list:
            label_list[rel] = [line]
        else:
            label_list[rel].append(line)

    # 250k tokens family: gpt-4o-2024-11-20, gpt-4.5-preview-2025-02-27, gpt-4.1-2025-04-14, gpt-4o-2024-08-06
    # 2.5M tokens family: gpt-4o-mini-2024-07-18, o4-mini-2025-04-16, gpt-4.1-mini-2025-04-14, o3-mini-2025-01-31, gpt-4.1-nano-2025-04-14
    model_id = "gpt-4o-mini-2024-07-18"
    total_est_gen = 3000  # total relation examples to be generated (estimated)
    generation_counts = relations_gen_count(total_est_gen, datasetname)  # calculate number of examples to be generated per relation type
    relation_totals = {k: 0 for k in generation_counts}
    total_gen = sum(generation_counts.values())
    print("Model id:", model_id)

    with open(output_file, 'a') as f:
        while sum(relation_totals.values()) < total_gen:
            for label in label_list:
                # if label != "no_relation":  # select labels
                #    continue
                if relation_totals[label] >= generation_counts[label]:
                    continue
                prompt = (
                    "One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. "
                    "The head entity has the relation with the tail entity and entities are pre-categorized as the following types: "
                    f"{', '.join(entity_types[datasetname])}.\n"
                    f"Below are some samples for the relation {label}:\n"
                )

                v = random.sample(label_list[label], min(args.k, len(label_list[label])))  # k-shot, or sample all if labels < k
                for i in range(len(v)):
                    sample = (
                        f"Relation: {label}. "
                        f"Context: {' '.join([convert_token(token) for token in v[i]['token']])} "
                        f"Head Entity: {' '.join([convert_token(token) for token in v[i]['token'][v[i]['subj_start']:v[i]['subj_end'] + 1]])}. "
                        f"Head Type: {v[i]['subj_type']}. "
                        f"Tail Entity: {' '.join([convert_token(token) for token in v[i]['token'][v[i]['obj_start']:v[i]['obj_end'] + 1]])}. "
                        f"Tail Type: {v[i]['obj_type']}.\n"
                    )
                    prompt = prompt + sample
                gen_prompt = (
                    f"Now generate 10 more samples in the same format for the relation: {label}."
                )
                prompt += gen_prompt
                print("🧾 Input Prompt:\n", prompt)

                # model response
                try:
                    response = openai.ChatCompletion.create(
                        model=model_id,
                        messages=[
                            {"role": "system",
                             "content": "You are a helpful assistant that generates structured relation extraction samples. Follow the formatting instructions exactly, use only valid entity types, and do not add extra text or markdown."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.0,
                        # temperature=random.choice([0.7, 0.9, 1.0]),  # 0.3–0.5: Slight diversity, still focused, 0.7: Balanced between diversity & control, 1.0: Creative, diverse outputs
                        # top_p=0.95,
                        # frequency_penalty=0.3,  # 0.5: Moderately discourages repetition, 1.0+: Strongly discourages repeated words
                        # presence_penalty=0.6,  # 0.5: Mild push for diversity, 1.0+: Strongly encourages new tokens
                        # max_tokens=3000,
                        max_completion_tokens=2000,  # for newer models
                        timeout=30,  # 30 seconds timeout
                    )
                    decoded = response["choices"][0]["message"]["content"].strip()
                except openai.error.Timeout as e:
                    print(f"⏰ Timeout for relation '{label}'. Skipping this pass.")
                    continue  # retry this label on the next loop pass

                rels = normalize_fragmented_fields(decoded, datasetname)  # fix cases where the model doesn't follow instructions and produces output in different formats
                print("🔹 Model Generated Output:\n", decoded)
                for rel in rels:
                    if relation_totals[label] >= generation_counts[label]:
                        continue
                    DAdata = construct_relation(rel, label, datasetname)  # construct the relation example from the normalized model generated output
                    if DAdata is not None:
                        print("Generated relation:", json.dumps(DAdata, indent=2, ensure_ascii=False))
                        f.writelines(json.dumps(DAdata, ensure_ascii=False))
                        f.write('\n')
                        relation_totals[label] += 1  # increment relation count
                print(f"✅ Generated {relation_totals[label]} total for relation '{label}' | Total generated: {sum(relation_totals.values())}/{total_gen}")
