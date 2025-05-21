import json
import spacy
import os
import uuid
import pandas as pd

nlp = spacy.load("en_core_web_sm")


# convert the generated DA output files from UnleaseLLMRE to the data format of the original TACRED, TACREV or Re-TACRED
def convert_generated_to_tacred(input_path, output_path):
    print("generating new data examples:")
    converted = []
    with open(input_path) as fin:
        for i, line in enumerate(fin):
            data = json.loads(line)
            text = data["text"]
            tokens = [token.text for token in nlp(text)]

            try:
                subj_tokens = [token.text for token in nlp(data["subj"])]
                obj_tokens = [token.text for token in nlp(data["obj"])]

                subj_start = next(i for i in range(len(tokens)) if tokens[i:i + len(subj_tokens)] == subj_tokens)
                subj_end = subj_start + len(subj_tokens) - 1

                obj_start = next(i for i in range(len(tokens)) if tokens[i:i + len(obj_tokens)] == obj_tokens)
                obj_end = obj_start + len(obj_tokens) - 1

                converted.append({
                    "id": uuid.uuid4().hex[:20],
                    "relation": data["relation"],
                    "token": tokens,
                    "subj_start": subj_start,
                    "subj_end": subj_end,
                    "subj_type": data["subj_type"],
                    "obj_start": obj_start,
                    "obj_end": obj_end,
                    "obj_type": data["obj_type"]
                })

            except StopIteration:
                print(f"Skipping line {i}: couldn't locate subj/obj in tokens.")

    new_converted = [ex for ex in converted]
    with open(output_path, 'w') as fout:
        json.dump(new_converted, fout, indent=2)
    print(f"Saved {len(new_converted)} new converted examples to {output_path}")


def merge_datasets(orig, gen, merged_file, orig_limit=None, gen_limit=None):
    print("merging datasets:")
    seen_ids = set()
    merged = []

    for path in [orig, gen]:
        with open(path) as f:
            try:
                data_list = json.load(f)  # Expecting a JSON array
                if path == orig and orig_limit is not None:
                    data_list = data_list[:orig_limit]
                if path == gen and gen_limit is not None:
                    data_list = data_list[:gen_limit]
                print(f"{len(data_list)} examples loaded from {path}")

                for data in data_list:
                    if data['id'] not in seen_ids:
                        merged.append(data)
                        seen_ids.add(data['id'])
            except Exception as e:
                print(f"Error reading {path}: {e}")

    with open(merged_file, 'w') as fout:
        json.dump(merged, fout, indent=2)
    print(f"Merged {len(merged)} examples into {merged_file}")


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


def count_relation_types(path):
    with open(path) as f:
        data = json.load(f)

    unique_relations = set(item['relation'] for item in data)
    print(len(unique_relations))
    print(unique_relations)


# Function to convert TACRED-format data to ICL prompts
def convert_tacred_to_icl_prompts(json_path, output_path, format_type="text"):
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


if __name__ == '__main__':
    generated_path = "./generated/generated_deepseek-coder-1.3b-instruct_tacred.json"
    tac_gen_path = "./data/train_new.json"  # convert to TACRED, TACREV, or Re-TACRED dataset format
    tac_orig_path = "./data/train.json"
    merged_path = "./data/train_merged.json"
    relations_out_path = "./data/relation.csv"
    icl_prompts_path = "./data/train_prompts.json"

    # convert_generated_to_tacred(generated_path, tac_gen_path)  # generate new training data
    # merge_datasets(tac_orig_path, tac_gen_path, merged_path, orig_limit=4500, gen_limit=1500)

    # count_relation_types(tac_gen_path)
    # extract_relations_to_csv(tac_gen_path, relations_out_path)

    # convert_tacred_to_icl_prompts(tac_gen_path, icl_prompts_path, format_type="text")
