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
from processDA import convert_token, entity_types
import re
import uuid
import spacy
from difflib import SequenceMatcher

nlp = spacy.load("en_core_web_sm")


def best_fuzzy_span(tokens, target_tokens, threshold=0.8):
    """
    Returns (start, end, score) of the best fuzzy-matching span in tokens to target_tokens.
    Only returns if the average similarity >= threshold, else returns None.
    """
    best_score = 0
    best_span = None
    n = len(target_tokens)
    for i in range(len(tokens) - n + 1):
        window = tokens[i:i + n]
        scores = []
        for t1, t2 in zip(window, target_tokens):
            t1_lower, t2_lower = t1.lower(), t2.lower()  # case insensitive matching
            if t1_lower.isdigit() and t2_lower.startswith(t1_lower):  # matches digits, e.g. 82 matches 82nd
                scores.append(1.0)
            elif t2_lower.isdigit() and t1_lower.startswith(t2_lower):
                scores.append(1.0)
            else:
                scores.append(SequenceMatcher(None, t1_lower, t2_lower).ratio())
        avg_score = sum(scores) / n
        if avg_score > best_score:
            best_score = avg_score
            best_span = (i, i + n - 1)
    if best_score >= threshold:
        return *best_span, best_score
    return None


def convert_generated_to_tac(data, model_id):
    """
    Convert a single generated data dictionary to TACRED format as a JSON string.
    Returns: json_string if conversion succeeded, else None
    """
    tokens = [token.text for token in nlp(data["text"])]

    try:
        subj_tokens = [token.text for token in nlp(data["subj"].strip())]
        obj_tokens = [token.text for token in nlp(data["obj"].strip())]

        # Find subject span: try exact match, else fuzzy
        try:
            subj_start = next(i for i in range(len(tokens)) if tokens[i:i + len(subj_tokens)] == subj_tokens)
            subj_end = subj_start + len(subj_tokens) - 1
        except StopIteration:
            fuzzy = best_fuzzy_span(tokens, subj_tokens, threshold=0.8)
            if fuzzy:
                subj_start, subj_end, fuzzy_score = fuzzy
                print(f"[FUZZY] Matched subj '{data['subj']}' in tokens at span ({subj_start}, {subj_end}) with avg sim: {fuzzy_score:.3f}")
            else:
                print("Skipping example: couldn't locate subj in tokens (even with fuzzy).")
                print("DEBUG: Tokens:", tokens)
                print("DEBUG: Subj tokens:", subj_tokens)
                print("DEBUG: Subj string:", data['subj'])
                return None

        # Find object span: try exact match, else fuzzy
        try:
            obj_start = next(i for i in range(len(tokens)) if tokens[i:i + len(obj_tokens)] == obj_tokens)
            obj_end = obj_start + len(obj_tokens) - 1
        except StopIteration:
            fuzzy = best_fuzzy_span(tokens, obj_tokens, threshold=0.8)
            if fuzzy:
                obj_start, obj_end, fuzzy_score = fuzzy
                print(f"[FUZZY] Matched obj '{data['obj']}' in tokens at span ({obj_start}, {obj_end}) with avg sim: {fuzzy_score:.3f}")
            else:
                print("Skipping example: couldn't locate obj in tokens (even with fuzzy).")
                print("DEBUG: Tokens:", tokens)
                print("DEBUG: Obj tokens:", obj_tokens)
                print("DEBUG: Obj string:", data['obj'])
                return None

        output_line = {
            "id": uuid.uuid4().hex[:20],
            "modelid": model_id,
            "relation": data["relation"],
            "token": tokens,
            "subj_start": subj_start,
            "subj_end": subj_end,
            "obj_start": obj_start,
            "obj_end": obj_end,
            "subj_type": data["subj_type"],
            "obj_type": data["obj_type"]
        }
        return json.dumps(output_line, ensure_ascii=False)
    except StopIteration:
        print("Skipping example: couldn't locate subj/obj in tokens.")
        print("DEBUG: Tokens:", tokens)
        print("DEBUG: Subj tokens:", subj_tokens)
        print("DEBUG: Obj tokens:", obj_tokens)
        print("DEBUG: Subj string:", data['subj'])
        print("DEBUG: Obj string:", data['obj'])
        return None


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
        timestamp = datetime.now().strftime("%m%d_%H%M")
        gen_out_file = os.path.join(args.output_dir, f"generated_{timestamp}.json")
        tac_out_file = os.path.join(args.output_dir, f"train_{timestamp}.json")
    else:
        gen_out_file = os.path.join(args.output_dir, "generated.json")
        tac_out_file = os.path.join(args.output_dir, "train.json")

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
    total_est_gen = 25000  # total relation examples to be generated (estimated)
    generation_counts = relations_gen_count(total_est_gen, datasetname)  # calculate number of examples to be generated per relation type
    relation_totals = {k: 0 for k in generation_counts}
    total_gen = sum(generation_counts.values())
    print("Model id:", model_id)

    with open(gen_out_file, 'a') as g, open(tac_out_file, 'w') as t:
        t.write('[\n')  # Write opening bracket for JSON file
        first = True
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
                except openai.error.OpenAIError as e:
                    print(f"❌ OpenAI API error for relation '{label}': {e}. Skipping this pass.")
                    continue
                except Exception as e:
                    print(f"❗️Unexpected error for relation '{label}': {e}. Skipping this pass.")
                    continue

                print("🔹 Model Generated Output:\n", decoded)
                res = decoded.split('\n')

                # parse and write relation examples to output
                for line in res:
                    if relation_totals[label] >= generation_counts[label]:
                        continue
                    if len(line) == 0:
                        continue

                    try:
                        DAdata = {}
                        data1 = line.split('Relation:')[-1].strip()
                        onepoint = data1.index('.')
                        relation = data1[:onepoint]
                        if relation == label:
                            relation = label
                        else:
                            continue

                        # text
                        data2 = data1.split('Context:')[-1].strip()
                        data2lower = data2.lower()
                        if "head entity:" in data2lower:
                            textend = data2lower.index('head entity:')
                            text = data2[:textend].strip()
                            data3 = data2[textend + len('head entity:'):].strip()
                        else:
                            continue

                        DAdata['text'] = text

                        # head entity
                        data3lower = data3.lower()
                        if ". head type:" in data3lower:
                            headend = data3lower.index(". head type:")
                            head = data3[:headend]
                            data4 = data3[headend + len(". head type:"):].strip()
                        else:
                            continue

                        # head type
                        data4lower = data4.lower()
                        if ". tail entity:" in data4lower:
                            htend = data4lower.index(". tail entity:")
                            headtype = data4[:htend]
                            if headtype in entity_types[datasetname] or headtype.replace('_', ' ') in entity_types[
                                datasetname]:
                                if datasetname in ["tacrev", "tacred", "retacred"]:
                                    headtype = headtype.upper()
                                    if headtype == "MISCELLANEOUS":
                                        headtype = "MISC"
                                    else:
                                        headtype = headtype.replace(" ", "_")
                                    DAdata['subj_type'] = headtype
                                elif datasetname == "SciERC":
                                    DAdata['subj_type'] = headtype.title()
                            else:
                                continue
                            data5 = data4[htend + len(". tail entity:"):].strip()
                        else:
                            continue

                        # tail entity
                        data5lower = data5.lower()
                        if ". tail type:" in data5lower:
                            tailend = data5lower.index(". tail type:")
                            tail = data5[:tailend]
                            data6 = data5[tailend + len(". tail type:"):].strip()
                        else:
                            continue

                        # tail type
                        tailtype = data6[:-1].strip()
                        if tailtype in entity_types[datasetname] or tailtype.replace("_", " ") in entity_types[datasetname]:
                            if datasetname in ["tacrev", "tacred", "retacred"]:
                                tailtype = tailtype.upper()
                                if tailtype == "MISCELLANEOUS":
                                    tailtype = "MISC"
                                else:
                                    tailtype = tailtype.replace(" ", "_")
                                DAdata['obj_type'] = tailtype
                            elif datasetname == "SciERC":
                                DAdata['obj_type'] = tailtype.title()
                        else:
                            continue

                        textlower = text.lower()
                        headlower = head.lower()
                        if headlower in textlower:
                            hpos1 = textlower.index(headlower)
                            hpos2 = hpos1 + len(headlower)
                            truehead = text[hpos1:hpos2]
                        else:
                            continue

                        taillower = tail.lower()
                        if taillower in textlower:
                            tpos1 = textlower.index(taillower)
                            tpos2 = tpos1 + len(taillower)
                            truetail = text[tpos1:tpos2]
                        else:
                            continue

                        DAdata['subj'] = truehead
                        DAdata['subj_start'], DAdata['subj_end'] = hpos1, hpos2
                        DAdata['obj'] = truetail
                        DAdata['obj_start'], DAdata['obj_end'] = tpos1, tpos2
                        DAdata['relation'] = label

                        print("Generated relation:", json.dumps(DAdata, indent=2, ensure_ascii=False))
                        tac_line = convert_generated_to_tac(DAdata, model_id)  # convert generated data to tacred format
                        if not tac_line:
                            continue
                        g.writelines(json.dumps(DAdata, ensure_ascii=False))
                        g.write("\n")
                        if not first:  # Write comma before every item except the first
                            t.write(',\n')
                        t.write(tac_line)
                        first = False
                        relation_totals[label] += 1  # increment relation count
                    except Exception as e:
                        print(f"Error processing line: {line[:80]} - {e}")
                        continue

                print(f"✅ Generated {relation_totals[label]} total for relation '{label}' | Total generated: {sum(relation_totals.values())}/{total_gen}")
        t.write('\n]\n')  # Write closing bracket at end
