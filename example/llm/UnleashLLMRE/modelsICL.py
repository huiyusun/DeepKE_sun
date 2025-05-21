import json
import random
import time
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import copy
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import string
import re
import difflib


# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if token.lower() == '-lrb-':
        return '('
    elif token.lower() == '-rrb-':
        return ')'
    elif token.lower() == '-lsb-':
        return '['
    elif token.lower() == '-rsb-':
        return ']'
    elif token.lower() == '-lcb-':
        return '{'
    elif token.lower() == '-rcb-':
        return '}'
    return token


def f1_score(true, pred_result, rel2id):
    correct = 0
    total = len(true)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable']:
        if name in rel2id:
            neg = rel2id[name]
            break
    for i in range(total):
        golden = true[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total) if total != 0 else 0
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    return result


def normalize(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower().replace(':', ' ').replace('-', ' ')).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--api_key', '-ak', type=str, required=True)
    parser.add_argument('--train_path', '-tp', type=str, required=True,
                        help="The path of training / demonstration data.")
    parser.add_argument('--test_path', '-ttp', type=str, required=True, help="The path of test data.")
    parser.add_argument('--output_success', '-success', type=str, required=True,
                        help="The output directory of successful ICL samples.")
    parser.add_argument('--output_nores', '-nores', type=str, required=True,
                        help="The output directory of failed ICL samples.")
    parser.add_argument('--prompt', type=str, required=True,
                        choices=["text", "text_schema", "instruct", "instruct_schema"])
    parser.add_argument('--k', type=int, default=1, help="k-shot demonstrations")
    args = parser.parse_args()

    # load model
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    torch.set_float32_matmul_precision('medium')
    if torch.__version__ >= "2" and torch.cuda.is_available():
        model = torch.compile(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Train / Demostration Set
    with open(args.train_path, 'r') as f:
        train = json.load(f)
    label_list = {}
    for line in train:
        rel = line['relation']
        if rel not in label_list:
            label_list[rel] = [line]
        else:
            label_list[rel].append(line)

    # Relations and label mappings from ALL relations in the training data
    all_rels = list(label_list.keys())
    rel2id = {rel: i for i, rel in enumerate(all_rels)}

    # Label words for all relations
    rel2labelword = {}
    for rel in all_rels:
        rel2labelword[rel] = (rel.lower().replace("_", " ").replace("-", " ")
                              .replace("per", "person").replace("org", "organization").replace("stateor", "state or "))
    labelword2rel = {}
    for k, v in rel2labelword.items():
        labelword2rel[v] = k

    # For ICL prompt construction, select a random subset of relations
    rels = all_rels.copy()
    random.shuffle(rels)
    if 'no_relation' in rels:
        rels.remove('no_relation')
    rels = rels[:min(len(rels), 15)]
    rels.append('no_relation')

    # Test Set
    test_number = 20
    with open(args.test_path, 'r') as f:
        test_data = json.load(f)
        # Select test examples with at least 1/n having a relation (not "no_relation")
        num_tests = min(test_number, len(test_data))
        test_with_relation = [ex for ex in test_data if ex['relation'] != 'no_relation']
        test_without_relation = [ex for ex in test_data if ex['relation'] == 'no_relation']
        num_with_relation = max(1, num_tests // 5)
        num_without_relation = num_tests - num_with_relation
        test = random.sample(test_with_relation, min(num_with_relation, len(test_with_relation))) + \
               random.sample(test_without_relation, min(num_without_relation, len(test_without_relation)))
        random.shuffle(test)

    res = []
    true = []
    nores = []
    success = []
    token_budget = 440  # maximum allowed tokens for the full input

    with open(os.path.join(args.output_success, "success.json"), "w") as f:
        for input in tqdm(test):  # loop through all test examples
            random.shuffle(rels)
            try:
                prompt_base = "There are candidate relations: " + ', '.join(
                    labelword2rel.keys()) + ".\n" if "text" in args.prompt else \
                    "Given a context, a pair of head and tail entities in the context, decide the relationship between the head and tail entities from candidate relations: " + \
                    ', '.join(labelword2rel.keys()) + ".\n"
                prompt_kshots = ""

                for rel in rels:
                    random.shuffle(label_list[rel])
                    kshot = label_list[rel][:args.k]
                    for data in kshot:
                        ss, se = data['subj_start'], data['subj_end']
                        head = ' '.join(data['token'][ss:se + 1])
                        headtype = data['subj_type'].lower().replace('_', ' ')
                        if headtype == "misc":
                            headtype = "miscellaneous"
                        oS, oe = data['obj_start'], data['obj_end']
                        tail = ' '.join(data['token'][oS:oe + 1])
                        tailtype = data['obj_type'].lower().replace('_', ' ')
                        if tailtype == "misc":
                            tailtype = "miscellaneous"
                        sentence = ' '.join([convert_token(token) for token in data['token']])
                        relation = rel2labelword[data['relation']]
                        if "schema" in args.prompt:
                            context_line = "Context: " + sentence + " The relation between " + headtype + " '" + head + "' and " + tailtype + " '" + tail + "' in the context is " + relation + ".\n"
                        else:
                            context_line = "Context: " + sentence + " The relation between " + headtype + " '" + head + "' and " + tailtype + " '" + tail + "' in the context is " + relation + ".\n"

                        # Check token count
                        temp_prompt = prompt_base + prompt_kshots + context_line
                        token_count = len(tokenizer(temp_prompt, return_tensors="pt")["input_ids"][0])
                        if token_count >= token_budget:
                            break
                        prompt_kshots += context_line

                prompt = prompt_base + prompt_kshots

                # Add the final test example (inference query) at the end of the prompt
                tss, tse = input['subj_start'], input['subj_end']
                testhead = ' '.join(input['token'][tss:tse + 1])
                testheadtype = input['subj_type'].lower().replace('_', ' ')
                if testheadtype == "misc":
                    testheadtype = "miscellaneous"
                tos, toe = input['obj_start'], input['obj_end']
                testtail = ' '.join(input['token'][tos:toe + 1])
                testtailtype = input['obj_type'].lower().replace('_', ' ')
                if testtailtype == "misc":
                    testtailtype = "miscellaneous"
                testsen = ' '.join(input['token'])
                # Append this as the last context without providing the answer
                if "schema" in args.prompt:
                    prompt += "Context: " + testsen + " The relation between " + testheadtype + " '" + testhead + "' and " + testtailtype + " '" + testtail + "' in the context is "
                else:
                    prompt += "Context: " + testsen + " The relation between " + testheadtype + " '" + testhead + "' and " + testtailtype + " '" + testtail + "' in the context is "
                    # prompt += " The relation between '" + testhead + "' and '" + testtail + "' in the context '" + testsen + "' is "

                prompt = prompt.rstrip()
                prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=eos_id, eos_token_id=eos_id)

                generated_ids = outputs[0]
                decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                decoded = re.sub(r'\n+', '\n', decoded)

                # Print both full prompt and model output
                print("🧾 FULL PROMPT (first 500 chars):")
                print(prompt[:500])
                print("🔹 MODEL GENERATED:")
                print(decoded)

                decoded = decoded.lower()
                decoded = decoded.replace(":", " ").replace("-", " ")
                resrel = ""

                # Early fallback to no_relation on vague/ambiguous answers
                if any(phrase in decoded for phrase in
                       ["no relation", "not specified", "not clear", "none", "unknown"]):
                    resrel = "no relation"
                elif ',' in decoded or 'and' in decoded:
                    resrel = "no relation"

                decoded_norm = normalize(decoded)
                # Only run matching if not already set by early fallback
                if not resrel:
                    # First try exact match
                    for label in labelword2rel:
                        if normalize(label) in decoded_norm:
                            resrel = label
                            break
                # If not found, do fuzzy match with similarity score logging and fallback
                similarities = []
                if not resrel:
                    labels_norm = [normalize(lab) for lab in labelword2rel]
                    similarities = [(label, difflib.SequenceMatcher(None, decoded_norm, norm_label).ratio())
                                    for label, norm_label in zip(labelword2rel.keys(), labels_norm)]
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    print("🔍 Top fuzzy matches (label, score):", similarities[:3])

                    # First try matches above cutoff
                    best_match = [label for label, score in similarities if score >= 0.85]
                    if best_match:
                        resrel = best_match[0]
                    elif similarities:
                        print("⚠️ Using fallback fuzzy match below cutoff:", similarities[0])
                        resrel = similarities[0][0]
                # If no strong match, fallback to no_relation
                if not resrel:
                    # similarities is defined above if not resrel
                    # If similarities exists and top score < 0.3, fallback
                    if similarities and similarities[0][1] < 0.3:
                        resrel = "no relation"

                print("Matched label:", resrel)
                print("🔍 True label:", input['relation'], "Predicted label:", labelword2rel.get(resrel, 'UNKNOWN'))

                if resrel in labelword2rel:
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("city" in resrel) and (resrel.replace("city", "cities") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("city", "cities")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("city", "cities")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("country" in resrel) and (resrel.replace("country", "countries") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("country", "countries")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("country", "countries")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("province" in resrel) and (resrel.replace("province", "provinces") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("province", "provinces")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("province", "provinces")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("cities" in resrel) and (resrel.replace("cities", "city") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("cities", "city")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("cities", "city")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("states" in resrel) and (resrel.replace("states", "state") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("states", "state")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("states", "state")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("countries" in resrel) and (resrel.replace("countries", "country") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("countries", "country")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("countries", "country")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                else:
                    input['pr'] = resrel
                    nores.append(input)
            except Exception as e:
                print("ERROR:", e)
                if "quota" in str(e).lower():
                    break
                nores.append(input)
                time.sleep(30)

    if len(nores) != 0:
        with open(os.path.join(args.output_nores, "nores.json"), 'w') as f:
            json.dump(nores, f)
    print(f"✅ Success: {len(success)}")
    print(f"❌ Failed: {len(nores)}")
    print(f1_score(true, res, rel2id))
