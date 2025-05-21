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

entity_types = {
    "tacrev": ['URL', 'LOCATION', 'IDEOLOGY', 'CRIMINAL CHARGE', 'TITLE', 'STATE OR PROVINCE', 'DATE', 'PERSON',
               'NUMBER', 'CITY', 'DURATION', 'CAUSE OF DEATH', 'COUNTRY', 'NATIONALITY', 'RELIGION', 'ORGANIZATION',
               'MISCELLANEOUS'],
    # "SciERC": ['Generic', 'Material', 'Method', 'Metric', 'OtherScientificTerm', 'Task'],
    "retacred": ['IDEOLOGY', 'ORGANIZATION', 'URL', 'PERSON', 'DURATION', 'COUNTRY', 'LOCATION', 'NATIONALITY', 'TITLE',
                 'RELIGION', 'NUMBER', 'CITY', 'CAUSE OF DEATH', 'DATE', 'STATE OR PROVINCE', 'CRIMINAL CHARGE'],
    "tacred": ['COUNTRY', 'IDEOLOGY', 'LOCATION', 'DATE', 'PERSON', 'NATIONALITY', 'RELIGION', 'CITY', 'MISCELLANEOUS',
               'CAUSE OF DEATH', 'TITLE', 'URL', 'NUMBER', 'ORGANIZATION', 'STATE OR PROVINCE', 'DURATION',
               'CRIMINAL CHARGE']
}


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


# compute number of examples to generate for each relation matching training data(e.g. tacred) relation distribution
def relations_gen_count(lab_list, tot_gen, dataset):
    if dataset == "tacrev":
        dataset = "tacred"  # tacrev and tacred has the same training data
    with open(f"./generated/relation_frequencies_{dataset}.json", "r") as freq_file:
        relation_freqs = json.load(freq_file)

    total_examples = sum(relation_freqs.values())
    label_distribution = {
        r: relation_freqs.get(r, 0) / total_examples
        for r in lab_list
    }

    generation_count = {r: int(label_distribution[r] * tot_gen) for r in lab_list}
    # Ensure at least one example per relation
    for r in generation_count:
        if generation_count[r] == 0:
            generation_count[r] = 1
            tot_gen += 1
    # print(total_gen, generation_counts)

    return generation_count, tot_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-ak', type=str, required=True)
    parser.add_argument('--demo_path', '-dp', type=str, required=True, help="The directory of demonstration data.")
    parser.add_argument('--output_dir', type=str, required=True, help="The output directory of generated data.")
    parser.add_argument('--dataset', type=str, required=True, choices=["tacred", "tacrev", "retacred"])
    parser.add_argument('--k', type=int, default=3, help="k-shot demonstrations")
    parser.add_argument('--timestamp_output', action='store_true',
                        help="If set, generate a new output file with a timestamp")
    args = parser.parse_args()

    openai.api_key = args.api_key
    model_id = "gpt-4o-2024-11-20"
    # model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    # model.eval()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

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

    '''
    One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. 
    The head entity has the relation with the tail entity and entities are pre-categorized as the following types: URL, LOCATION, IDEOLOGY, CRIMINAL CHARGE, TITLE, STATE OR PROVINCE, DATE, PERSON, NUMBER, CITY, DURATION, CAUSE OF DEATH, COUNTRY, NATIONALITY, RELIGION, ORGANIZATION, MISCELLANEOUS. 
    Here are some samples for relation 'org:founded_by':
    Relation: org:founded_by. Context: President Lee Teng-hui confers the Order of the Brilliant Star with a Violet Grand Cordon on Samuel Noordhoff , founder of the Noordhoff Craniofacial Foundation , for his devoted service to local citizens over the past four decades. Head Entity: Noordhoff Craniofacial Foundation . Head Type: ORGANIZATION. Tail Entity: Samuel Noordhoff. Tail Type: PERSON.
    Relation: org:founded_by. Context: Talansky is also the US contact for the New Jerusalem Foundation , an organization founded by Olmert while he was Jerusalem 's mayor . Head Entity: New Jerusalem Foundation. Head Type: ORGANIZATION. Tail Entity: Olmert. Tail Type: PERSON.
    Relation: org:founded_by. Context: Sharpton has said he will not endorse any candidate until hearing more about their views on civil rights and other issues at his National Action Network convention next week in New York City . Head Entity: National Action Network. Head Type: ORGANIZATION. Tail Entity: his. Tail Type: PERSON.
    Relation: org:founded_by. Context: `` We believe that we can best serve our clients by offering a single multistrategy hedge fund platform , '' wrote John Havens , who was a founder of Old Lane with Pandit and is president of the alternative investment group . Head Entity: Old Lane. Head Type: ORGANIZATION. Tail Entity: John Havens. Tail Type: PERSON.
    Generate more samples for the relation 'org:founded_by'.
    '''

    total_gen = 150  # total relation examples to be generated
    generation_counts, total_gen = relations_gen_count(label_list, total_gen, datasetname)
    relation_totals = {k: 0 for k in generation_counts}
    print("Examples to be generated for each relation label:\n", generation_counts)

    with open(output_file, 'a') as f:
        while sum(relation_totals.values()) < total_gen:
            for label in random.sample(list(label_list.keys()), len(label_list)):
                if relation_totals[label] >= generation_counts[label]:
                    continue
                prompt = "One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. The head entity has the relation with the tail entity and entities are pre-categorized as the following types: " + \
                         (', '.join(entity_types[datasetname])) + \
                         ". Here are some samples for relation '" + label + "':\n"

                v = random.sample(label_list[label], min(args.k, len(label_list[label])))  # k-shot, or sample all if labels < k
                for i in range(len(v)):
                    sample = "Relation: " + label + ". Context: " + ' '.join(
                        [convert_token(token) for token in v[i]['token']]) + ' ' + "Head Entity: " + ' '.join(
                        [convert_token(token) for token in
                         v[i]['token'][v[i]['subj_start']:v[i]['subj_end'] + 1]]) + '. ' + "Head Type: " + v[i]['subj_type'] + '. ' + "Tail Entity: " + ' '.join(
                        [convert_token(token) for token in v[i]['token'][v[i]['obj_start']:v[i]['obj_end'] + 1]]) + ". " + "Tail Type: " + v[i]['obj_type'] + ".\n"
                    prompt = prompt + sample
                # gen_prompt = "Generate more samples in the same format like above for the relation '" + k + "'."  # format 1
                gen_prompt = "Generate more samples in the same plain text format, without using bullets or markdown, for the relation '" + label + "'."  # format 2
                prompt += gen_prompt

                # model response
                # print("🧾 Input prompt:\n", prompt)
                response = openai.ChatCompletion.create(
                    model="gpt-4o-2024-11-20",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates structured relation extraction examples in the correct format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.0,
                    max_tokens=3500,
                )
                decoded = response["choices"][0]["message"]["content"].strip()
                # inputs = tokenizer(prompt, return_tensors="pt").to(device)
                # with torch.inference_mode():
                #    outputs = model.generate(**inputs, max_new_tokens=3500, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
                # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                # print("🔹 Model generated output:\n", decoded)
                res = decoded.split('\n')

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

                        # print("generated relation:", json.dumps(DAdata, indent=2, ensure_ascii=False))
                        f.writelines(json.dumps(DAdata, ensure_ascii=False))
                        f.write('\n')
                        relation_totals[label] += 1  # increment relation count
                    except Exception as e:
                        print(f"Error processing line: {line[:80]} - {e}")
                        continue

                print(f"✅ Generated {relation_totals[label]} total for relation '{label}' | Total generated: {sum(relation_totals.values())}/{total_gen}")
