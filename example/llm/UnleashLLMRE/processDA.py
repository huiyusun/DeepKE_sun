import difflib
import re
import json
import os

entity_types = {
    "tacrev": ['URL', 'LOCATION', 'IDEOLOGY', 'CRIMINAL CHARGE', 'TITLE', 'STATE OR PROVINCE', 'DATE', 'PERSON', 'NUMBER', 'CITY', 'DURATION', 'CAUSE OF DEATH', 'COUNTRY', 'NATIONALITY',
               'RELIGION', 'ORGANIZATION', 'MISCELLANEOUS'],
    # "SciERC": ['Generic', 'Material', 'Method', 'Metric', 'OtherScientificTerm', 'Task'],
    "retacred": ['IDEOLOGY', 'ORGANIZATION', 'URL', 'PERSON', 'DURATION', 'COUNTRY', 'LOCATION', 'NATIONALITY', 'TITLE', 'RELIGION', 'NUMBER', 'CITY', 'CAUSE OF DEATH', 'DATE',
                 'STATE OR PROVINCE', 'CRIMINAL CHARGE'],
    "tacred": ['COUNTRY', 'IDEOLOGY', 'LOCATION', 'DATE', 'PERSON', 'NATIONALITY', 'RELIGION', 'CITY', 'MISCELLANEOUS', 'CAUSE OF DEATH', 'TITLE', 'URL', 'NUMBER', 'ORGANIZATION',
               'STATE OR PROVINCE', 'DURATION', 'CRIMINAL CHARGE']
}

relation_types = {
    "tacred": ['no_relation', 'per:title', 'org:top_members/employees', 'per:employee_of', 'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence', 'per:age',
               'org:city_of_headquarters', 'per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:origin', 'org:subsidiaries', 'org:parents', 'per:spouse',
               'org:stateorprovince_of_headquarters', 'per:children', 'per:other_family', 'org:members', 'per:siblings', 'per:parents', 'per:schools_attended', 'per:date_of_death',
               'org:founded_by', 'org:member_of', 'per:cause_of_death', 'org:website', 'org:political/religious_affiliation', 'per:alternate_names', 'org:founded', 'per:city_of_death',
               'org:shareholders', 'org:number_of_employees/members', 'per:charges', 'per:city_of_birth', 'per:date_of_birth', 'per:religion', 'per:stateorprovince_of_death',
               'per:stateorprovince_of_birth', 'per:country_of_birth', 'org:dissolved', 'per:country_of_death'],
    "retacred": ['no_relation', 'per:title', 'org:top_members/employees', 'per:employee_of', 'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence', 'per:age',
                 'org:city_of_headquarters', 'per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:origin', 'org:subsidiaries', 'org:parents', 'per:spouse',
                 'org:stateorprovince_of_headquarters', 'per:children', 'per:other_family', 'org:members', 'per:siblings', 'per:parents', 'per:schools_attended', 'per:date_of_death',
                 'org:founded_by', 'org:member_of', 'per:cause_of_death', 'org:website', 'org:political/religious_affiliation', 'per:alternate_names', 'org:founded', 'per:city_of_death',
                 'org:shareholders', 'org:number_of_employees/members', 'per:charges', 'per:city_of_birth', 'per:date_of_birth', 'per:religion', 'per:stateorprovince_of_death',
                 'per:stateorprovince_of_birth', 'per:country_of_birth', 'org:dissolved', 'per:country_of_death']
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


def normalize_fragmented_fields(decoded, dataset='tacred'):
    """
    Normalize and extract structured fields from model-generated relation extraction examples.
    Returns:
        List[Dict[str, str]]: Each dict contains the normalized string as 'text' and extracted fields:
            'relation', 'context', 'head_entity', 'head_type', 'tail_entity', 'tail_type'
    """

    decoded = decoded.replace("：", ":").replace("。", ".")
    decoded = re.sub(r"\n+", "\n", decoded).strip()

    # Improved preprocessing: robustly collapse field headers and values even if glued or spaced inconsistently
    lines = decoded.splitlines()
    field_labels = ["Relation", "Context", "Head Entity", "Head Type", "Tail Entity", "Tail Type"]
    collapsed_lines = []
    current_field = None
    field_buffer = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_field_header = any(re.match(rf"^{label}\b", line, re.IGNORECASE) for label in field_labels)
        if is_field_header:
            if current_field:
                collapsed_lines.append(f"{current_field} {' '.join(field_buffer).strip()}")
            # Find which label matched
            matched_label = next(label for label in field_labels if re.match(rf"^{label}\b", line, re.IGNORECASE))
            # normalize header to have colon and use only the label part (for cases like "Relation : ...")
            # Find position after label (possibly with spaces/colons)
            header_match = re.match(rf"^{matched_label}\s*:?", line, re.IGNORECASE)
            if header_match:
                start_idx = header_match.end()
            else:
                start_idx = len(matched_label)
            current_field = f"{matched_label}:"
            field_buffer = [line[start_idx:].strip()] if len(line) > start_idx else []
        else:
            field_buffer.append(line)
    if current_field:
        collapsed_lines.append(f"{current_field} {' '.join(field_buffer).strip()}")

    decoded = "\n".join(collapsed_lines)

    # Now split into blocks by Relation:
    blocks = re.split(r"\n(?=Relation\s*:)", decoded, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]

    formatted = []
    for block in blocks:
        try:
            # regex for more lenient and robust field extraction
            rel_match = re.search(r"Relation\s*:\s*(.+?)\s*Context\s*:", block, re.DOTALL | re.IGNORECASE)
            ctx_match = re.search(r"Context\s*:\s*(.+?)\s*Head\s*Entity\s*:", block, re.DOTALL | re.IGNORECASE)
            he_match = re.search(r"Head\s*Entity\s*:\s*(.+?)\s*Head\s*Type\s*:", block, re.DOTALL | re.IGNORECASE)
            ht_match = re.search(r"Head\s*Type\s*:\s*(.+?)\s*Tail\s*Entity\s*:", block, re.DOTALL | re.IGNORECASE)
            te_match = re.search(r"Tail\s*Entity\s*:\s*(.+?)\s*Tail\s*Type\s*:", block, re.DOTALL | re.IGNORECASE)
            tt_match = re.search(r"Tail\s*Type\s*:\s*([A-Z_]+)\.?", block, re.IGNORECASE)

            if not all([rel_match, ctx_match, he_match, ht_match, te_match, tt_match]):
                print("Error: Could not extract all fields:\n", block)
                continue

            relation = rel_match.group(1).replace("\n", "").replace(" ", "").strip().rstrip(".")

            # Normalize common typos in known relations (including missing colons, etc.)
            relation_fixes = {
                "per:sateorprovinces_of_residence": "per:stateorprovinces_of_residence",
                "orgtop_members/employees": "org:top_members/employees",
                "orgpolitical/religious_affiliation": "org:political/religious_affiliation"
            }
            relation = relation_fixes.get(relation, relation)

            # Improved context extraction and punctuation handling
            context = ctx_match.group(1).strip()
            context = context.replace("\n", " ").strip()
            context = re.sub(r'\s*\.*\s*$', '', context) + '.'
            head_entity = he_match.group(1).strip()
            head_type = ht_match.group(1).strip()
            tail_entity = te_match.group(1).strip()
            tail_type = tt_match.group(1).strip()

            # Remove surrounding '**' or similar markdown artifacts from fields, including markdown-style asterisks and spaces
            def clean_field(val):
                return re.sub(r"^\s*\*+\s*|\s*\*+\s*$", "", val.strip())

            relation = clean_field(relation)
            context = clean_field(context)
            head_entity = clean_field(head_entity)
            head_type = clean_field(head_type)
            tail_entity = clean_field(tail_entity)
            tail_type = clean_field(tail_type)

            # --- Fuzzy match correction for known types ---
            def fuzzy_correct(item, choices):
                match = difflib.get_close_matches(item, choices, n=1, cutoff=0.85)
                return match[0] if match else item

            relation = fuzzy_correct(relation, relation_types[dataset])
            head_type = fuzzy_correct(head_type, entity_types[dataset])
            tail_type = fuzzy_correct(tail_type, entity_types[dataset])

            # Normalize all fields for trailing newlines, whitespace, and trailing periods
            head_entity = head_entity.replace("\n", " ").strip().rstrip(".")
            head_type = head_type.replace("\n", " ").strip().rstrip(".")
            tail_entity = tail_entity.replace("\n", " ").strip().rstrip(".")
            tail_type = tail_type.replace("\n", " ").strip().rstrip(".")

            output = f"Relation: {relation}. Context: {context} Head Entity: {head_entity}. Head Type: {head_type}. Tail Entity: {tail_entity}. Tail Type: {tail_type}."
            formatted.append({
                "text": output,
                "relation": relation,
                "context": context,
                "head_entity": head_entity,
                "head_type": head_type,
                "tail_entity": tail_entity,
                "tail_type": tail_type
            })

        except Exception as e:
            print("Parsing error:", e)
            continue

    return formatted


# extract the relation label and entity information matching the dataset format (e.g., tacred, retacred, tacrev)
def construct_relation(rel, label, datasetname):
    # takes each output of normalize_fragmented_fields() as input rel
    if not rel or not isinstance(rel, dict):
        return

    if rel["relation"] != label:
        return

    # text
    DAdata = {
        'text': rel["context"]
    }

    # head entity and type
    head = rel["head_entity"]
    headtype = rel["head_type"]
    if headtype in entity_types[datasetname] or headtype.replace('_', ' ') in entity_types[datasetname]:
        headtype = headtype.upper()
        if headtype == "MISCELLANEOUS":
            headtype = "MISC"
        else:
            headtype = headtype.replace(" ", "_")
        DAdata["subj_type"] = headtype
    else:
        return

    # tail entity and type
    tail = rel["tail_entity"]
    tailtype = rel["tail_type"]
    if tailtype in entity_types[datasetname] or tailtype.replace('_', ' ') in entity_types[datasetname]:
        tailtype = tailtype.upper()
        if tailtype == "MISCELLANEOUS":
            tailtype = "MISC"
        else:
            tailtype = tailtype.replace(" ", "_")
        DAdata["obj_type"] = tailtype
    else:
        return

    # head and tail positions
    textlower = rel["context"].lower()
    headlower = head.lower()
    if headlower in textlower:
        hpos1 = textlower.index(headlower)
        hpos2 = hpos1 + len(headlower)
        truehead = rel["context"][hpos1:hpos2]
    else:
        return
    taillower = tail.lower()
    if taillower in textlower:
        tpos1 = textlower.index(taillower)
        tpos2 = tpos1 + len(taillower)
        truetail = rel["context"][tpos1:tpos2]
    else:
        return

    DAdata["subj"] = truehead
    DAdata["subj_start"], DAdata["subj_end"] = hpos1, hpos2
    DAdata["obj"] = truetail
    DAdata["obj_start"], DAdata["obj_end"] = tpos1, tpos2
    DAdata["relation"] = label

    return DAdata


# --- TACRED format validator and file block tester ---
def is_valid_tacred_format(line, dataset):
    pattern = (
        r"Relation: ([^\.]+)\. Context: .+?\."
        r".*?\bHead Entity: .+?\."
        r".*?\bHead Type: ([A-Z_]+)\."
        r".*?\bTail Entity: .+?\."
        r".*?\bTail Type: ([A-Z_]+)\."
    )
    match = re.fullmatch(pattern, line.strip(), re.IGNORECASE | re.DOTALL)
    if not match:
        return False, "Pattern match failed"

    rel, head_type, tail_type = match.groups()
    rel_list = relation_types[dataset]
    ent_list = entity_types[dataset]

    # Normalize MISC to MISCELLANEOUS
    if head_type == "MISC":
        head_type = "MISCELLANEOUS"
    if tail_type == "MISC":
        tail_type = "MISCELLANEOUS"

    if rel not in rel_list:
        return False, f"Invalid relation: {rel}"
    if head_type not in ent_list:
        return False, f"Invalid head type: {head_type}"
    if tail_type not in ent_list:
        return False, f"Invalid tail type: {tail_type}"

    return True, "OK"


def test_from_txt_file(filepath, dataset='tacred'):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    in_block = False
    block = []

    for line in lines:
        if "Model Generated Output" in line:
            in_block = True
            block = []
            continue
        if in_block:
            if "✅ Generated" in line or "Error processing line" in line or "🔹 Model Generated Output" in line:
                if block:
                    raw_text = ''.join(block).strip()
                    normalized = normalize_fragmented_fields(raw_text)
                    for i, entry in enumerate(normalized):
                        valid, reason = is_valid_tacred_format(entry["text"], dataset)
                        results.append((entry, valid, reason))
                in_block = False
                block = []
            else:
                block.append(line)

    # Write results to output file
    output_path = os.path.join(os.path.dirname(filepath), "validation_results.txt")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("--- FILE VALIDATION RESULTS ---\n")
        for i, (entry, valid, reason) in enumerate(results):
            out_f.write(f"\nExample {i + 1} - {'✅ VALID' if valid else '❌ INVALID'} ({reason}):\n")
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # --- Append summary statistics ---
        total = len(results)
        valid_count = sum(1 for _, valid, _ in results if valid)
        invalid_count = total - valid_count
        out_f.write("\n--- SUMMARY ---\n")
        out_f.write(f"Total Entries: {total}\n")
        out_f.write(f"Valid Entries: {valid_count}\n")
        out_f.write(f"Invalid Entries: {invalid_count}\n")


if __name__ == '__main__':
    test_from_txt_file("./generated/terminal_output-prompt+generated_examples.txt", dataset='tacred')
