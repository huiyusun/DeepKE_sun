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


def normalize_fragmented_fields(decoded, dataset='tacred', verbose=True):
    headers = ['Relation', 'Context', 'Head Entity', 'Head Type', 'Tail Entity', 'Tail Type']
    outputs = []

    # splits BEFORE 'Relation' header, allowing for optional asterisks and whitespace
    entry_pattern = r'(?<![a-zA-Z0-9_])[*]*\s*Relation\s*[*]*[\s:.\-：\*]+'
    matches = list(re.finditer(entry_pattern, decoded, flags=re.IGNORECASE))
    starts = [m.start() for m in matches]
    starts.append(len(decoded))
    raw_entries = []
    for i in range(len(starts) - 1):
        block = decoded[starts[i]:starts[i + 1]].strip()
        if block:
            raw_entries.append(block)

    # For each entry, robustly extract each header/value pair
    def extract_field(entry, header, next_header=None):
        # Accepts asterisks and whitespace before/after header
        header_pattern = rf'[*]*\s*{header}\s*[*]*[\s:.\-：\*]*'
        if next_header:
            next_header_pattern = rf'[*]*\s*{next_header}\s*[*]*[\s:.\-：\*]*'
            pattern = rf'{header_pattern}(.*?)(?={next_header_pattern}|$)'
        else:
            pattern = rf'{header_pattern}(.*)'
        match = re.search(pattern, entry, re.IGNORECASE | re.DOTALL)
        if match:
            val = match.group(1).strip()
            # Collapse multiple trailing periods to one, for abbreviations (e.g. 'Inc..' -> 'Inc.')
            if re.search(r'\.\.+$', val):
                val = val.rstrip('.') + '.'
            else:
                val = val.rstrip('.')
            # Remove surrounding quotes if present
            val = val.strip('"').strip("'")
            return val
        return ""

    for raw_entry in raw_entries:
        # Ignore garbage if not a true entry (e.g. if split left garbage before first header)
        if not re.search(r'[*]*\s*Relation\s*[*]*[\s:.\-：\*]+', raw_entry, re.IGNORECASE):
            continue

        fields = {}
        for i, header in enumerate(headers):
            next_header = headers[i + 1] if i + 1 < len(headers) else None
            fields[header.lower()] = extract_field(raw_entry, header, next_header)

        if not all(fields.values()):
            if verbose:
                print(f"[ERROR] Skipping incomplete block: {fields}")
            continue

        rel = fields['relation'].strip()
        ctx = fields['context'].strip()
        head_ent = fields['head entity'].strip()
        head_type = fields['head type'].strip().upper()
        tail_ent = fields['tail entity'].strip()
        tail_type = fields['tail type'].strip().upper()

        if head_type in {"MISCELLANEOUS", "MISC"}:
            head_type = "MISC"
        if tail_type in {"MISCELLANEOUS", "MISC"}:
            tail_type = "MISC"

        # Entity type validation: pick allowed types for this dataset (default to tacred)
        allowed_types = entity_types.get(dataset, entity_types["tacred"])
        if head_type not in allowed_types or tail_type not in allowed_types:
            print(f"[ERROR] Invalid entity type: Head Type '{head_type}' or Tail Type '{tail_type}' not found in allowed types for dataset '{dataset}'.")
            continue

        # Entity-in-context span check (case-insensitive, stripped)
        ctx_cmp = ctx.lower()
        head_cmp = head_ent.lower()
        tail_cmp = tail_ent.lower()
        head_found = head_cmp in ctx_cmp
        tail_found = tail_cmp in ctx_cmp
        if not (head_found and tail_found):
            print(f"[ERROR] Entity span not found in context. Head: '{head_ent}' or Tail: '{tail_ent}' not found in context: {ctx}")
            continue

        head_type_out = head_type.replace(' ', '_')
        tail_type_out = tail_type.replace(' ', '_')
        rel_out = rel.lower().strip()
        text = (f"Relation: {rel_out}. Context: {ctx}. Head Entity: {head_ent}. "
                f"Head Type: {head_type_out}. Tail Entity: {tail_ent}. Tail Type: {tail_type_out}.")
        outputs.append({
            "relation": rel_out,
            "context": ctx,
            "head_entity": head_ent,
            "head_type": head_type,
            "tail_entity": tail_ent,
            "tail_type": tail_type,
            "text": text
        })
        if verbose:
            print("[DEBUG] Parsed block:")
            print(" ", fields)
            print("  → Output text:", text)
            print("")

    return outputs


# extract the relation label and entity information matching the dataset format (e.g., tacred, retacred, tacrev)
def construct_relation(rel, label, datasetname='tacred'):
    # Takes each output of normalize_fragmented_fields() as input rel
    # Compare after stripping trailing periods to avoid mismatches due to punctuation
    if rel["relation"] != label:
        return
    DAdata = {'text': rel["context"]}

    # find the entity span in the context
    def find_entity_span(entity, context):
        # Try as-is (case-insensitive)
        ent = entity.strip()
        ctx = context
        ctx_lower = ctx.lower()
        ent_var = ent
        while True:
            ent_var_strip = ent_var.strip()
            if not ent_var_strip:
                break
            ent_var_lower = ent_var_strip.lower()
            idx = ctx_lower.find(ent_var_lower)
            if idx != -1:
                matched_text = ctx[idx:idx + len(ent_var_strip)]
                # Only remove trailing period if it's not part of the original entity
                if matched_text.endswith('.') and not entity.strip().endswith('.'):
                    matched_text = matched_text[:-1]
                end_idx = idx + len(matched_text)
                return idx, end_idx, matched_text
            # Remove trailing periods and try again
            ent_var = re.sub(r"\.*$", "", ent_var_strip)
            if ent_var == ent_var_strip:
                break  # No more trailing periods to remove
        return -1, -1, None

    # head entity and type
    head = rel["head_entity"]
    headtype = rel["head_type"]
    DAdata["subj_type"] = headtype
    # tail entity and type
    tail = rel["tail_entity"]
    tailtype = rel["tail_type"]
    DAdata["obj_type"] = tailtype

    # find head and tail entity spans and values
    hpos1, hpos2, truehead = find_entity_span(head, rel["context"])
    if hpos1 == -1 or truehead is None:
        return
    tpos1, tpos2, truetail = find_entity_span(tail, rel["context"])
    if tpos1 == -1 or truetail is None:
        return

    DAdata["subj"] = truehead
    DAdata["subj_start"], DAdata["subj_end"] = hpos1, hpos2
    DAdata["obj"] = truetail
    DAdata["obj_start"], DAdata["obj_end"] = tpos1, tpos2
    DAdata["relation"] = label

    return DAdata


def normalize_and_construct(raw_text, target_label, dataset='tacred', verbose=False):
    """
    Normalize raw relation extraction text and construct relation dicts matching target_label.

    Args:
        raw_text (str): Raw input text containing relation blocks.
        target_label (str): Relation label to filter by, e.g., 'org:country_of_headquarters'.
        dataset (str): Dataset name for canonicalization, default 'tacred'.
        verbose (bool): If True, prints debug info.

    Returns:
        List[dict]: List of constructed relation dicts with keys:
            'text', 'subj_type', 'obj_type', 'subj', 'subj_start', 'subj_end',
            'obj', 'obj_start', 'obj_end', 'relation'
    """
    normalized_entries = normalize_fragmented_fields(raw_text, dataset=dataset, verbose=verbose)
    constructed_relations = []
    for entry in normalized_entries:
        if verbose:
            print(f"Processing entry with relation: {entry['relation']}")
        constructed = construct_relation(entry, target_label, datasetname=dataset)
        if constructed:
            constructed_relations.append(constructed)
        elif verbose:
            print(f"Entry relation {entry['relation']} does not match target {target_label} or failed span matching.")
    return constructed_relations


# test normalize_fragmented_fields
def test_from_terminal_file(filepath, dataset='tacred'):
    import re
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    in_block = False
    block = []

    # Strict format pattern for TACRED
    strict_pattern = (
        r"^Relation: [^\.]+\. Context: .+?\. Head Entity: .+?\. Head Type: [A-Z_ ]+\. Tail Entity: .+?\. Tail Type: [A-Z_ ]+\.$"
    )

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
                        # Only consider valid if the normalized output matches the strict TACRED format
                        valid_format = bool(re.fullmatch(strict_pattern, entry["text"].strip()))
                        results.append((entry, valid_format))
                in_block = False
                block = []
            else:
                block.append(line)

    # Write results to output file
    output_path = os.path.join(os.path.dirname(filepath), "validation_results.txt")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("--- FILE VALIDATION RESULTS (STRICT FORMAT) ---\n")
        for i, (entry, valid_format) in enumerate(results):
            out_f.write(f"\nExample {i + 1} - {'✅ STRICT FORMAT' if valid_format else '❌ INVALID FORMAT'}:\n")
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # --- Append summary statistics ---
        total = len(results)
        valid_count = sum(1 for _, valid in results if valid)
        invalid_count = total - valid_count
        out_f.write("\n--- SUMMARY ---\n")
        out_f.write(f"Total Entries: {total}\n")
        out_f.write(f"Valid Format Entries: {valid_count}\n")
        out_f.write(f"Invalid Format Entries: {invalid_count}\n")


# Test normalize_fragmented_fields against hand picked examples
def test_from_examples_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Separate model input and expected output
    input_section = re.search(r"Input examples:(.*)Expected output:", content, re.DOTALL)
    expected_section = re.search(r"Expected output:(.*)", content, re.DOTALL)

    if not input_section or not expected_section:
        print("❌ Failed to parse input or expected output sections.")
        return

    input_text = input_section.group(1).strip()
    expected_text = expected_section.group(1).strip()
    expected_lines = [
        re.sub(
            r"(Head Type:|Tail Type:)\s*(\w+)",
            lambda m: f"{m.group(1)} {m.group(2).upper()}",
            line.strip()
        )
        for line in expected_text.splitlines() if line.strip()
    ]

    normalized = normalize_fragmented_fields(input_text)
    generated_lines = [entry["text"].strip() for entry in normalized]

    print("--- TEST RESULTS ---")
    print(f"\n[DEBUG] Generated {len(generated_lines)} outputs from {len(expected_lines)} expected.")
    if len(generated_lines) == 0:
        print("[DEBUG] No outputs! Printing normalized field dictionaries for inspection:")
        for entry in normalized:
            print(entry)
    passed = True
    for i, (gen, exp) in enumerate(zip(generated_lines, expected_lines)):
        print(f"\n--- Example {i + 1} ---")
        if gen != exp:
            passed = False
            print(f"❌ Mismatch")
        else:
            print(f"✅ Match")
        print(f"Expected output : {exp}")
        print(f"Generated output: {gen}")

    if len(generated_lines) != len(expected_lines):
        passed = False
        print(f"\n❌ Number of outputs mismatch. Expected {len(expected_lines)}, got {len(generated_lines)}")

    if passed:
        print("\n✅ All normalized outputs match expected results.")
    else:
        print("\n❌ Some outputs did not match expected.")


if __name__ == '__main__':
    test_from_terminal_file("./generated/terminal_output-varying_temp,freq_pen,presence_pen.txt", dataset='tacred')
    # test_from_examples_file("./generated/validate_examples (do not edit this file).txt")

    raw_input = """
Relation: org:country_of_headquarters. Context: The International Monetary Fund is located in Washington D.C., United States, where it provides financial advice and support to countries worldwide. Head Entity: International Monetary Fund.. Head Type: ORGANIZATION. Tail Entity: United States. Tail Type: COUNTRY.  
Relation: org:country_of_headquarters. Context: The United Nations Educational, Scientific and Cultural Organization (UNESCO) has its headquarters in Paris, France, focusing on education and cultural preservation globally. Head Entity: UNESCO. Head Type: ORGANIZATION. Tail Entity: France. Tail Type: COUNTRY.  
Relation: org:country_of_headquarters. Context: Samsung Electronics operates from Suwon, South Korea, where it develops various electronic products and technologies. Head Entity: Samsung Electronics. Head Type: ORGANIZATION. Tail Entity: South Korea. Tail Type: COUNTRY.  
Relation: org:country_of_headquarters. Context: The international non-profit organization Amnesty International is headquartered in London, England, advocating for human rights around the world. Head Entity: Amnesty International. Head Type: ORGANIZATION. Tail Entity: England. Tail Type: COUNTRY.
Relation: org:country_of_headquarters. Context: The famous fashion brand Gucci is based in Florence, Italy, where it creates luxury goods that are recognized worldwide.
Head Entity: Gucci.
Head Type: ORGANIZATION.
Tail Entity: Italy.
Tail Type: COUNTRY.
Relation: org:country_of_headquarters. Context:The global banking giant HSBC Holdings plc has its headquarters located in London, United Kingdom, providing financial services around the globe.
Head Entity : HSBC Holdings plc.
Head Type : ORGANIZATION.
Tail Entity : United Kingdom.
Tail Type : COUNTRY.
 """

    target = "org:country_of_headquarters"
    results = normalize_and_construct(raw_input, target, verbose=True)

    for res in results:
        print(json.dumps(res, ensure_ascii=False))
