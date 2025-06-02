import difflib
import re
import json
import os
import re

entity_types = {
    "tacrev": ['URL', 'LOCATION', 'IDEOLOGY', 'CRIMINAL CHARGE', 'TITLE', 'STATE OR PROVINCE', 'DATE', 'PERSON', 'NUMBER', 'CITY', 'DURATION', 'CAUSE OF DEATH', 'COUNTRY', 'NATIONALITY',
               'RELIGION', 'ORGANIZATION', 'MISC'],
    # "SciERC": ['Generic', 'Material', 'Method', 'Metric', 'OtherScientificTerm', 'Task'],
    "retacred": ['IDEOLOGY', 'ORGANIZATION', 'URL', 'PERSON', 'DURATION', 'COUNTRY', 'LOCATION', 'NATIONALITY', 'TITLE', 'RELIGION', 'NUMBER', 'CITY', 'CAUSE OF DEATH', 'DATE',
                 'STATE OR PROVINCE', 'CRIMINAL CHARGE'],
    "tacred": ['COUNTRY', 'IDEOLOGY', 'LOCATION', 'DATE', 'PERSON', 'NATIONALITY', 'RELIGION', 'CITY', 'MISC', 'CAUSE OF DEATH', 'TITLE', 'URL', 'NUMBER', 'ORGANIZATION',
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


def normalize_fragmented_fields(decoded, dataset='tacred', verbose=True):
    # All possible field headers (variants included for noise)
    header_map = {
        'RELATION': [r'relation[\s:]*'],
        'CONTEXT': [r'context[\s:]*'],
        'HEAD_ENTITY': [r'head\s*entity[\s:]*'],
        'HEAD_TYPE': [r'head\s*type[\s:]*'],
        'TAIL_ENTITY': [r'tail\s*entity[\s:]*'],
        'TAIL_TYPE': [r'tail\s*type[\s:]*'],
    }

    # Build a big regex to match any header, case-insensitive
    header_patterns = []
    for field, variants in header_map.items():
        for variant in variants:
            header_patterns.append(f'(?P<{field}>{variant})')
    header_regex = re.compile('|'.join(header_patterns), re.IGNORECASE)

    # Find all header positions
    matches = list(header_regex.finditer(decoded))
    if not matches:
        return []

    outputs = []
    rel_indices = [i for i, m in enumerate(matches) if m.lastgroup == 'RELATION']

    def clean_field(s, kind):
        # --- Normalize whitespace and punctuation ---
        s = s.replace('\n', ' ').replace('。', '.').replace('：', ':')
        s = ' '.join(s.split())
        # --- Remove leading junk and field headers in one go ---
        s = re.sub(r'^[\s\*\&\^,;:\.]+', '', s)
        s = re.sub(r'^(head entity|tail entity|head type|tail type|entity|head|tail|relation|context)[\s\*\&\^,;:\.]*', '', s, flags=re.IGNORECASE)
        # --- Remove trailing junk ---
        s = re.sub(r'[\s\*\&\^,;:]+$', '', s)
        # --- Remove markdown everywhere except context ---
        if kind != "context":
            s = re.sub(r'(\*\*|\*|`|_)', '', s)
        s = s.strip()
        # --- Relation label normalization ---
        if kind == "relation":
            valid_labels = relation_types.get(dataset, [])
            cleaned = s.replace(' ', '').replace('\n', '').lower().rstrip('.')
            import difflib
            best = difflib.get_close_matches(cleaned, [lbl.replace(' ', '').lower() for lbl in valid_labels], n=1, cutoff=0.80)
            if best:
                idx = [lbl.replace(' ', '').lower() for lbl in valid_labels].index(best[0])
                match = valid_labels[idx]
                return match + '.'
            idx = s.find('.')
            if idx >= 0:
                return s[:idx + 1]
            return s
        # --- Type label normalization ---
        if kind == "type":
            s = s.upper().strip()
            valid_types = ['PERSON', 'ORGANIZATION', 'TITLE', 'CAUSE OF DEATH']
            for vt in valid_types:
                if vt in s:
                    return vt
            return s
        # --- Entity field: collapse terminal periods unless abbreviation ---
        if kind in ("entity", "tail_entity", "head_entity"):
            if re.search(r'(\b[A-Z][a-zA-Z]*\.)\.$', s) or re.search(r'\bInc\.\.$', s) or re.search(r'\bCorp\.\.$', s):
                pass
            else:
                s = re.sub(r'\.{2,}$', '.', s)
        # --- Context: preserve all periods, just ensure one at end ---
        if kind == "context":
            if not s.endswith('.'):
                s += '.'
        return s

    def ensure_period(s):
        s = s.strip()
        return s if s.endswith('.') else s + '.'

    for ex_idx, rel_idx in enumerate(rel_indices):
        start = matches[rel_idx].start()
        end = matches[rel_indices[ex_idx + 1]].start() if ex_idx + 1 < len(rel_indices) else len(decoded)
        block = decoded[start:end]
        # Find headers inside this block (ordered)
        block_matches = list(header_regex.finditer(block))
        field_positions = []
        for m in block_matches:
            field_positions.append((m.lastgroup, m.start(), m.end()))
        # Extract field values between each header
        fields = {}
        for i, (fname, s, e) in enumerate(field_positions):
            next_start = field_positions[i + 1][1] if i + 1 < len(field_positions) else len(block)
            val = block[e:next_start]
            # Only truncate at header marker if not context field
            if fname not in ['CONTEXT']:
                val = re.split(r'(head entity|tail entity|head type|tail type|entity|head|tail|relation|context)[\s:]*', val, flags=re.IGNORECASE)[0]
            val = val.lstrip('\n ').replace('\n', ' ')
            # Clean both leading and trailing junk
            val = clean_field(val, fname.lower())
            fields[fname] = val
        # Ensure all fields present
        field_order = ['RELATION', 'CONTEXT', 'HEAD_ENTITY', 'HEAD_TYPE', 'TAIL_ENTITY', 'TAIL_TYPE']
        if all(f in fields and fields[f].strip() for f in field_order):
            rel = clean_field(fields['RELATION'], "relation")
            ctx = ensure_period(clean_field(fields['CONTEXT'], "context"))
            head_ent = ensure_period(clean_field(fields['HEAD_ENTITY'], "entity"))
            tail_ent = ensure_period(clean_field(fields['TAIL_ENTITY'], "entity"))
            head_type = ensure_period(clean_field(fields['HEAD_TYPE'], "type"))
            tail_type = ensure_period(clean_field(fields['TAIL_TYPE'], "type"))
            output = (f"Relation: {ensure_period(rel)} Context: {ctx} Head Entity: {head_ent} Head Type: {head_type} Tail Entity: {tail_ent} Tail Type: {tail_type}")
            outputs.append(dict(
                relation=rel,
                context=ctx,
                head_entity=head_ent,
                head_type=head_type,
                tail_entity=tail_ent,
                tail_type=tail_type,
                text=output.strip()
            ))
        else:
            print(f"\n[normalize] Example {ex_idx + 1} skipped, missing fields: {[f for f in field_order if f not in fields or not fields[f].strip()]}")
    return outputs


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
    test_from_examples_file("./generated/validate_examples (do not edit this file).txt")
