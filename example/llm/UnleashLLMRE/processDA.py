from difflib import get_close_matches
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
    def clean_entity(entity, context):
        s = entity.strip()
        # Remove all trailing non-word, non-space, non-period chars, but preserve periods
        s_core = re.sub(r'[^\w\s.]+$', '', s)
        # Try context match: two, one, or no periods, but only if next char in context is not a word char
        for period_count in (2, 1, 0):
            candidate = s_core + ('.' * period_count)
            pattern = re.escape(candidate) + r'(\W|$)'
            if candidate and re.search(pattern, context):
                return candidate
        # If not in context, reduce trailing periods to max one (unless original expected two, but default to one)
        m = re.match(r'^(.*?)(\.{1,})?$', s_core)
        core = m.group(1)
        trailing = m.group(2) if m.group(2) else ''
        if trailing:
            trailing = '.'  # Always reduce to one
        # Remove all other trailing punctuation (except periods)
        result = (core + trailing).strip()
        result = re.sub(r'[^\w\s.]+$', '', result)
        return result

    # Normalize separators and noise
    decoded = decoded.replace("：", ":")
    decoded = re.sub(r"[*@#\^•]+", "", decoded)
    decoded = re.sub(r"\s+", " ", decoded)
    # Insert newline before every 'Relation:' not at start of line
    decoded = re.sub(r"(?<!\n)(Relation\s*[:：])", r"\n\1", decoded, flags=re.IGNORECASE)

    # Split on every new Relation field, so each block is a separate example
    parts = re.split(r"\bRelation\s*[:：]", decoded, flags=re.IGNORECASE)
    outputs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Prepend so our regex can work as usual
        text = "Relation: " + part

        # field extraction using position indexing to avoid overlap/label confusion
        field_labels = [
            ("relation", "Relation"),
            ("context", "Context"),
            ("head_entity", "Head Entity"),
            ("head_type", "Head Type"),
            ("tail_entity", "Tail Entity"),
            ("tail_type", "Tail Type"),
        ]
        found = {}
        for m in re.finditer(r"(Relation|Context|Head Entity|Head Type|Tail Entity|Tail Type)\s*[:：]", text, re.IGNORECASE):
            label = m.group(1)
            start = m.end()
            # Find end of this field (start of next label or end of string)
            next_match = re.search(
                r"(Relation|Context|Head Entity|Head Type|Tail Entity|Tail Type)\s*[:：]", text[start:], re.IGNORECASE
            )
            end = start + next_match.start() if next_match else len(text)
            value = text[start:end].strip()
            # Only use the first occurrence of each label
            if label.lower() not in found:
                found[label.lower()] = value

        # Apply cleaning for each field
        relation = found.get("relation", "").replace(" ", "").strip(" .。:：").lower()
        context = found.get("context", "").strip()
        head_entity = clean_entity(found.get("head entity", "").strip(), context)
        head_type = re.sub(r"[^A-Z]", "", found.get("head type", "").upper())
        # Find the 'Tail Entity' immediately before 'Tail Type'
        tail_entity = ""
        tail_entity_matches = list(
            re.finditer(r"Tail Entity\s*[:：]\s*(.*?)(?=(Tail Type\s*[:：]|Head Entity\s*[:：]|Relation\s*[:：]|Context\s*[:：]|Head Type\s*[:：]|$))", text, re.IGNORECASE | re.DOTALL))
        if tail_entity_matches:
            tail_entity = tail_entity_matches[-1].group(1).strip()
        else:
            tail_entity = found.get("tail entity", "").strip()
        tail_entity = clean_entity(tail_entity, context)

        # --- Fuzzy entity correction using context ---
        from difflib import get_close_matches

        def best_match_in_context(entity, context):
            # Prefer candidates that match full words and are long enough
            entity_words = entity.strip().split()
            entity_len = len(entity_words)
            context_words = context.strip().split()
            candidates = []
            for n in range(max(1, entity_len-2), entity_len+3):
                for i in range(len(context_words) - n + 1):
                    chunk = " ".join(context_words[i:i+n])
                    candidates.append(chunk)
            # Fuzzy match with difflib
            matches = get_close_matches(entity, candidates, n=1, cutoff=0.7)
            if matches:
                # Return the match after removing trailing non-alphanum except one period
                match = matches[0]
                match = re.sub(r'[^\w\s]+$', '', match)
                if (match + '.') in context:
                    return match + '.'
                return match
            # fallback to cleaned entity
            return clean_entity(entity, context)

        # After cleaning, check for exact case-sensitive match first
        if head_entity not in context:
            corrected_head = best_match_in_context(head_entity, context)
            if corrected_head:
                head_entity = corrected_head

        if tail_entity not in context:
            corrected_tail = best_match_in_context(tail_entity, context)
            if corrected_tail:
                tail_entity = corrected_tail
        tail_type = re.sub(r"[^A-Z]", "", found.get("tail type", "").upper())

        # Context punctuation: end with period unless already ends with . .. ...
        context = re.sub(r"\s+", " ", context)
        if context and not re.search(r"\.\.\.$", context) and not re.search(r"\.\.$", context) and not re.search(r"\.$", context):
            context += "."

        # Fuzzy matching for relation and entity types
        rel_types = relation_types.get(dataset, [])
        ent_types = entity_types.get(dataset, [])

        # match relation
        if relation not in rel_types:
            matches = get_close_matches(relation, rel_types, n=1, cutoff=0.7)
            if matches:
                relation = matches[0]
        # match head_type
        if head_type not in ent_types:
            matches = get_close_matches(head_type, ent_types, n=1, cutoff=0.7)
            if matches:
                head_type = matches[0]
        # match tail_type
        if tail_type not in ent_types:
            matches = get_close_matches(tail_type, ent_types, n=1, cutoff=0.7)
            if matches:
                tail_type = matches[0]

        output = (
            f"Relation: {relation}. Context: {context} "
            f"Head Entity: {head_entity}. Head Type: {head_type}. "
            f"Tail Entity: {tail_entity}. Tail Type: {tail_type}."
        )
        fields = dict(
            relation=relation,
            context=context,
            head_entity=head_entity,
            head_type=head_type,
            tail_entity=tail_entity,
            tail_type=tail_type,
            text=output.strip()
        )

        # Only include output if all fields are present
        if not all([relation, context, head_entity, head_type, tail_entity, tail_type]):
            continue
        outputs.append(fields)

    return outputs


# extract the relation label and entity information matching the dataset format (e.g., tacred, retacred, tacrev)
def construct_relation(rel, label, datasetname, verbose=True):
    # Takes each output of normalize_fragmented_fields() as input rel
    if not rel or not isinstance(rel, dict):
        if verbose:
            print(f"[construct_relation] Skipping: input is None or not a dict: {rel}")
        return

    if rel["relation"] != label:
        if verbose:
            print(f"[construct_relation] Skipping: relation '{rel['relation']}' != target label '{label}'")
        return

    DAdata = {
        'text': rel["context"]
    }

    def validate_type(type_val, typeset, field):
        t_upper = type_val.upper()
        if (
                type_val in typeset
                or type_val.replace('_', ' ') in typeset
                or t_upper == "MISCELLANEOUS"
                or t_upper == "MISC"
        ):
            return "MISC" if t_upper in ("MISCELLANEOUS", "MISC") else type_val.upper().replace(" ", "_")
        else:
            if verbose:
                print(f"[construct_relation] Invalid {field} type: '{type_val}' not in {typeset}")
            return None

    # head entity and type
    head = rel["head_entity"]
    headtype = validate_type(rel["head_type"], entity_types[datasetname], "head")
    if not headtype:
        if verbose:
            print(f"[construct_relation] Example: {rel}")
        return
    DAdata["subj_type"] = headtype

    # tail entity and type
    tail = rel["tail_entity"]
    tailtype = validate_type(rel["tail_type"], entity_types[datasetname], "tail")
    if not tailtype:
        if verbose:
            print(f"[construct_relation] Example: {rel}")
        return
    DAdata["obj_type"] = tailtype

    # head and tail positions (case-insensitive matching, returns first occurrence)
    textlower = rel["context"].lower()
    headlower = head.lower()
    if headlower in textlower:
        hpos1 = textlower.index(headlower)
        hpos2 = hpos1 + len(headlower)
        truehead = rel["context"][hpos1:hpos2]
    else:
        if verbose:
            print(f"[construct_relation] Head entity '{head}' not found in context: {rel['context']}")
            print(f"Example: {rel}")
        return

    taillower = tail.lower()
    if taillower in textlower:
        tpos1 = textlower.index(taillower)
        tpos2 = tpos1 + len(taillower)
        truetail = rel["context"][tpos1:tpos2]
    else:
        if verbose:
            print(f"[construct_relation] Tail entity '{tail}' not found in context: {rel['context']}")
            print(f"Example: {rel}")
        return

    DAdata["subj"] = truehead
    DAdata["subj_start"], DAdata["subj_end"] = hpos1, hpos2
    DAdata["obj"] = truetail
    DAdata["obj_start"], DAdata["obj_end"] = tpos1, tpos2
    DAdata["relation"] = label

    return DAdata


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
    expected_lines = [line.strip() for line in expected_text.splitlines() if line.strip()]

    normalized = normalize_fragmented_fields(input_text)
    generated_lines = [entry["text"].strip() for entry in normalized]

    print("--- TEST RESULTS ---")
    passed = True
    input_blocks = [blk.strip() for blk in re.split(r"\n\s*\n", input_text) if blk.strip()]
    for i, (gen, exp) in enumerate(zip(generated_lines, expected_lines)):
        print(f"\n--- Example {i + 1} ---")
        if i < len(input_blocks):
            print(f"Input example:\n{input_blocks[i]}")
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
    test_from_examples_file("./generated/validate_examples.txt")
