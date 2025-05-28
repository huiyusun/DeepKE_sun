import difflib
import re
import json
import os


def normalize_fragmented_fields(decoded, dataset='tacred'):
    def strip_all_trailing_punct_special(s):
        s = s.rstrip()
        abbreviations = [
            "Inc.", "Co.", "Ltd.", "Corp.", "Dr.", "Mr.", "Ms.", "Mrs.", "Jr.", "Sr.", "St.", "Mt.",
            "U.S.", "U.K.", "U.N.", "U.A.E.", "Ph.D.", "M.D.", "B.A.", "B.S.", "M.A.", "M.S.", "Prof."
        ]
        # Find how many trailing periods etc.
        m = re.match(r"^(.*?)([.。;]+)?$", s)
        if m:
            core, punct = m.groups()
            punct = punct or ""
            # If the core matches a known abbreviation (with or without trailing dot), preserve up to two periods
            for abbr in abbreviations:
                abbr_base = abbr[:-1]
                if core.strip().lower() == abbr.lower() or core.strip().lower() == abbr_base.lower():
                    return core.strip() + punct[:2]
            # Otherwise, for all others, remove all trailing punctuation
            return core.strip()
        return s.strip()

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

        # Robust field extraction using position indexing to avoid overlap/label confusion
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

        # Apply your normal cleaning for each field
        relation = found.get("relation", "").replace(" ", "").strip(" .。:：").lower()
        context = found.get("context", "").strip()
        head_entity = strip_all_trailing_punct_special(found.get("head entity", "").strip())
        head_type = re.sub(r"[^A-Z]", "", found.get("head type", "").upper())
        # Find the 'Tail Entity' immediately before 'Tail Type'
        tail_entity = ""
        tail_entity_matches = list(
            re.finditer(r"Tail Entity\s*[:：]\s*(.*?)(?=(Tail Type\s*[:：]|Head Entity\s*[:：]|Relation\s*[:：]|Context\s*[:：]|Head Type\s*[:：]|$))", text, re.IGNORECASE | re.DOTALL))
        if tail_entity_matches:
            tail_entity = tail_entity_matches[-1].group(1).strip()
        else:
            # Fallback: use old method if no match
            tail_entity = found.get("tail entity", "").strip()
        tail_entity = strip_all_trailing_punct_special(tail_entity)
        tail_type = re.sub(r"[^A-Z]", "", found.get("tail type", "").upper())

        # Context punctuation: end with period unless already ends with . .. ...
        context = re.sub(r"\s+", " ", context)
        if context and not re.search(r"\.\.\.$", context) and not re.search(r"\.\.$", context) and not re.search(r"\.$", context):
            context += "."

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
        # Only include output if at least relation and context are present
        if not (relation and context):
            continue
        outputs.append(fields)
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
    test_from_examples_file("./generated/validate_examples.txt")
