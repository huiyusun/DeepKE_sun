import json
import re
import difflib
import os
from processDA import normalize_fragmented_fields


def test_normalize_fragmented_fields():
    test_inputs = [
        # 1. Clean, single-line example
        """Relation: per:title. Context: The author J.K. Rowling wrote the famous Harry Potter series. Head Entity: J.K. Rowling. Head Type: PERSON. Tail Entity: author. Tail Type: TITLE.""",
        # 2. Fragmented field values
        """Relation :
        org :
        top_members /
        employees .
        Context : "The CEO of Apple, Tim Cook, presented new products." 
        Head entity : Apple. 
        Head type : ORGANIZATION. 
        Tail entity : Tim Cook. 
        Tail type : PERSON.""",
        # 3. Missing colon in relation
        """Relation : org top_members / employees .Context : "The founder and CEO of SpaceX , Elon Musk , celebrated the recent launch success with his team ."Head entity  : SpaceX.Head type  :
        ORGANIZATION.Tail entity  :
        Elon Musk.Tail type  :
        PERSON.""",
        # 4. Fully lowercase malformed field headers
        """relation:
        per:
        cities_of_residence.
        context:
        "Barack Obama lived in Chicago for several years."
        head entity:
        Barack Obama.
        head type:
        PERSON.
        tail entity:
        Chicago.
        tail type:
        CITY.""",
        # 5. Extra spacing or formatting noise
        """  Relation  :  per:employee_of . 
        Context : "John Doe works at Google as a senior engineer." 
        Head Entity : John Doe. 
        Head Type : PERSON. 
        Tail Entity : Google. 
        Tail Type : ORGANIZATION. """
    ]

    print("\n--- TESTING normalize_fragmented_fields ---")
    for i, inp in enumerate(test_inputs):
        print(f"\nTest {i + 1} Input:")
        print(inp)
        result = normalize_fragmented_fields(inp)
        print(f"Test {i + 1} Output:")
        for line in result:
            print(line)
        print("---")

if __name__ == '__main__':
    test_normalize_fragmented_fields()