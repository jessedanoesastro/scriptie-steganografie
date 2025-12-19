import csv
from pathlib import Path

INPUT_FILE = "GiGaNT-Molex-afkortingen.tsv"
OUTPUT_FILE = "abbreviations_dict.py"

def convert_tsv_to_dict():
    abbr_dict = {}
    
    print(f"Bezig met lezen van {INPUT_FILE}...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        
        count = 0
        for row in reader:
            if len(row) < 4:
                continue

            afkorting = row[1].strip()
            pos_tag = row[2]
            voluit = row[3].strip()
            
            if len(afkorting) >= len(voluit):
                continue
            
            abbr_dict[voluit.lower()] = afkorting
            count += 1

    print(f"Gevonden afkortingen: {count}")
    
    print(f"Schrijven naar {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# Deze lijst is gegenereerd uit de GiGaNT-Molex dataset van het INT\n")
        out.write("ABBREVIATIONS = {\n")
        for full, abbr in sorted(abbr_dict.items()):
            full_clean = full.replace("'", "\\'")
            abbr_clean = abbr.replace("'", "\\'")
            out.write(f"    '{full_clean}': '{abbr_clean}',\n")
        out.write("}\n")
    
    print("Klaar! Je kunt nu 'from abbreviations_dict import ABBREVIATIONS' gebruiken.")

if __name__ == "__main__":
    convert_tsv_to_dict()