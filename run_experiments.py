import csv
import random
from pathlib import Path

import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from method_stego import (
    encode_bits_with_synonyms,
    encode_bits_with_abbreviations,
    encode_bits_with_combined,
    compute_capacity_synonyms,
    compute_capacity_abbreviations,
    compute_capacity_combined,
    pseudo_perplexity_bertje
)

INPUT_BESTAND = Path("cover_tekst.txt")

OUTPUT_BESTAND = Path("resultaten_scriptie_400.csv") 

AANTAL_ZINNEN = None


def laad_zinnen(pad, max_aantal=None):
    """Leest het tekstbestand in en geeft een lijst zinnen terug."""
    zinnen = []
    if not pad.exists():
        print(f"FOUT: Bestand {pad} niet gevonden!")
        return []
        
    with pad.open("r", encoding="utf-8") as f:
        for regel in f:
            zin = regel.strip()
            if zin:
                zinnen.append(zin)
                if max_aantal and len(zinnen) >= max_aantal:
                    break
    return zinnen


def genereer_random_bits(lengte=100):
    """
    Maakt een lange sliert willekeurige 0 en 1.
    We gebruiken dit om elke zin maximaal te vullen.
    """
    bits = ""
    for _ in range(lengte):
        bits += str(random.randint(0, 1))
    return bits


def main():
    print("--- START EXPERIMENT ---")
    
    zinnen = laad_zinnen(INPUT_BESTAND, max_aantal=AANTAL_ZINNEN)
    if not zinnen:
        return
    
    print(f"{len(zinnen)} zinnen ingeladen. Start verwerking...")

    with OUTPUT_BESTAND.open("w", encoding="utf-8", newline="") as f:
        schrijver = csv.writer(f)

        schrijver.writerow([
            "zin_id",
            "system",
            "cover_text",
            "stego_text",
            "capacity_bits",
            "bits_hidden",
            "ppl_cover",
            "ppl_stego",
            "delta_ppl"
        ])

        for i, cover_zin in enumerate(zinnen, start=1):
            print(f"Verwerken zin {i}/{len(zinnen)}...", end="\r")

            try:
                ppl_cover = pseudo_perplexity_bertje(cover_zin)
            except:
                ppl_cover = 0.0

            random_bitstring = genereer_random_bits(200)

            cap_syn = compute_capacity_synonyms(cover_zin)

            stego_syn, bits_gebruikt_syn, _ = encode_bits_with_synonyms(cover_zin, random_bitstring)

            try:
                ppl_syn = pseudo_perplexity_bertje(stego_syn)
            except:
                ppl_syn = 0.0

            schrijver.writerow([
                i, 
                "synonyms", 
                cover_zin, 
                stego_syn, 
                cap_syn, 
                bits_gebruikt_syn, 
                ppl_cover, 
                ppl_syn, 
                ppl_syn - ppl_cover
            ])

            cap_abbr = compute_capacity_abbreviations(cover_zin)
            stego_abbr, bits_gebruikt_abbr, _ = encode_bits_with_abbreviations(cover_zin, random_bitstring)
            
            try:
                ppl_abbr = pseudo_perplexity_bertje(stego_abbr)
            except:
                ppl_abbr = 0.0

            schrijver.writerow([
                i, "abbreviations", cover_zin, stego_abbr, cap_abbr, bits_gebruikt_abbr, 
                ppl_cover, ppl_abbr, ppl_abbr - ppl_cover
            ])

            cap_comb = compute_capacity_combined(cover_zin)
            stego_comb, bits_gebruikt_comb, _ = encode_bits_with_combined(cover_zin, random_bitstring)
            
            try:
                ppl_comb = pseudo_perplexity_bertje(stego_comb)
            except:
                ppl_comb = 0.0

            schrijver.writerow([
                i, "combined", cover_zin, stego_comb, cap_comb, bits_gebruikt_comb, 
                ppl_cover, ppl_comb, ppl_comb - ppl_cover
            ])

    print(f"\n\nKlaar! De resultaten staan in: {OUTPUT_BESTAND}")

if __name__ == "__main__":
    main()