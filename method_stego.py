import math
import re
import spacy
import wn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from abbreviations_dict import ABBREVIATIONS

print("Modellen laden... even geduld.")

nlp = spacy.load("nl_core_news_md")

BERTJE_MODEL_NAAM = "GroNLP/bert-base-dutch-cased"
tokenizer = AutoTokenizer.from_pretrained(BERTJE_MODEL_NAAM)
model = AutoModelForMaskedLM.from_pretrained(BERTJE_MODEL_NAAM)
model.eval()


TOEGESTANE_POS = ["NOUN", "PROPN", "ADJ", "VERB"]


def get_synonyms_from_wordnet(woord, pos_tag):
    """
    Zoekt synoniemen in Open Dutch WordNet.
    Geeft een lijstje terug van synoniemen die uit één woord bestaan.
    """
    woord = woord.lower()
    
    wn_pos = None
    if pos_tag.startswith("N"): wn_pos = "n"
    elif pos_tag.startswith("V"): wn_pos = "v"
    elif pos_tag.startswith("A"): wn_pos = "a"
    elif pos_tag.startswith("R"): wn_pos = "r"
    
    if not wn_pos:
        return []

    gevonden_synoniemen = set()
    try:
        synsets = wn.synsets(woord, pos=wn_pos, lang="nl")
        for synset in synsets:
            for sense in synset.senses():
                synoniem = sense.word().lemma().lower()
                
                if synoniem != woord and " " not in synoniem:
                    gevonden_synoniemen.add(synoniem)
    except:
        return []

    return list(gevonden_synoniemen)


def filter_met_bertje(zin, woord_index, kandidaten):
    """
    Gebruikt BERTje om te kijken welke synoniemen in de zin passen.
    We vervangen het woord door [MASK] en kijken wat BERTje voorspelt.
    """
    doc = nlp(zin)
    tokens = [t.text for t in doc]
    
    tokens[woord_index] = tokenizer.mask_token
    masked_zin = " ".join(tokens)
    
    inputs = tokenizer(masked_zin, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
    
    logits = output.logits[0, mask_token_index]
    top_50_indices = torch.topk(logits, 50).indices
    
    top_50_woorden = []
    for idx in top_50_indices:
        woord = tokenizer.decode([idx]).strip()
        top_50_woorden.append(woord)
        
    goede_kandidaten = []
    
    origineel = kandidaten[0]
    goede_kandidaten.append(origineel)
    
    for kandidaat in kandidaten[1:]:
        if kandidaat in top_50_woorden:
            goede_kandidaten.append(kandidaat)
            
    return goede_kandidaten


def maak_zin_van_tokens(tokens):
    """Plakt tokens aan elkaar, maar houdt rekening met leestekens."""
    zin = ""
    leestekens = [".", ",", ":", ";", "!", "?", ")"]
    
    for i, token in enumerate(tokens):
        if token in leestekens:
            zin += token
        elif i > 0 and tokens[i-1] in ["(", "[", "{", "'", '"']:
            zin += token
        else:
            if i == 0:
                zin += token
            else:
                zin += " " + token
    return zin


def encode_bits_with_synonyms(zin, bitstring):
    """
    Probeert bits te verstoppen door woorden te vervangen door synoniemen.
    0 = origineel woord
    1 = synoniem
    """
    doc = nlp(zin)
    tokens = [t.text for t in doc]
    
    huidige_bit_index = 0
    capaciteit = 0
    

    for i, token in enumerate(doc):
        if not token.is_alpha or token.pos_ not in TOEGESTANE_POS:
            continue
            
        synoniemen = get_synonyms_from_wordnet(token.lemma_, token.pos_)
        if not synoniemen:
            continue

        alle_opties = [token.text] + synoniemen

        gefilterde_opties = filter_met_bertje(zin, i, alle_opties)

        if len(gefilterde_opties) < 2:
            continue

        capaciteit += 1

        if huidige_bit_index < len(bitstring):
            bit = bitstring[huidige_bit_index]
            huidige_bit_index += 1

            if bit == "1":
                nieuw_woord = gefilterde_opties[1]

                if token.text[0].isupper():
                    nieuw_woord = nieuw_woord.capitalize()
                
                tokens[i] = nieuw_woord

    nieuwe_zin = maak_zin_van_tokens(tokens)
    return nieuwe_zin, huidige_bit_index, capaciteit


def compute_capacity_synonyms(zin):
    """Berekent hoeveel bits we in theorie in deze zin kwijt kunnen."""
    _, _, cap = encode_bits_with_synonyms(zin, "")
    return cap


def encode_bits_with_abbreviations(zin, bitstring):
    """
    Verstopt bits door woorden te vervangen door hun afkorting.
    Bijvoorbeeld: 'bijvoorbeeld' -> 'bv.'
    """
    nieuwe_zin = zin
    gebruikte_bits = 0
    capaciteit = 0
    gesorteerde_lijst = sorted(ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for voluit, afkorting in gesorteerde_lijst:

        patroon = re.compile(rf"\b{re.escape(voluit)}\b", flags=re.IGNORECASE)
        
        if patroon.search(nieuwe_zin):
            capaciteit += 1
            if gebruikte_bits < len(bitstring):
                bit = bitstring[gebruikte_bits]
                gebruikte_bits += 1
                if bit == "1":
                    nieuwe_zin = patroon.sub(afkorting, nieuwe_zin, count=1)
    
    return nieuwe_zin, gebruikte_bits, capaciteit


def compute_capacity_abbreviations(zin):
    _, _, cap = encode_bits_with_abbreviations(zin, "")
    return cap


def encode_bits_with_combined(zin, bitstring):
    zin_na_syn, bits_syn, cap_syn = encode_bits_with_synonyms(zin, bitstring)
    
    resterende_bits = bitstring[bits_syn:]
    zin_compleet, bits_abbr, cap_abbr = encode_bits_with_abbreviations(zin_na_syn, resterende_bits)
    
    totaal_gebruikt = bits_syn + bits_abbr
    totaal_capaciteit = cap_syn + cap_abbr
    
    return zin_compleet, totaal_gebruikt, totaal_capaciteit


def compute_capacity_combined(zin):
    return compute_capacity_synonyms(zin) + compute_capacity_abbreviations(zin)


def pseudo_perplexity_bertje(zin):
    """
    Berekent hoe 'vreemd' BERTje de zin vindt.
    Lagere score is beter (natuurlijker).
    """
    inputs = tokenizer(zin, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    
    nll_som = 0.0
    aantal_tokens = 0
    
    with torch.no_grad():
        for i in range(1, len(input_ids) - 1):
            target_id = input_ids[i].clone()

            temp_input = input_ids.clone()
            temp_input[i] = tokenizer.mask_token_id

            output = model(temp_input.unsqueeze(0))
            logits = output.logits[0, i]

            log_probs = torch.log_softmax(logits, dim=-1)
            token_nll = -log_probs[target_id]
            
            nll_som += token_nll.item()
            aantal_tokens += 1
            
    if aantal_tokens == 0:
        return 0.0
        
    gemiddelde_nll = nll_som / aantal_tokens
    return math.exp(gemiddelde_nll)