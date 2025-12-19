import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CSV_FILE = Path("resultaten_scriptie_400.csv")

def main():
    if not CSV_FILE.exists():
        print(f"Bestand {CSV_FILE} niet gevonden. Draai eerst je experiment!")
        return

    df = pd.read_csv(CSV_FILE)
    
    df = df.dropna(subset=["delta_ppl", "capacity_bits"])

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="system", y="delta_ppl", data=df, palette="Set2", showfliers=False)
    sns.stripplot(x="system", y="delta_ppl", data=df, color=".25", alpha=0.3, jitter=True)
    
    plt.title("Invloed van Steganografie op Perplexity (n=400)", fontsize=14)
    plt.ylabel("Stijging in Pseudo-Perplexity (Delta PPL)", fontsize=12)
    plt.xlabel("Systeem", fontsize=12)
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig("figuur_1_boxplot_ppl.png", dpi=400)
    print("Opgeslagen: figuur_1_boxplot_ppl.png")

    subset = df[df["system"] == "synonyms"]
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x="capacity_bits", y="delta_ppl", data=subset, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.title("Correlatie tussen Capaciteit en Detecteerbaarheid (Synoniemen)", fontsize=14)
    plt.xlabel("Aantal verborgen bits (Capaciteit)", fontsize=12)
    plt.ylabel("Stijging in PPL", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("figuur_2_scatter_cap_vs_ppl.png", dpi=400)
    print("Opgeslagen: figuur_2_scatter_cap_vs_ppl.png")

    summary = df.groupby("system").agg({
        "capacity_bits": ["mean", "max"],
        "delta_ppl": ["mean", "std"]
    }).round(2)
    
    print("\n=== TABEL VOOR SCRIPTIE ===")
    print(summary)
    print("===========================")

if __name__ == "__main__":
    main()