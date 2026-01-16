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

    df_plot = df.copy()
    df_plot['system'] = df_plot['system'].map({
        'synonyms': 'Synoniemen', 
        'abbreviations': 'Afkortingen', 
        'combined': 'Gecombineerd'
    })

    plt.figure(figsize=(10, 6))

    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    sns.boxplot(x="system", y="delta_ppl", data=df_plot, palette="Set2", 
                showfliers=False, width=0.5, zorder=1)

    sns.stripplot(x="system", y="delta_ppl", data=df_plot, color=".25", 
                  alpha=0.4, size=3, jitter=True, zorder=2)
    
    plt.title("Invloed van steganografie op perplexity (n=400) (Extreme uitschieters > 200 buiten beeld)", fontsize=14, fontweight='bold')
    plt.ylabel(r"Stijging in Pseudo-Perplexity ($\Delta$PPL)", fontsize=12)
    plt.xlabel("", fontsize=12)

    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.ylim(-20, 200) 

    plt.tight_layout()
    plt.savefig("figuur_1_boxplot_ppl.png", dpi=400)
    print("Opgeslagen: figuur_1_boxplot_ppl.png (Verbeterde versie)")

    subset = df[df["system"] == "synonyms"]
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x="capacity_bits", y="delta_ppl", data=subset, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.title("Correlatie tussen capaciteit en detecteerbaarheid (Synoniemen)", fontsize=14)
    plt.xlabel("Aantal verborgen bits (Capaciteit)", fontsize=12)
    plt.ylabel(r"Stijging in PPL ($\Delta$PPL)", fontsize=12)
    
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