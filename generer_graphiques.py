import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from core_modem import BITS_PAR_CARACTERE, TEXTE_REFERENCE

DOSSIER_PROJET = Path(__file__).resolve().parent
FICHIER_RESULTATS = DOSSIER_PROJET / "resultats_simulations_MFSK.csv"
DOSSIER_FIGURES = DOSSIER_PROJET / "resultats"
RESOLUTION_IMPRESSION_DPI = 300
PLANCHER_BER = 1 / (2 * len(TEXTE_REFERENCE) * BITS_PAR_CARACTERE)


def charger_resultats_par_valence():
    resultats_par_valence = defaultdict(lambda: defaultdict(list))
    with FICHIER_RESULTATS.open(newline="", encoding="utf-8") as fichier_csv:
        for ligne in csv.DictReader(fichier_csv):
            mesures_valence = resultats_par_valence[int(ligne["valence"])]
            for colonne, valeur in ligne.items():
                if colonne != "texte_decode":
                    mesures_valence[colonne].append(float(valeur))
    return resultats_par_valence


def tracer_figure(resultats_par_valence, colonne, titre, etiquette_ordonnee, nom_fichier, style_trace, configurer_axes):
    plt.figure(figsize=(10, 6))
    for valence in sorted(resultats_par_valence):
        mesures = resultats_par_valence[valence]
        plt.plot(mesures["snr_db"], mesures[colonne], style_trace, label=f"M = {valence}")
    configurer_axes()
    plt.title(titre, fontsize=16, pad=20)
    plt.xlabel("Rapport signal sur bruit - SNR (dB)", fontsize=14)
    plt.ylabel(etiquette_ordonnee, fontsize=14)
    plt.legend(title="Valence", fontsize=12)
    plt.tight_layout()
    plt.savefig(DOSSIER_FIGURES / nom_fichier, dpi=RESOLUTION_IMPRESSION_DPI)
    plt.close()


def configurer_axes_ber():
    plt.yscale("log")
    plt.ylim(PLANCHER_BER / 2, 1)
    plt.axhline(PLANCHER_BER, color="grey", linestyle=":", label="BER = 0 (plancher d'affichage)")


def configurer_axes_caracteres():
    plt.ylim(-0.05, 1.05)
    plt.axhline(1.0, color="red", linestyle=":", alpha=0.5, label="100 % de réussite")


def configurer_axes_debit():
    plt.ylim(bottom=0)


def generer_graphiques():
    if not FICHIER_RESULTATS.exists():
        raise FileNotFoundError(f"{FICHIER_RESULTATS} introuvable : lancez d'abord simuler_300_essais.py")

    resultats_par_valence = charger_resultats_par_valence()
    for mesures in resultats_par_valence.values():
        mesures["ber_affiche"] = [max(ber, PLANCHER_BER) for ber in mesures["ber"]]

    plt.style.use("seaborn-v0_8-whitegrid")
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    print("Génération des graphiques en cours...")

    tracer_figure(
        resultats_par_valence,
        "ber_affiche",
        "Robustesse : taux d'erreur binaire (BER) selon le SNR",
        "Taux d'erreur binaire (BER)",
        "poster_fig1_BER_vs_SNR.png",
        "o-",
        configurer_axes_ber,
    )
    tracer_figure(
        resultats_par_valence,
        "taux_caracteres_corrects",
        "Lisibilité : caractères décodés à l'identique",
        "Taux de caractères corrects",
        "poster_fig2_Caracteres_vs_SNR.png",
        "s--",
        configurer_axes_caracteres,
    )
    tracer_figure(
        resultats_par_valence,
        "debit_effectif_bits_par_seconde",
        "Débit effectif : bits utiles par seconde selon le SNR",
        "Débit effectif (bit/s)",
        "poster_fig3_Debit_vs_SNR.png",
        "^-",
        configurer_axes_debit,
    )
    print(f"Terminé : figures enregistrées dans {DOSSIER_FIGURES}")


if __name__ == "__main__":
    generer_graphiques()
