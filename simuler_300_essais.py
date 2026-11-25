import csv
from pathlib import Path

import numpy as np

from core_modem import (
    FREQUENCE_ECHANTILLONNAGE,
    TEXTE_REFERENCE,
    ajouter_bruit_blanc,
    demoduler_bits,
    generer_signal_emission,
    mesurer_performances,
    texte_vers_bits,
)

VALENCES_SIMULEES = (2, 4, 8, 16)
SNR_MAXIMAL_DB = 30.0
SNR_MINIMAL_DB = -5.0
NOMBRE_POINTS_SNR_PAR_VALENCE = 75
DUREE_SILENCE_MAXIMALE = 0.5
GRAINE_ALEATOIRE = 2026
FICHIER_RESULTATS = Path(__file__).resolve().parent / "resultats_simulations_MFSK.csv"
COLONNES_RESULTATS = (
    "valence",
    "snr_db",
    "ber",
    "taux_caracteres_corrects",
    "debit_brut_bits_par_seconde",
    "debit_effectif_bits_par_seconde",
    "texte_decode",
)


def simuler_essai(valence, snr_db, generateur_aleatoire):
    signal_emis = generer_signal_emission(TEXTE_REFERENCE, valence)
    echantillons_silence_maximal = round(DUREE_SILENCE_MAXIMALE * FREQUENCE_ECHANTILLONNAGE)
    silence_avant = np.zeros(generateur_aleatoire.integers(0, echantillons_silence_maximal + 1))
    silence_apres = np.zeros(generateur_aleatoire.integers(0, echantillons_silence_maximal + 1))
    signal_canal = np.concatenate((silence_avant, signal_emis, silence_apres))
    puissance_signal_emis = np.mean(signal_emis**2)
    signal_bruite = ajouter_bruit_blanc(signal_canal, snr_db, puissance_signal_emis, generateur_aleatoire)
    bits_decodes = demoduler_bits(signal_bruite, len(texte_vers_bits(TEXTE_REFERENCE)), valence)
    return {"valence": valence, "snr_db": snr_db, **mesurer_performances(TEXTE_REFERENCE, bits_decodes, valence)}


def lancer_simulations():
    generateur_aleatoire = np.random.default_rng(GRAINE_ALEATOIRE)
    valeurs_snr_db = np.linspace(SNR_MAXIMAL_DB, SNR_MINIMAL_DB, NOMBRE_POINTS_SNR_PAR_VALENCE)
    nombre_essais = len(VALENCES_SIMULEES) * len(valeurs_snr_db)
    print(f"Démarrage des {nombre_essais} simulations...")

    resultats = []
    for valence in VALENCES_SIMULEES:
        print(f"Simulation pour M = {valence}...")
        for snr_db in valeurs_snr_db:
            resultats.append(simuler_essai(valence, float(snr_db), generateur_aleatoire))

    with FICHIER_RESULTATS.open("w", newline="", encoding="utf-8") as fichier_csv:
        ecrivain = csv.DictWriter(fichier_csv, fieldnames=COLONNES_RESULTATS)
        ecrivain.writeheader()
        ecrivain.writerows(resultats)
    print(f"Terminé : {len(resultats)} essais enregistrés dans {FICHIER_RESULTATS}")


if __name__ == "__main__":
    lancer_simulations()
