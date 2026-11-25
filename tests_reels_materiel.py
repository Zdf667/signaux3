import csv
import re
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from core_modem import (
    FREQUENCE_ECHANTILLONNAGE,
    NIVEAU_MINIMAL_CONSEILLE,
    TEXTE_REFERENCE,
    VALENCES_AUTORISEES,
    afficher_spectrogramme,
    demoduler_bits,
    duree_emission,
    generer_signal_emission,
    mesurer_performances,
    texte_vers_bits,
)

DOSSIER_DONNEES_REELLES = Path(__file__).resolve().parent / "donnees_reelles"
FICHIER_REGISTRE = DOSSIER_DONNEES_REELLES / "registre_enregistrements.csv"
DELAI_AVANT_EMISSION = 3
MARGE_ECOUTE = 10.0
DISTANCES_AUTORISEES = ("1m", "2m")
BRUITS_AUTORISES = ("silence", "musique")
REPETITIONS_AUTORISEES = ("1", "2", "3")
COLONNES_REGISTRE = (
    "fichier_wav",
    "texte_emis",
    "valence",
    "distance",
    "bruit_ambiant",
    "repetition",
    "appareils",
    "texte_decode",
    "ber",
    "taux_caracteres_corrects",
    "debit_brut_bits_par_seconde",
    "debit_effectif_bits_par_seconde",
)


def demander_choix(question, choix_possibles):
    while True:
        reponse = input(f"{question} {choix_possibles} : ").strip().lower()
        if reponse in choix_possibles:
            return reponse
        print(f"Réponse invalide, choisir parmi {choix_possibles}.")


def demander_texte_libre(question):
    while True:
        reponse = re.sub(r"[^A-Za-z0-9-]+", "-", input(f"{question} : ").strip()).strip("-")
        if reponse:
            return reponse
        print("Réponse vide, recommencez.")


def demander_valence():
    choix_valence = tuple(str(valence) for valence in VALENCES_AUTORISEES)
    return int(demander_choix("Valence M", choix_valence))


def emettre_signal(signal_emission, description):
    for secondes_restantes in range(DELAI_AVANT_EMISSION, 0, -1):
        print(f"[TX] Émission de {description} dans {secondes_restantes} s...")
        time.sleep(1)
    sd.play(signal_emission, samplerate=FREQUENCE_ECHANTILLONNAGE)
    sd.wait()
    print("[TX] Terminé.")


def emettre_audio(texte, valence):
    emettre_signal(generer_signal_emission(texte, valence), f"'{texte}' (M = {valence})")


def enregistrer_audio(duree_ecoute):
    print(f"[RX] Enregistrement de {duree_ecoute:.1f} s : lancez l'émission maintenant.")
    enregistrement = sd.rec(
        round(duree_ecoute * FREQUENCE_ECHANTILLONNAGE),
        samplerate=FREQUENCE_ECHANTILLONNAGE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return enregistrement.ravel()


def enregistrer_dans_registre(ligne_registre):
    lignes_conservees = []
    if FICHIER_REGISTRE.exists():
        with FICHIER_REGISTRE.open(newline="", encoding="utf-8") as fichier_csv:
            lignes_conservees = [
                ligne
                for ligne in csv.DictReader(fichier_csv)
                if ligne["fichier_wav"] != ligne_registre["fichier_wav"]
            ]
    with FICHIER_REGISTRE.open("w", newline="", encoding="utf-8") as fichier_csv:
        ecrivain = csv.DictWriter(fichier_csv, fieldnames=COLONNES_REGISTRE)
        ecrivain.writeheader()
        ecrivain.writerows(lignes_conservees)
        ecrivain.writerow(ligne_registre)


def verifier_niveau(enregistrement):
    niveau_maximal = float(np.abs(enregistrement).max()) if enregistrement.size else 0.0
    print(f"[RX] Niveau maximal capté : {niveau_maximal:.3f} (1.000 = saturation)")
    if niveau_maximal < NIVEAU_MINIMAL_CONSEILLE:
        print(
            f"[RX] Attention : niveau inférieur à {NIVEAU_MINIMAL_CONSEILLE}, "
            "montez le volume de l'émetteur ou le gain du micro"
        )


def recevoir_audio(valence):
    etiquettes = {
        "texte_emis": TEXTE_REFERENCE,
        "valence": valence,
        "distance": demander_choix("Distance", DISTANCES_AUTORISEES),
        "bruit_ambiant": demander_choix("Bruit ambiant", BRUITS_AUTORISES),
        "repetition": demander_choix("Numéro de répétition", REPETITIONS_AUTORISEES),
        "appareils": demander_texte_libre("Appareils émetteur-récepteur (ex : PC-Dell_vers_iPhone12)"),
    }
    nom_fichier = (
        f"test_M{valence}_{etiquettes['distance']}_{etiquettes['bruit_ambiant']}"
        f"_rep{etiquettes['repetition']}_{etiquettes['appareils']}.wav"
    )
    chemin_wav = DOSSIER_DONNEES_REELLES / nom_fichier
    if chemin_wav.exists():
        raise FileExistsError(f"{chemin_wav} existe déjà : supprimez-le ou changez d'étiquettes")

    enregistrement = enregistrer_audio(duree_emission(len(TEXTE_REFERENCE), valence) + MARGE_ECOUTE)
    wavfile.write(chemin_wav, FREQUENCE_ECHANTILLONNAGE, enregistrement)
    print(f"[RX] Audio sauvegardé : {chemin_wav}")
    verifier_niveau(enregistrement)

    bits_decodes = demoduler_bits(enregistrement, len(texte_vers_bits(TEXTE_REFERENCE)), valence)
    performances = mesurer_performances(TEXTE_REFERENCE, bits_decodes, valence)
    enregistrer_dans_registre({"fichier_wav": nom_fichier, **etiquettes, **performances})

    print(f">>> Texte décodé : {performances['texte_decode']!r}")
    print(f">>> BER : {performances['ber']:.2%}")
    print(f">>> Caractères corrects : {performances['taux_caracteres_corrects']:.0%}")
    print(f">>> Débit effectif : {performances['debit_effectif_bits_par_seconde']:.1f} bit/s")
    afficher_spectrogramme(enregistrement, titre=f"Spectrogramme - {nom_fichier}")


def lancer_test_materiel():
    print("=== MODEM AUDIO M-FSK ===")
    DOSSIER_DONNEES_REELLES.mkdir(exist_ok=True)
    mode = demander_choix("Mode", ("tx", "rx"))
    valence = demander_valence()
    if mode == "tx":
        emettre_audio(TEXTE_REFERENCE, valence)
    else:
        recevoir_audio(valence)


if __name__ == "__main__":
    lancer_test_materiel()
