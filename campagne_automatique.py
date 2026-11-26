import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from core_modem import (
    FREQUENCE_ECHANTILLONNAGE,
    NIVEAU_MINIMAL_CONSEILLE,
    SEUIL_DETECTION_PREAMBULE,
    TEXTE_REFERENCE,
    generer_preambule,
    generer_signal_emission,
    localiser_preambule,
)
from tests_reels_materiel import (
    construire_nom_fichier,
    demander_choix,
    demander_texte_libre,
    mesurer_niveau,
    sauvegarder_et_analyser,
)

DOSSIER_CAMPAGNE = Path(__file__).resolve().parent / "donnees_campagne"
CONDITIONS = {
    "1": {"distance": "1m", "bruit_ambiant": "silence"},
    "2": {"distance": "2m", "bruit_ambiant": "silence"},
    "3": {"distance": "1m", "bruit_ambiant": "musique"},
    "4": {"distance": "2m", "bruit_ambiant": "musique"},
}
VALENCES_CAMPAGNE = (2, 4, 8)
REPETITIONS_CAMPAGNE = ("1", "2", "3")
DUREE_SILENCE_AVANT = 1.0
DUREE_SILENCE_APRES = 1.5
PAUSE_ENTRE_ESSAIS = 2.0
TENTATIVES_MAXIMALES = 3
NIVEAU_MAXIMAL_CONSEILLE = 0.9


def emettre_et_enregistrer(signal_emission):
    signal_complet = np.concatenate(
        (
            np.zeros(round(DUREE_SILENCE_AVANT * FREQUENCE_ECHANTILLONNAGE)),
            signal_emission,
            np.zeros(round(DUREE_SILENCE_APRES * FREQUENCE_ECHANTILLONNAGE)),
        )
    ).astype(np.float32)
    enregistrement = sd.playrec(signal_complet, samplerate=FREQUENCE_ECHANTILLONNAGE, channels=1, dtype="float32")
    sd.wait()
    return enregistrement.ravel()


def etiquettes_essai(condition, valence, repetition, appareils):
    return {
        "texte_emis": TEXTE_REFERENCE,
        "valence": valence,
        "distance": condition["distance"],
        "bruit_ambiant": condition["bruit_ambiant"],
        "repetition": repetition,
        "appareils": appareils,
    }


def lister_essais(condition, appareils):
    return [
        etiquettes_essai(condition, valence, repetition, appareils)
        for valence in VALENCES_CAMPAGNE
        for repetition in REPETITIONS_CAMPAGNE
    ]


def essai_deja_fait(etiquettes):
    return (DOSSIER_CAMPAGNE / construire_nom_fichier(etiquettes)).exists()


def decrire_condition(condition):
    bruit = "musique de fond" if condition["bruit_ambiant"] == "musique" else "silence"
    return f"{condition['distance']}, {bruit}"


def afficher_menu(appareils):
    print("\n=== CAMPAGNE AUTOMATIQUE M-FSK ===")
    for numero, condition in CONDITIONS.items():
        essais = lister_essais(condition, appareils)
        nombre_faits = sum(essai_deja_fait(etiquettes) for etiquettes in essais)
        print(f"  {numero}. {decrire_condition(condition).ljust(22)} {nombre_faits}/{len(essais)} essais faits")
    print("  q. Quitter")


def preparer_setup(condition):
    print(f"\n--- Préparation : {decrire_condition(condition)} ---")
    print(f"1. Placez le micro à {condition['distance']} du haut-parleur du PC.")
    if condition["bruit_ambiant"] == "musique":
        print("2. Lancez la musique sur le téléphone (toujours le même morceau, même volume, même place).")
    else:
        print("2. Coupez toute source de bruit dans la pièce.")
    print("3. Ne bougez plus et restez silencieux pendant les essais.")
    input("Appuyez sur Entrée quand le setup est prêt...")


def calibrer_niveau():
    while True:
        print("[CALIBRATION] Émission d'un préambule de test...")
        niveau = mesurer_niveau(emettre_et_enregistrer(generer_preambule()))
        print(f"[CALIBRATION] Niveau capté : {niveau:.3f} (conseillé entre {NIVEAU_MINIMAL_CONSEILLE} et {NIVEAU_MAXIMAL_CONSEILLE})")
        if NIVEAU_MINIMAL_CONSEILLE <= niveau <= NIVEAU_MAXIMAL_CONSEILLE:
            return True
        conseil = "montez" if niveau < NIVEAU_MINIMAL_CONSEILLE else "baissez"
        print(f"[CALIBRATION] Niveau hors plage : {conseil} le volume du PC ou le gain du micro.")
        reponse = demander_choix("Recommencer la calibration, continuer quand même ou annuler", ("r", "c", "a"))
        if reponse != "r":
            return reponse == "c"


def realiser_essai(etiquettes):
    signal_emission = generer_signal_emission(TEXTE_REFERENCE, etiquettes["valence"])
    for tentative in range(1, TENTATIVES_MAXIMALES + 1):
        enregistrement = emettre_et_enregistrer(signal_emission)
        _, correlation_normalisee = localiser_preambule(enregistrement.astype(np.float64))
        if correlation_normalisee >= SEUIL_DETECTION_PREAMBULE:
            return sauvegarder_et_analyser(enregistrement, etiquettes, DOSSIER_CAMPAGNE)
        print(
            f"    Préambule non détecté (corrélation {correlation_normalisee:.3f}), "
            f"tentative {tentative}/{TENTATIVES_MAXIMALES}"
        )
        time.sleep(PAUSE_ENTRE_ESSAIS)
    return None


def lancer_condition(condition, appareils):
    essais = lister_essais(condition, appareils)
    essais_a_faire = [etiquettes for etiquettes in essais if not essai_deja_fait(etiquettes)]
    if not essais_a_faire:
        if demander_choix("Cette condition est déjà complète. Tout refaire ?", ("o", "n")) == "n":
            return
        essais_a_faire = essais

    preparer_setup(condition)
    if not calibrer_niveau():
        print("Condition annulée.")
        return

    resultats = []
    for numero_essai, etiquettes in enumerate(essais_a_faire, start=1):
        print(
            f"\n[{numero_essai}/{len(essais_a_faire)}] M = {etiquettes['valence']}, "
            f"répétition {etiquettes['repetition']}..."
        )
        resultat = realiser_essai(etiquettes)
        if resultat is None:
            print("Échec répété : vérifiez le micro et le volume, puis relancez cette condition.")
            break
        nom_fichier, performances = resultat
        resultats.append((etiquettes, performances))
        print(
            f"    BER {performances['ber']:.2%} | caractères corrects "
            f"{performances['taux_caracteres_corrects']:.0%} | texte {performances['texte_decode']!r}"
        )
        time.sleep(PAUSE_ENTRE_ESSAIS)

    print(f"\n--- Bilan : {decrire_condition(condition)} ---")
    for etiquettes, performances in resultats:
        print(
            f"  M = {etiquettes['valence']} rép. {etiquettes['repetition']} : BER {performances['ber']:.2%}, "
            f"débit effectif {performances['debit_effectif_bits_par_seconde']:.1f} bit/s"
        )


def lancer_campagne():
    DOSSIER_CAMPAGNE.mkdir(exist_ok=True)
    print("Périphériques audio utilisés :")
    print(f"  sortie : {sd.query_devices(kind='output')['name']}")
    print(f"  entrée : {sd.query_devices(kind='input')['name']}")
    appareils = demander_texte_libre("Nom du setup (ex : PC-Asus_vers_micro-Blue)")
    while True:
        afficher_menu(appareils)
        choix = demander_choix("Option", (*CONDITIONS, "q"))
        if choix == "q":
            return
        lancer_condition(CONDITIONS[choix], appareils)


if __name__ == "__main__":
    lancer_campagne()
