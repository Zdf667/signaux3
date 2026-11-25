import numpy as np
import sounddevice as sd

from core_modem import (
    FREQUENCE_ECHANTILLONNAGE,
    LONGUEUR_MAXIMALE_MESSAGE,
    demoduler_message,
    generer_signal_message,
)
from tests_reels_materiel import demander_choix, demander_valence, emettre_signal, verifier_niveau


def demander_message():
    while True:
        message = input(f"Message à envoyer (ASCII, 1 à {LONGUEUR_MAXIMALE_MESSAGE} caractères) : ")
        if not message:
            print("Message vide, recommencez.")
        elif not message.isascii():
            print("Accents et caractères spéciaux interdits : ASCII uniquement.")
        elif len(message) > LONGUEUR_MAXIMALE_MESSAGE:
            print(f"Message trop long ({len(message)} caractères).")
        else:
            return message


def envoyer_message():
    message = demander_message()
    valence = demander_valence()
    signal_message = generer_signal_message(message, valence)
    print(f"[TX] Durée d'émission : {len(signal_message) / FREQUENCE_ECHANTILLONNAGE:.1f} s")
    emettre_signal(signal_message, f"'{message}' (M = {valence})")


def ecouter_jusqu_a_entree():
    blocs_audio = []
    with sd.InputStream(
        samplerate=FREQUENCE_ECHANTILLONNAGE,
        channels=1,
        dtype="float32",
        callback=lambda bloc, _nombre_trames, _temps, _statut: blocs_audio.append(bloc.copy()),
    ):
        input("[RX] Écoute en cours : lancez l'émission, puis appuyez sur Entrée quand elle est terminée.")
    if not blocs_audio:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocs_audio).ravel()


def recevoir_message():
    enregistrement = ecouter_jusqu_a_entree()
    if enregistrement.size == 0:
        print("[RX] Échec : aucun son capté.")
        return
    verifier_niveau(enregistrement)
    try:
        message, valence, correlation_normalisee = demoduler_message(enregistrement)
    except ValueError as erreur:
        print(f"[RX] Échec : {erreur}")
        return
    print(f"[RX] Préambule détecté (corrélation {correlation_normalisee:.2f}), valence annoncée M = {valence}")
    print(f">>> Message reçu : {message!r}")


def lancer_messagerie():
    print("=== MESSAGERIE AUDIO M-FSK ===")
    if demander_choix("Mode", ("envoyer", "recevoir")) == "envoyer":
        envoyer_message()
    else:
        recevoir_message()


if __name__ == "__main__":
    lancer_messagerie()
