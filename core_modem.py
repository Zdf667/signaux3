import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate, spectrogram

FREQUENCE_ECHANTILLONNAGE = 44100
DUREE_SYMBOLE = 0.05
FREQUENCE_MIN = 1000.0
FREQUENCE_MAX = 4000.0
DUREE_PREAMBULE = 0.5
DUREE_GARDE = 0.05
AMPLITUDE_EMISSION = 0.8
FRACTION_ANALYSEE_PAR_SYMBOLE = 0.4
NIVEAU_MINIMAL_CONSEILLE = 0.05
COEFFICIENT_TRAINEE_MAXIMAL = 0.9
BITS_PAR_CARACTERE = 8
VALENCES_AUTORISEES = (2, 4, 8, 16)
TEXTE_REFERENCE = "Signaux3_Projet_MFSK"
VALENCE_EN_TETE = 2
BITS_EN_TETE_VALENCE = 2
BITS_EN_TETE_LONGUEUR = 8
REPETITIONS_EN_TETE = 3
LONGUEUR_MAXIMALE_MESSAGE = 2**BITS_EN_TETE_LONGUEUR - 1
SEUIL_DETECTION_PREAMBULE = 0.1
DESACCORDS_MAXIMAUX_EN_TETE = 2

ECHANTILLONS_PAR_SYMBOLE = round(DUREE_SYMBOLE * FREQUENCE_ECHANTILLONNAGE)
ECHANTILLONS_PREAMBULE = round(DUREE_PREAMBULE * FREQUENCE_ECHANTILLONNAGE)
ECHANTILLONS_GARDE = round(DUREE_GARDE * FREQUENCE_ECHANTILLONNAGE)
ECHANTILLONS_ANALYSES_PAR_SYMBOLE = round(FRACTION_ANALYSEE_PAR_SYMBOLE * ECHANTILLONS_PAR_SYMBOLE)


def verifier_valence(valence):
    if valence not in VALENCES_AUTORISEES:
        raise ValueError(f"Valence {valence} invalide, valeurs possibles : {VALENCES_AUTORISEES}")


def bits_par_symbole(valence):
    verifier_valence(valence)
    return valence.bit_length() - 1


def frequences_porteuses(valence):
    verifier_valence(valence)
    return np.linspace(FREQUENCE_MIN, FREQUENCE_MAX, valence)


def generer_preambule():
    instants = np.arange(ECHANTILLONS_PREAMBULE) / FREQUENCE_ECHANTILLONNAGE
    return chirp(instants, f0=FREQUENCE_MIN, t1=DUREE_PREAMBULE, f1=FREQUENCE_MAX, method="linear")


def texte_vers_bits(texte):
    if not texte:
        raise ValueError("Le texte à transmettre est vide")
    if not texte.isascii():
        raise ValueError("Le texte à transmettre doit être en ASCII")
    octets = np.frombuffer(texte.encode("ascii"), dtype=np.uint8)
    return np.unpackbits(octets)


def bits_vers_texte(bits):
    nombre_octets_complets = len(bits) // BITS_PAR_CARACTERE
    bits_complets = np.asarray(bits[: nombre_octets_complets * BITS_PAR_CARACTERE], dtype=np.uint8)
    return np.packbits(bits_complets).tobytes().decode("latin-1")


def nombre_symboles(nombre_bits, valence):
    return -(-nombre_bits // bits_par_symbole(valence))


def bits_vers_symboles(bits, valence):
    taille_symbole = bits_par_symbole(valence)
    nombre_bits_bourrage = nombre_symboles(len(bits), valence) * taille_symbole - len(bits)
    bits_complets = np.concatenate((bits, np.zeros(nombre_bits_bourrage, dtype=np.uint8)))
    groupes_de_bits = bits_complets.reshape(-1, taille_symbole)
    poids_binaires = 1 << np.arange(taille_symbole - 1, -1, -1)
    return groupes_de_bits @ poids_binaires


def symboles_vers_bits(symboles, valence):
    taille_symbole = bits_par_symbole(valence)
    decalages = np.arange(taille_symbole - 1, -1, -1)
    return ((np.asarray(symboles)[:, None] >> decalages) & 1).astype(np.uint8).ravel()


def entier_vers_bits(valeur, nombre_bits):
    decalages = np.arange(nombre_bits - 1, -1, -1)
    return ((valeur >> decalages) & 1).astype(np.uint8)


def bits_vers_entier(bits):
    return int(bits @ (1 << np.arange(len(bits) - 1, -1, -1)))


def sequence_apprentissage(valence):
    return np.append(np.arange(valence), 0)


def frequences_du_bloc(symboles, valence):
    symboles_avec_apprentissage = np.concatenate((sequence_apprentissage(valence), symboles))
    return frequences_porteuses(valence)[symboles_avec_apprentissage]


def moduler_frequences(frequences_par_symbole):
    frequences_instantanees = np.repeat(frequences_par_symbole, ECHANTILLONS_PAR_SYMBOLE)
    phase_continue = 2 * np.pi * np.cumsum(frequences_instantanees) / FREQUENCE_ECHANTILLONNAGE
    return np.sin(phase_continue)


def assembler_trame(frequences_par_symbole):
    signal_emission = np.concatenate(
        (generer_preambule(), np.zeros(ECHANTILLONS_GARDE), moduler_frequences(frequences_par_symbole))
    )
    return AMPLITUDE_EMISSION * signal_emission


def generer_signal_emission(texte, valence):
    symboles = bits_vers_symboles(texte_vers_bits(texte), valence)
    return assembler_trame(frequences_du_bloc(symboles, valence))


def generer_bits_en_tete(valence, nombre_caracteres):
    if not 1 <= nombre_caracteres <= LONGUEUR_MAXIMALE_MESSAGE:
        raise ValueError(f"Le message doit faire entre 1 et {LONGUEUR_MAXIMALE_MESSAGE} caractères")
    bits_en_tete = np.concatenate(
        (
            entier_vers_bits(VALENCES_AUTORISEES.index(valence), BITS_EN_TETE_VALENCE),
            entier_vers_bits(nombre_caracteres, BITS_EN_TETE_LONGUEUR),
        )
    )
    return np.tile(bits_en_tete, REPETITIONS_EN_TETE)


def nombre_symboles_en_tete():
    return nombre_symboles((BITS_EN_TETE_VALENCE + BITS_EN_TETE_LONGUEUR) * REPETITIONS_EN_TETE, VALENCE_EN_TETE)


def generer_signal_message(texte, valence):
    symboles_en_tete = bits_vers_symboles(generer_bits_en_tete(valence, len(texte)), VALENCE_EN_TETE)
    symboles_message = bits_vers_symboles(texte_vers_bits(texte), valence)
    frequences_trame = np.concatenate(
        (
            frequences_du_bloc(symboles_en_tete, VALENCE_EN_TETE),
            frequences_du_bloc(symboles_message, valence),
        )
    )
    return assembler_trame(frequences_trame)


def duree_donnees(nombre_caracteres, valence):
    return nombre_symboles(nombre_caracteres * BITS_PAR_CARACTERE, valence) * DUREE_SYMBOLE


def duree_emission(nombre_caracteres, valence):
    duree_apprentissage = len(sequence_apprentissage(valence)) * DUREE_SYMBOLE
    return DUREE_PREAMBULE + DUREE_GARDE + duree_apprentissage + duree_donnees(nombre_caracteres, valence)


def localiser_preambule(signal_recu):
    preambule = generer_preambule()
    if len(signal_recu) < len(preambule):
        raise ValueError("Signal reçu plus court que le préambule")
    correlation = correlate(signal_recu, preambule, mode="valid", method="fft")
    debut_preambule = int(np.argmax(np.abs(correlation)))
    segment_preambule = signal_recu[debut_preambule : debut_preambule + ECHANTILLONS_PREAMBULE]
    produit_des_normes = np.linalg.norm(segment_preambule) * np.linalg.norm(preambule)
    correlation_normalisee = abs(correlation[debut_preambule]) / produit_des_normes if produit_des_normes > 0 else 0.0
    return debut_preambule, float(correlation_normalisee)


def localiser_debut_donnees(signal_recu):
    debut_preambule, _ = localiser_preambule(signal_recu)
    return debut_preambule + ECHANTILLONS_PREAMBULE + ECHANTILLONS_GARDE


def mesurer_energies(signal_recu, debut, nombre_symboles_attendus, valence):
    nombre_echantillons_attendus = nombre_symboles_attendus * ECHANTILLONS_PAR_SYMBOLE
    segment = signal_recu[debut : debut + nombre_echantillons_attendus]
    segment = np.pad(segment, (0, nombre_echantillons_attendus - len(segment)))
    fenetres_analysees = segment.reshape(-1, ECHANTILLONS_PAR_SYMBOLE)[:, -ECHANTILLONS_ANALYSES_PAR_SYMBOLE:]
    instants = np.arange(ECHANTILLONS_ANALYSES_PAR_SYMBOLE) / FREQUENCE_ECHANTILLONNAGE
    references_complexes = np.exp(-2j * np.pi * np.outer(frequences_porteuses(valence), instants))
    return np.abs(fenetres_analysees @ references_complexes.T)


def estimer_coefficient_trainee(energies_apprentissage, symboles_apprentissage):
    indices_symboles = np.arange(1, len(symboles_apprentissage))
    tons_precedents = symboles_apprentissage[:-1]
    energie_residuelle = energies_apprentissage[indices_symboles, tons_precedents]
    energie_emise = np.maximum(energies_apprentissage[indices_symboles - 1, tons_precedents], np.finfo(float).tiny)
    return float(np.clip(np.median(energie_residuelle / energie_emise), 0.0, COEFFICIENT_TRAINEE_MAXIMAL))


def decoder_bloc(signal_recu, debut, nombre_symboles_attendus, valence):
    symboles_apprentissage = sequence_apprentissage(valence)
    nombre_symboles_apprentissage = len(symboles_apprentissage)
    energies_bloc = mesurer_energies(
        signal_recu, debut, nombre_symboles_apprentissage + nombre_symboles_attendus, valence
    )
    energies_apprentissage = energies_bloc[:nombre_symboles_apprentissage]
    gains_par_porteuse = np.maximum(np.diag(energies_apprentissage[:valence]), np.finfo(float).tiny)
    coefficient_trainee = estimer_coefficient_trainee(energies_apprentissage, symboles_apprentissage)
    energies_sans_trainee = energies_bloc[1:] - coefficient_trainee * energies_bloc[:-1]
    energies_donnees = energies_sans_trainee[nombre_symboles_apprentissage - 1 :]
    symboles_detectes = np.argmax(energies_donnees / gains_par_porteuse, axis=1)
    fin_bloc = debut + (nombre_symboles_apprentissage + nombre_symboles_attendus) * ECHANTILLONS_PAR_SYMBOLE
    return symboles_detectes, fin_bloc


def demoduler_bits(signal_recu, nombre_bits, valence):
    signal_recu = np.asarray(signal_recu, dtype=np.float64).ravel()
    debut_bloc = localiser_debut_donnees(signal_recu)
    symboles_detectes, _ = decoder_bloc(signal_recu, debut_bloc, nombre_symboles(nombre_bits, valence), valence)
    return symboles_vers_bits(symboles_detectes, valence)[:nombre_bits]


def decoder_en_tete(symboles_en_tete):
    copies_en_tete = symboles_vers_bits(symboles_en_tete, VALENCE_EN_TETE).reshape(REPETITIONS_EN_TETE, -1)
    votes_par_bit = copies_en_tete.sum(axis=0)
    nombre_bits_en_desaccord = int(np.count_nonzero((votes_par_bit > 0) & (votes_par_bit < REPETITIONS_EN_TETE)))
    if nombre_bits_en_desaccord > DESACCORDS_MAXIMAUX_EN_TETE:
        raise ValueError(
            f"En-tête illisible : {nombre_bits_en_desaccord} bits incohérents entre les {REPETITIONS_EN_TETE} copies"
        )
    bits_majoritaires = (2 * votes_par_bit > REPETITIONS_EN_TETE).astype(np.uint8)
    valence = VALENCES_AUTORISEES[bits_vers_entier(bits_majoritaires[:BITS_EN_TETE_VALENCE])]
    nombre_caracteres = bits_vers_entier(bits_majoritaires[BITS_EN_TETE_VALENCE:])
    if nombre_caracteres == 0:
        raise ValueError("En-tête illisible : longueur de message nulle")
    return valence, nombre_caracteres


def demoduler_message(signal_recu):
    signal_recu = np.asarray(signal_recu, dtype=np.float64).ravel()
    debut_preambule, correlation_normalisee = localiser_preambule(signal_recu)
    if correlation_normalisee < SEUIL_DETECTION_PREAMBULE:
        raise ValueError(
            f"Aucun préambule détecté (corrélation {correlation_normalisee:.3f} < {SEUIL_DETECTION_PREAMBULE}) : "
            "vérifiez le micro, le volume et le moment de l'émission"
        )
    debut_en_tete = debut_preambule + ECHANTILLONS_PREAMBULE + ECHANTILLONS_GARDE
    symboles_en_tete, debut_message = decoder_bloc(
        signal_recu, debut_en_tete, nombre_symboles_en_tete(), VALENCE_EN_TETE
    )
    valence, nombre_caracteres = decoder_en_tete(symboles_en_tete)
    nombre_bits = nombre_caracteres * BITS_PAR_CARACTERE
    symboles_message, _ = decoder_bloc(signal_recu, debut_message, nombre_symboles(nombre_bits, valence), valence)
    bits_message = symboles_vers_bits(symboles_message, valence)[:nombre_bits]
    return bits_vers_texte(bits_message), valence, correlation_normalisee


def mesurer_performances(texte_emis, bits_decodes, valence):
    bits_emis = texte_vers_bits(texte_emis)
    texte_decode = bits_vers_texte(bits_decodes)
    nombre_bits_faux = int(np.count_nonzero(bits_emis != bits_decodes))
    nombre_caracteres_corrects = sum(
        caractere_emis == caractere_decode for caractere_emis, caractere_decode in zip(texte_emis, texte_decode)
    )
    nombre_bits_corrects = len(bits_emis) - nombre_bits_faux
    return {
        "texte_decode": texte_decode,
        "ber": nombre_bits_faux / len(bits_emis),
        "taux_caracteres_corrects": nombre_caracteres_corrects / len(texte_emis),
        "debit_brut_bits_par_seconde": bits_par_symbole(valence) / DUREE_SYMBOLE,
        "debit_effectif_bits_par_seconde": nombre_bits_corrects / duree_emission(len(texte_emis), valence),
    }


def ajouter_bruit_blanc(signal_canal, snr_db, puissance_reference, generateur_aleatoire):
    puissance_bruit = puissance_reference / 10 ** (snr_db / 10)
    bruit = generateur_aleatoire.normal(0.0, np.sqrt(puissance_bruit), len(signal_canal))
    return signal_canal + bruit


def afficher_spectrogramme(signal_audio, titre="Spectrogramme M-FSK"):
    frequences, instants, densite_spectrale = spectrogram(
        signal_audio, FREQUENCE_ECHANTILLONNAGE, nperseg=2048, noverlap=1024
    )
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(instants, frequences, 10 * np.log10(densite_spectrale + 1e-12), shading="gouraud", cmap="viridis")
    plt.title(titre)
    plt.xlabel("Temps [s]")
    plt.ylabel("Fréquence [Hz]")
    plt.ylim(0, FREQUENCE_MAX + 1000)
    plt.colorbar(label="Intensité [dB]")
    plt.tight_layout()
    plt.show()
