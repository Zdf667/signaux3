import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOSSIER_PROJET = Path(__file__).resolve().parent
FICHIER_REGISTRE = DOSSIER_PROJET / "donnees_campagne" / "registre_enregistrements.csv"
DOSSIER_FIGURES = DOSSIER_PROJET / "resultats"
RESOLUTION_IMPRESSION_DPI = 300
OBJECTIF_BER = 0.01
CONDITIONS = (
    ("1m", "silence", "1 m, silence", "#257de0"),
    ("2m", "silence", "2 m, silence", "#f47230"),
    ("1m", "musique", "1 m, musique", "#00b894"),
    ("2m", "musique", "2 m, musique", "#f2a600"),
)
COULEUR_TEXTE = "#1f1f1f"
COULEUR_TEXTE_SECONDAIRE = "#6b6b6b"
COULEUR_GRILLE = "#e6e6e6"
LARGEUR_BARRE = 0.19
ECART_BARRES = 0.01


def charger_mesures_par_setup():
    mesures_par_setup = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with FICHIER_REGISTRE.open(newline="", encoding="utf-8") as fichier_csv:
        for ligne in csv.DictReader(fichier_csv):
            cle = (int(ligne["valence"]), ligne["distance"], ligne["bruit_ambiant"])
            mesures_setup = mesures_par_setup[ligne["appareils"]]
            mesures_setup[cle]["ber"].append(float(ligne["ber"]) * 100)
            mesures_setup[cle]["debit"].append(float(ligne["debit_effectif_bits_par_seconde"]))
    return mesures_par_setup


def styliser_axes(axe):
    axe.spines[["top", "right", "left"]].set_visible(False)
    axe.spines["bottom"].set_color(COULEUR_TEXTE_SECONDAIRE)
    axe.tick_params(colors=COULEUR_TEXTE_SECONDAIRE, length=0)
    axe.yaxis.grid(True, color=COULEUR_GRILLE, linewidth=0.8)
    axe.set_axisbelow(True)


def tracer_barres(axe, mesures, valences, grandeur, format_valeur):
    positions_valences = np.arange(len(valences))
    for indice_condition, (distance, bruit, libelle, couleur) in enumerate(CONDITIONS):
        decalage = (indice_condition - (len(CONDITIONS) - 1) / 2) * (LARGEUR_BARRE + ECART_BARRES)
        for indice_valence, valence in enumerate(valences):
            valeurs = mesures[(valence, distance, bruit)][grandeur]
            if not valeurs:
                continue
            abscisse = positions_valences[indice_valence] + decalage
            moyenne = float(np.mean(valeurs))
            axe.bar(
                abscisse,
                moyenne,
                width=LARGEUR_BARRE,
                color=couleur,
                label=libelle if indice_valence == 0 else None,
                zorder=2,
            )
            axe.scatter(
                np.full(len(valeurs), abscisse),
                valeurs,
                s=22,
                color="white",
                edgecolor=COULEUR_TEXTE,
                linewidth=0.9,
                zorder=3,
            )
            axe.annotate(
                format_valeur(moyenne),
                (abscisse, max(max(valeurs), moyenne)),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COULEUR_TEXTE,
            )
    axe.set_xticks(positions_valences, [f"M = {valence}" for valence in valences], fontsize=12, color=COULEUR_TEXTE)


def finaliser_figure(figure, axe, titre, sous_titre, etiquette_ordonnee, nom_fichier):
    axe.set_ylabel(etiquette_ordonnee, fontsize=12, color=COULEUR_TEXTE)
    figure.suptitle(titre, x=0.06, ha="left", fontsize=16, fontweight="bold", color=COULEUR_TEXTE)
    figure.text(0.06, 0.885, sous_titre, ha="left", fontsize=11, color=COULEUR_TEXTE_SECONDAIRE)
    axe.legend(
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, 1.16),
        fontsize=10,
        labelcolor=COULEUR_TEXTE,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    figure.savefig(DOSSIER_FIGURES / nom_fichier, dpi=RESOLUTION_IMPRESSION_DPI)
    plt.close(figure)


def tracer_ber(mesures, valences, setup):
    figure, axe = plt.subplots(figsize=(11, 6.5))
    styliser_axes(axe)
    tracer_barres(axe, mesures, valences, "ber", lambda valeur: f"{valeur:.1f} %")
    ber_maximal = max(max(grandeurs["ber"]) for grandeurs in mesures.values())
    limite_haute = max(5.0, 1.2 * ber_maximal)
    axe.axhline(
        OBJECTIF_BER * 100,
        color=COULEUR_TEXTE,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        zorder=4,
        label="Objectif MVP : BER < 1 %",
    )
    if limite_haute > 50:
        axe.axhline(50, color=COULEUR_TEXTE_SECONDAIRE, linestyle=":", linewidth=1, zorder=1)
        axe.annotate(
            "50 % = décodage au hasard",
            (len(valences) - 0.5, 50),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            color=COULEUR_TEXTE_SECONDAIRE,
        )
    axe.set_ylim(0, limite_haute)
    finaliser_figure(
        figure,
        axe,
        f"Taux d'erreur binaire mesuré selon la valence M (setup {setup})",
        "Barre = moyenne des 3 répétitions, points = chaque répétition. Plus bas = meilleur.",
        "BER (%)",
        f"poster_fig4_BER_reel_par_condition_{setup}.png",
    )


def tracer_debit(mesures, valences, setup):
    figure, axe = plt.subplots(figsize=(11, 6.5))
    styliser_axes(axe)
    tracer_barres(axe, mesures, valences, "debit", lambda valeur: f"{valeur:.0f}")
    for indice_valence, valence in enumerate(valences):
        debit_brut = np.log2(valence) / 0.05
        axe.hlines(
            debit_brut,
            indice_valence - 0.45,
            indice_valence + 0.45,
            color=COULEUR_TEXTE,
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            zorder=4,
        )
        axe.annotate(
            f"débit brut {debit_brut:.0f} bit/s",
            (indice_valence + 0.45, debit_brut),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=COULEUR_TEXTE,
        )
    axe.set_ylim(0, 75)
    finaliser_figure(
        figure,
        axe,
        f"Débit effectif mesuré selon la valence M (setup {setup})",
        "Bits reçus corrects par seconde d'émission. Pointillés = débit théorique sans erreur.",
        "Débit effectif (bit/s)",
        f"poster_fig5_Debit_reel_par_condition_{setup}.png",
    )


def generer_graphiques_campagne():
    if not FICHIER_REGISTRE.exists():
        raise FileNotFoundError(f"{FICHIER_REGISTRE} introuvable : lancez d'abord campagne_automatique.py")
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    for setup, mesures in charger_mesures_par_setup().items():
        valences = sorted({valence for valence, _, _ in mesures})
        tracer_ber(mesures, valences, setup)
        tracer_debit(mesures, valences, setup)
        print(f"Setup {setup} : figures enregistrées dans {DOSSIER_FIGURES}")


if __name__ == "__main__":
    generer_graphiques_campagne()
