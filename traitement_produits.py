"""Script d'association entre produits Shopify et fournisseurs.

Ce fichier lit trois CSV contenant les produits Shopify, les fiches
fournisseur et l'analyse textuelle des fiches. Les données sont fusionnées
puis comparées afin de trouver la meilleure correspondance pour chaque
produit Shopify. Un aperçu est ensuite sauvegardé dans ``P3_PREVIEW.csv``.

L'appel à l'API OpenAI nécessite la variable d'environnement
``OPENAI_API_KEY`` contenant votre clé personnelle.
"""

import logging
import os
import re
from typing import Dict

import pandas as pd
from rapidfuzz import fuzz

# Configuration du logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Fichiers utilisés par le script
FICHIER_P1 = "P1_DUPLICATED_WITH_TITLES.csv"
FICHIER_FOURNISSEURS = "resultats_touslesliens.csv"
FICHIER_ANALYSE = "ANALYSE_FOURNISSEURS.csv"
FICHIER_PREVIEW = "P3_PREVIEW.csv"


def gpt_match_and_score(shopify_title: str, supplier_info: Dict[str, str]):
    """Calcule un score de similarité entre le titre Shopify et le modèle fournisseur."""
    supplier_model = supplier_info.get("Modèle", "")
    score = fuzz.token_set_ratio(shopify_title.lower(), supplier_model.lower())
    return score / 100, "Score basé sur la similarité du modèle"


def gpt_analyze_image(image_url: str) -> str:
    """Effectue une analyse visuelle à l'aide de l'API OpenAI si disponible."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY non défini, analyse visuelle simulée.")
        return "Analyse visuelle simulée."
    try:
        import openai  # type: ignore
    except Exception as exc:  # pragma: no cover - dépendance externe
        logging.error("Impossible d'importer openai: %s", exc)
        return "Analyse visuelle indisponible"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Voici une image d’échappement. Dis-moi ce que tu observes : "
                        "ligne complète, clapets visibles, downpipe présent, etc."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    try:
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(model="gpt-4o", messages=messages)
        else:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - appel réseau
        logging.error("Erreur lors de l'appel OpenAI: %s", exc)
        return "Analyse visuelle indisponible"


def normalize_text(text: str) -> str:
    """Nettoie et normalise un texte pour faciliter les comparaisons."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_analysis_string(analysis_str: str) -> Dict[str, str]:
    """Extrait les informations structurées contenues dans une chaîne multi-lignes."""
    parsed_data: Dict[str, str] = {}
    if pd.notna(analysis_str):
        for line in analysis_str.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_data[key.strip()] = value.strip()
    return parsed_data


def traiter_produits():
    logging.info("📂 Chargement des fichiers…")
    try:
        p1 = pd.read_csv(FICHIER_P1)
        fournisseurs = pd.read_csv(FICHIER_FOURNISSEURS)
        analyse_fournisseurs = pd.read_csv(FICHIER_ANALYSE)
    except FileNotFoundError as err:
        logging.error("❌ Erreur fatale: %s", err)
        return None

    # Harmonisation des noms de colonnes
    fournisseurs = fournisseurs.rename(
        columns={
            "url": "Supplier_ID",
            "titre": "Supplier_Title",
            "description": "Supplier_Description",
            "prix": "Supplier_Price",
            "image_1": "Supplier_Image_URL",
        }
    )
    analyse_fournisseurs = analyse_fournisseurs.rename(columns={"url": "Supplier_ID"})

    parsed_analysis = analyse_fournisseurs["analyse"].apply(parse_analysis_string)
    for key in ["Marque", "Modèle", "Génération", "Type", "Matériau", "Clapets"]:
        analyse_fournisseurs[key] = parsed_analysis.apply(lambda x: x.get(key, ""))

    merged_df = pd.merge(fournisseurs, analyse_fournisseurs, on="Supplier_ID", how="left")

    final_matches = []

    for _, shopify_row in p1.iterrows():
        shopify_id = shopify_row["Shopify_ID"]
        shopify_title = shopify_row["Title"]
        normalized_shopify_title = normalize_text(shopify_title)

        best_match = None
        best_score = -1.0
        best_analysis = ""

        for _, supplier_row in merged_df.iterrows():
            extracted_material = supplier_row.get("Matériau", "")
            extracted_type = supplier_row.get("Type", "")
            extracted_model = supplier_row.get("Modèle", "")

            material_valid = extracted_material and normalize_text(extracted_material) in normalized_shopify_title
            type_valid = extracted_type and normalize_text(extracted_type) in normalized_shopify_title

            if (extracted_material and not material_valid) or (
                extracted_type and not type_valid
            ):
                continue

            model_match = False
            if extracted_model:
                norm_model = normalize_text(extracted_model)
                if any(part in normalized_shopify_title for part in norm_model.split()):
                    model_match = True
            if extracted_model and not model_match:
                continue

            score, _ = gpt_match_and_score(shopify_title, supplier_row)
            if score > best_score:
                best_score = score
                best_match = supplier_row
                best_analysis = gpt_analyze_image(supplier_row["Supplier_Image_URL"])

        if best_match is not None:
            final_matches.append(
                {
                    "Shopify_ID": shopify_id,
                    "Shopify_Title": shopify_title,
                    "Supplier_ID": best_match["Supplier_ID"],
                    "Supplier_Title": best_match["Supplier_Title"],
                    "Supplier_Description": best_match["Supplier_Description"],
                    "Supplier_Price": best_match["Supplier_Price"],
                    "Supplier_Image_URL": best_match["Supplier_Image_URL"],
                    "Extracted_Marque": best_match.get("Marque", ""),
                    "Extracted_Modèle": best_match.get("Modèle", ""),
                    "Extracted_Génération": best_match.get("Génération", ""),
                    "Extracted_Type": best_match.get("Type", ""),
                    "Extracted_Matériau": best_match.get("Matériau", ""),
                    "Extracted_Clapets": best_match.get("Clapets", ""),
                    "Match_Score": best_score,
                    "Visual_Analysis": best_analysis,
                }
            )

    final_df = pd.DataFrame(final_matches)
    final_df.to_csv(FICHIER_PREVIEW, index=False)
    logging.info("✅ Fichier %s généré avec succès.", FICHIER_PREVIEW)
    return final_df


if __name__ == "__main__":
    resultat = traiter_produits()
    if resultat is not None:
        logging.info("🎉 Traitement terminé avec succès.")
    else:
        logging.error("❌ Échec du traitement.")
