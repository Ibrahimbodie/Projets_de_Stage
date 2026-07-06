# Schéma du Benchmark

## Fichier : `questions.csv`

| Colonne | Type | Description | Exemple |
|---|---|---|---|
| ID | str | Identifiant unique de la question | Q01 |
| Question | str | Question en langage naturel | Quelle est la note minimale...? |
| Ground_Truth | str | Réponse attendue (vérité terrain) | La note minimale est 5/10 |
| Source_Attendue | str | Document source de référence | Contrat d'apprentissage SIM |
| Niveau_Complexite | str | Niveau de difficulté taxonomique | Niveau 1 : Extraction de fait |
| Type_Reponse | str | Type de réponse attendue | factuel, procédural, conditionnel |
| Variation_Linguistique | str | Type de reformulation | directe, reformulée, contextuelle |
| Categorie_Metier | str | Domaine métier | finance, scolarité, visa, discipline, résidence |
| Justification_Scientifique | str | Motivation académique de la question | Évalue la capacité d'extraction... |

## Niveaux de Complexité
- Niveau 1 : Extraction de fait — information explicite dans un document
- Niveau 2 : Synthèse / Multi-documents — combiner plusieurs sources
- Niveau 3 : Inférence et Raisonnement — déduction logique
- Niveau 4 : Résilience (Hors-domaine) — information absente des documents

## Extension
Pour ajouter une question, ajouter une ligne dans `questions.csv`.
Tous les champs sont obligatoires sauf `Justification_Scientifique`.
