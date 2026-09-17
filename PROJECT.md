# Project Requirements

## 1. Semantic Retrieval

User can enter a natural language query such as:

"new buildings near water"

System returns relevant satellite tiles ranked by relevance.

## 2. Image Similarity

User can select an image and find visually/semantically similar locations.

## 3. Temporal Change

Given a geographic location and date range:

- find available observations
- align observations
- compare them
- identify meaningful changes
- estimate earliest supported change

## 4. False Alarm Suppression

Consider:

- clouds
- haze
- shadows
- seasonal variation
- registration errors
- sensor differences

## 5. Analyst Workflow

Display:

- before image
- after image
- change map
- confidence
- date
- location
- sensor
- provenance

## 6. Offline

Everything required for inference must run locally.

