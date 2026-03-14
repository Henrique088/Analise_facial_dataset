# scripts/02_mapear_classes_final.py

import json
from pathlib import Path

# Mapeamento universal DAS 7 EMOÇÕES que usaremos
EMOCOES_UNIVERSAIS = {
    0: "Raiva",
    1: "Nojo",
    2: "Medo",
    3: "Felicidade",
    4: "Tristeza",
    5: "Surpresa",
    6: "Neutro"
}

# MAPEAMENTO DETALHADO PARA CADA DATASET
MAPEAMENTOS = {
    "affectnet": {
        # AffectNet: 0=Anger,1=Contempt,2=Disgust,3=Fear,4=Happy,5=Neutral,6=Sad,7=Surprise
        "0": 0,  # Anger → Raiva
        "1": 1,  # Contempt → Nojo (ou poderia ser Neutro? Decidi Nojo)
        "2": 1,  # Disgust → Nojo
        "3": 2,  # Fear → Medo
        "4": 3,  # Happy → Felicidade
        "5": 6,  # Neutral → Neutro
        "6": 4,  # Sad → Tristeza
        "7": 5,  # Surprise → Surpresa
    },
    
    "ckplus": {
        "anger": 0,
        "contempt": 1,  # Mapeando para Nojo
        "disgust": 1,
        "fear": 2,
        "happiness": 3,
        "neutral": 6,
        "sadness": 4,
        "surprise": 5,
    },
    
    "ferplus": {
        # FERPlus: nomes das pastas
        "angry": 0,
        "contempt": 1,  # Mapeando para Nojo
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 6,
        "sad": 4,
        "suprise": 5,  # Nota: está escrito "suprise" no seu path
    },
    
    "kdef": {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 6,
        "sad": 4,
        "surprise": 5,
    },
    
    "rafdb": {
        # RAF-DB: 1=Surprise,2=Fear,3=Disgust,4=Happiness,5=Sadness,6=Anger,7=Neutral
        "1": 5,  # Surprise
        "2": 2,  # Fear
        "3": 1,  # Disgust
        "4": 3,  # Happiness
        "5": 4,  # Sadness
        "6": 0,  # Anger
        "7": 6,  # Neutral
    }
}

# Salvar mapeamentos
output = {
    "classes": EMOCOES_UNIVERSAIS,
    "mapeamentos": MAPEAMENTOS
}

with open("../data/mapeamento_classes.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Mapeamento de classes criado!")
print("📁 Arquivo salvo em: data/mapeamento_classes.json")