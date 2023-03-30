import os
import pandas as pd
import numpy as np

# Définir le chemin vers le dossier racine de données
data_path = "/Users/user/Desktop/CentraleSupelec/POLE IA /Implementation-patchnet-main/Dataset_essai"

# Définir les noms des dossiers pour chaque ensemble de données
train_folder = "Train"
val_folder = "Val"
test_folder = "Test"

# Définir la liste des classes
classes = ["Real", "Spoof"]

# Initialiser les listes pour stocker les informations de chaque ensemble de données
train_data = []
val_data = []
test_data = []

# Parcourir chaque classe
for class_name in classes:
    class_path = os.path.join(data_path, train_folder, class_name)
    files = os.listdir(class_path)
    n_files = len(files)

    # Obtenir les index des images à utiliser pour chaque ensemble de données
    idx = np.random.permutation(n_files)
    idx_train = idx[:int(n_files * 0.8)]
    idx_val = idx[int(n_files * 0.8):int(n_files * 0.9)]
    idx_test = idx[int(n_files * 0.9):]

    # Ajouter les informations de chaque image à la liste correspondante
    for i, file in enumerate(files):
        file_path = os.path.join(class_path, file)
        row = {"filename": file_path, "class": class_name}

        if i in idx_train:
            train_data.append(row)
        elif i in idx_val:
            val_data.append(row)
        elif i in idx_test:
            test_data.append(row)

# Convertir les listes en DataFrames et les enregistrer en tant que fichiers CSV
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
val_df.to_csv(os.path.join(data_path, "val.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
