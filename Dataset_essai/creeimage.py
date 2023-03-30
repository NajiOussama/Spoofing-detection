import os

dataset_dir = '/chemin/vers/Dataset5Go'
image_list_file = 'images.txt'

# Ouvrir le fichier pour écrire la liste d'images
with open(image_list_file, 'w') as f:
  
  # Boucler sur les répertoires 'Real' et 'Spoof' dans les répertoires 'Train', 'Val' et 'Test'
  for split in ['Train', 'Val', 'Test']:
    for label in ['Real', 'Spoof']:
      
      # Récupérer le chemin vers le répertoire contenant les images
      image_dir = os.path.join(dataset_dir, split, label)
      
      # Boucler sur les images et écrire leur chemin absolu dans le fichier
      for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        f.write(image_path + '\n')