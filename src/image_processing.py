import os
import numpy as np
import cv2

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Carga y preprocesa una imagen para la red neuronal, vectoriza y normaliza
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    img = cv2.resize(img, target_size)
    # Asegurar que la imagen esté en el formato correcto para la red neuronal
    img_flat = img.reshape(-1, 1) / 255.0
    return img, img_flat

def load_images_from_folder(folder_path, label, target_size=(128, 128)):
    """
    Carga todas las imágenes de una carpeta y las preprocesa
    """
    data = []
    if not os.path.exists(folder_path):
        raise ValueError(f"La carpeta no existe: {folder_path}")
        
    for filename in os.listdir(folder_path):
        try:
            path = os.path.join(folder_path, filename)
            _, img_flat = load_and_preprocess_image(path, target_size)
            data.append((img_flat, label))
        except Exception as e:
            print(f"Error al cargar {filename}: {str(e)}")
            continue
    return data
