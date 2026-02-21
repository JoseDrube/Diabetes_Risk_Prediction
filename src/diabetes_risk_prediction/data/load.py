import kagglehub
import os
import shutil

def download_and_setup_data():
    # 1. Definir la ruta de destino dentro de tu proyecto
    # Usamos una ruta relativa para que funcione en cualquier carpeta
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(base_dir, "../../data") # Sube a src y luego a data
    
    # Crear la carpeta data si no existe
    os.makedirs(target_folder, exist_ok=True)

    print("Descargando dataset desde Kaggle...")
    # 2. Descargar (esto lo baja a la caché de kagglehub)
    temp_path = kagglehub.dataset_download("vishardmehta/diabetes-risk-prediction-dataset")

    # 3. Mover los archivos de la caché a tu carpeta del proyecto
    print(f"Moviendo archivos de {temp_path} a {target_folder}...")
    
    for filename in os.listdir(temp_path):
        file_source = os.path.join(temp_path, filename)
        file_destination = os.path.join(target_folder, filename)
        
        # Copiar el archivo (usamos copy2 para mantener metadatos)
        shutil.copy2(file_source, file_destination)
        print(f"Archivo listo: {filename}")

    print("\n✅ Dataset importado con éxito a la carpeta /data de tu proyecto.")

if __name__ == "__main__":
    download_and_setup_data()