# Refleja los csv's generados con mediapipe holistic (pose,face,left hand, right hand)
# Refleja las x: x=1-x
# Intercambia las coordenadas del bloque left hand por las del right hand

import os
import pandas as pd
from glob import glob

def reflejar_x(csv_path, output_suffix='x'):
    df = pd.read_csv(csv_path)
    columnas = df.columns.tolist()

    # Suponemos que:
    # columnas[0] = frame, columnas[1] = timestamp, columnas[-1] = label
    frame_col = columnas[0]
    time_col = columnas[1]
    label_col = columnas[-1]

    # Coordenadas entre la 3ª y penúltima columna
    coord_cols = columnas[2:-1]

    # Cantidad total de columnas de coordenadas
    total_coord_cols = len(coord_cols)

    # Cada landmark tiene 4 columnas
    # Asumimos:
    # Pose: 33 puntos → 33*4 = 132 columnas
    # Face: 468 puntos → 468*4 = 1872 columnas
    # Hands: 21 puntos cada una → 21*4 = 84 columnas

    pose_cols = coord_cols[0:132]
    face_cols = coord_cols[132:132+1872]
    left_hand_cols = coord_cols[132+1872:132+1872+84]
    right_hand_cols = coord_cols[132+1872+84:132+1872+84+84]

    # Reflejar X en todas las coordenadas
    for i in range(0, len(coord_cols), 4):
        x_col = coord_cols[i]
        df[x_col] = 1 - df[x_col]

    # Intercambiar columnas de mano izquierda y derecha
    for i in range(0, 84, 4):
        l_cols = left_hand_cols[i:i+4]
        r_cols = right_hand_cols[i:i+4]
        df[l_cols], df[r_cols] = df[r_cols].copy(), df[l_cols].copy()

    # Guardar
    dir_path, filename = os.path.split(csv_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{output_suffix}{ext}"
    output_path = os.path.join(dir_path, new_filename)

    df.to_csv(output_path, index=False)
    print(f"✅ Archivo reflejado guardado como: {output_path}")

def reflejar_multiples_csvs(directorio, patron="*.csv"): 
    archivos = glob(os.path.join(directorio, patron))
    for archivo in archivos:
        reflejar_x(archivo)

# Ejemplo de uso
reflejar_multiples_csvs('C:\\UPC IA\\Media Pipe\\videos_train\\mp100_mirror')
