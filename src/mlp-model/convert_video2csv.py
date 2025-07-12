import os
import cv2
import mediapipe as mp
import csv

# Extensiones comunes de archivos de video
extensiones_video = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

# Ruta a la carpeta que deseas explorar
carpeta = 'C:\\UPC IA\\Media Pipe\\videos_val'

# Ruta a la carpeta donde dejará los archivos csv mediapipe
carpeta_csv = 'C:\\UPC IA\\Media Pipe\\videos_val\\mp13'
print("Ruta destino:", carpeta_csv)

# Leer lista de nombres válidos desde el CSV
csv_lista = 'C:\\UPC IA\\Media Pipe\\videos_train\\Labels OK Train Order Descending_13 - mp13.csv'
nombres_validos = set()

with open(csv_lista, mode='r', newline='', encoding='utf-8') as archivo_csv:
    lector = csv.DictReader(archivo_csv)
    for fila in lector:
        nombre = fila['Name'].strip()
        if nombre:
            nombres_validos.add(nombre.lower())

# Inicializar Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

# Recorre los archivos de la carpeta y subcarpetas
for carpeta_actual, subcarpetas, archivos in os.walk(carpeta):
    for archivo in archivos:
        if archivo.lower().endswith(extensiones_video):
            nombre_sin_ext = os.path.splitext(archivo)[0].lower()
            if nombre_sin_ext in nombres_validos:
                ruta_completa = os.path.join(carpeta_actual, archivo)

                video_path = ruta_completa
                cap = cv2.VideoCapture(video_path)

                nombre_subcarpeta = os.path.basename(carpeta_actual)
                nombre_csv = nombre_subcarpeta + '.csv'
                csv_path = os.path.join(carpeta_csv, nombre_csv)

                csv_file = open(csv_path, mode='w', newline='')
                csv_writer = csv.writer(csv_file)

                print('*******************************************')
                print('Archivo video:', ruta_completa)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print('# frames/segundo =', fps)
                num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print('# frames video =', num_frames)
                num_segundos = num_frames / fps
                print('# segundos video =', num_segundos)

                # Crear cabecera CSV
                header =['frame', 'timestamp_ms']

                def add_landmark_headers(prefix, num_landmarks):
                    for i in range(num_landmarks):
                        header.extend([f'{prefix}_lm{i}_x', f'{prefix}_lm{i}_y', f'{prefix}_lm{i}_z', f'{prefix}_lm{i}_visibility'])

                add_landmark_headers('pose', 33)
                add_landmark_headers('face', 468)
                add_landmark_headers('left_hand', 21)
                add_landmark_headers('right_hand', 21)

                header.append('label')
                csv_writer.writerow(header)

                frame_idx = 0

                # Listas para registrar todos los valores
                all_x, all_y, all_z, all_vis = [], [], [], []

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)

                    frame_duration_ms = 1000 / fps
                    timestamp_ms = frame_idx * frame_duration_ms
                    row = [frame_idx, timestamp_ms]

                    # Versión que guarda x, y, z, visibilidad en listas
                    def add_landmark_data(landmarks, expected_count):
                        if landmarks:
                            for lm in landmarks.landmark:
                                x, y, z = lm.x, lm.y, lm.z
                                vis = lm.visibility if hasattr(lm, 'visibility') else None
                                row.extend([x, y, z, vis])
                                all_x.append(x)
                                all_y.append(y)
                                all_z.append(z)
                                if vis is not None:
                                    all_vis.append(vis)
                        else:
                            row.extend([None] * expected_count * 4)

                    add_landmark_data(results.pose_landmarks, 33)
                    add_landmark_data(results.face_landmarks, 468)
                    add_landmark_data(results.left_hand_landmarks, 21)
                    add_landmark_data(results.right_hand_landmarks, 21)

                    row.append(nombre_sin_ext)
                    csv_writer.writerow(row)
                    frame_idx += 1

                cap.release()
                csv_file.close()

                # Mostrar estadísticas por video
                print('--- Coordenadas ---')
                if all_x and all_y and all_z:
                    print(f'Mínimo X: {min(all_x):.4f}, Máximo X: {max(all_x):.4f}')
                    print(f'Mínimo Y: {min(all_y):.4f}, Máximo Y: {max(all_y):.4f}')
                    print(f'Mínimo Z: {min(all_z):.4f}, Máximo Z: {max(all_z):.4f}')
                else:
                    print('No se encontraron landmarks con coordenadas.')

                if all_vis:
                    print(f'Mínima visibilidad: {min(all_vis):.4f}, Máxima visibilidad: {max(all_vis):.4f}')
                else:
                    print('No se encontró información de visibilidad.')