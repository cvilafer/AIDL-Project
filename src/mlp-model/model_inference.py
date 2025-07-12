"""
Frame-level sign classification with precomputed features + Keras MLP (2 fases) + generator MLP1
"""

import os
import glob
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import strftime
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
PROCESSED_TRAIN_FOLDER = r"C:\UPC IA\Media Pipe\videos_train\mp1000_mirror_proc"
PROCESSED_VAL_FOLDER = r"C:\UPC IA\Media Pipe\videos_val\mp1000_mirror_proc"
PROCESSED_TEST_FOLDER=r"C:\UPC IA\Media Pipe\videos_test\mp1000_proc"

csv_filtro_signos = r"C:\\UPC IA\\Media Pipe\\MSASL_classes_10.csv"

SAVED_MODEL_FOLDER = r"C:\UPC IA\Media Pipe\saved_models"
os.makedirs(SAVED_MODEL_FOLDER, exist_ok=True)

# Definir carpeta para guardar las imágenes y para CSVs originales
output_img_folder = r"C:\\UPC IA\\Media Pipe\\test_frames_img"
video_root_folder = r"C:\\UPC IA\\Media Pipe\\videos_test"
original_csv_folder = r"C:\UPC IA\Media Pipe\videos_test\mp1000"
os.makedirs(output_img_folder, exist_ok=True)

# ------------------------------------------------------------------
# FUNCIONES
# ------------------------------------------------------------------
def detectar_mejor_frame(X, y, modelo, encoder):
    if not len(X):
        return None, None, None, None
    probas = modelo.predict(X)
    label_real = str(y[0])
    class_index = encoder.transform([label_real])[0]
    probas_clase_real = probas[:, class_index]
    idx = np.argmax(probas_clase_real)
    prob = probas_clase_real[idx]
    clase_predicha = np.argmax(probas[idx])
    signo_predicho = str(encoder.inverse_transform([clase_predicha])[0])
    return signo_predicho, idx, prob, X[idx]

def generar_datos_mlp2(X, y, mejor_frame_input, mejor_frame_idx):
    Xi_mlp2 = []
    for i in range(X.shape[0]):
        if i < mejor_frame_idx:
            Xi_mlp2.append(np.hstack([X[i], mejor_frame_input]))
        else:
            Xi_mlp2.append(np.hstack([mejor_frame_input, X[i]]))
    return np.array(Xi_mlp2), y

print(f"Hora actual: {strftime('%H:%M:%S')}")

# ------------------------------------------------------------------
# PREPARAR DATOS
# ------------------------------------------------------------------
# Leer filtro de signos permitidos desde CSV
df_filtro = pd.read_csv(csv_filtro_signos)
signos_permitidos = set(df_filtro.iloc[:, 0].astype(str).str.lower().str.strip())

def filtrar_archivos_por_signo(archivos_csv, signos_validos):
    archivos_filtrados = []
    for f in archivos_csv:
        base = os.path.basename(f)
        if '-' not in base:
            continue
        # Obtener la parte después del primer guion
        parte_signo = base.split('-', 1)[1].lower().strip()

        # Eliminar la extensión final: .csv o _x.csv
        if parte_signo.endswith('_x.csv'):
            signo = parte_signo[:-6]  # quitar "_x.csv"
        elif parte_signo.endswith('.csv'):
            signo = parte_signo[:-4]  # quitar ".csv"
        else:
            continue  # ignorar formatos inesperados

        if signo in signos_validos:
            archivos_filtrados.append(f)
    return archivos_filtrados

# Obtener listas de archivos
train_files = glob.glob(os.path.join(PROCESSED_TRAIN_FOLDER, "*.csv"))
val_files = glob.glob(os.path.join(PROCESSED_VAL_FOLDER, "*.csv"))
csv_test_files = glob.glob(os.path.join(PROCESSED_TEST_FOLDER, "*.csv"))

# Aplicar filtro
train_files = filtrar_archivos_por_signo(train_files, signos_permitidos)
val_files = filtrar_archivos_por_signo(val_files, signos_permitidos)
csv_test_files = filtrar_archivos_por_signo(csv_test_files, signos_permitidos)

# Fit LabelEncoder y StandardScaler con muestras
all_labels = []
sample_X = []
train_X_blocks = []
train_y_blocks = []

for f in tqdm(train_files, desc="Train: Cargando y preparando datos"):
    df = pd.read_csv(f)
    if df.empty:
        continue
    X = df.drop(columns=["label"]).values
    y = df["label"].astype(str).values

    all_labels.append(y)
    if len(sample_X) < 100:
        sample_X.append(X)

    train_X_blocks.append(X)
    train_y_blocks.append(y)

# Ajustar encoder y scaler
all_labels_flat = np.concatenate(all_labels)
le = LabelEncoder().fit(all_labels_flat)
sample_X = np.vstack(sample_X)
scaler = StandardScaler().fit(sample_X)

# Aplicar escalado y codificación
train_X_blocks = [scaler.transform(X) for X in train_X_blocks]
train_y_blocks = [le.transform(y) for y in train_y_blocks]

print(f"Hora actual: {strftime('%H:%M:%S')}")

# Cargar validación (en memoria)
val_X_list = []
val_y_list = []

for f in tqdm(val_files, desc="Val: Cargando validación en memoria"):
    df = pd.read_csv(f)
    val_X_list.append(df.drop(columns=["label"]).values)
    val_y_list.append(df["label"].values)

val_X = scaler.transform(np.vstack(val_X_list))
val_y = le.transform(np.concatenate(val_y_list).astype(str))

print(f"Hora actual: {strftime('%H:%M:%S')}")

print(f"Hora actual: {strftime('%H:%M:%S')}")

# ------------------------------------------------------------------
# MLP1
# ------------------------------------------------------------------
input_dim = sample_X.shape[1]

model1 = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax')
])

model1.compile(
    optimizer=keras.optimizers.Adam(0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

batch_size = 200
total_train_samples = sum(x.shape[0] for x in train_y_blocks)
steps_per_epoch = total_train_samples // batch_size

print("Iniciando entrenamiento MLP1...")
print("batch_size: ",batch_size)
print("total_train_samples: ",total_train_samples)
print("steps_per_epoch: ",steps_per_epoch)

# -------------------------
# MLP1 Dataset (memoria)
# -------------------------
X_train_mlp1 = np.concatenate([x.astype(np.float32) for x in train_X_blocks], axis=0)
y_train_mlp1 = np.concatenate(train_y_blocks, axis=0)

history1 = model1.fit(
    X_train_mlp1, y_train_mlp1,
    batch_size=batch_size,
    epochs=300,
    validation_data=(val_X, val_y),
    callbacks=callbacks,
    verbose=2
)

model1.save(os.path.join(SAVED_MODEL_FOLDER, "mlp1_model.h5"))

# ------------------------------------------------------------------
# MLP2 DATOS (en memoria, sin archivos temporales)
# ------------------------------------------------------------------
X2_train_list = []
y2_train_list = []

for csv_file in tqdm(train_files, desc="Generando datos MLP2 train"):
    df = pd.read_csv(csv_file)
    X = scaler.transform(df.drop(columns=["label"]).values)
    y = df["label"].astype(str).values

    signo_detectado, mejor_idx, prob, mejor_input = detectar_mejor_frame(X, y, model1, le)
    if mejor_input is not None:
        Xi2, yi2 = generar_datos_mlp2(X, y, mejor_input, mejor_idx)
        X2_train_list.append(Xi2.astype(np.float32))
        y2_train_list.append(le.transform(yi2))

X_train_mlp2 = np.concatenate(X2_train_list, axis=0)
y_train_mlp2 = np.concatenate(y2_train_list, axis=0)

# Validación en memoria
X2_val_list = []
y2_val_list = []

for csv_file in tqdm(val_files, desc="Generando datos MLP2 val"):
    df = pd.read_csv(csv_file)
    X = scaler.transform(df.drop(columns=["label"]).values)
    y = df["label"].astype(str).values

    signo_detectado, mejor_idx, prob, mejor_input = detectar_mejor_frame(X, y, model1, le)
    if mejor_input is not None:
        Xi2, yi2 = generar_datos_mlp2(X, y, mejor_input, mejor_idx)
        X2_val_list.append(Xi2.astype(np.float32))
        y2_val_list.append(le.transform(yi2))

X_val_mlp2 = np.concatenate(X2_val_list, axis=0) if X2_val_list else None
y_val_mlp2 = np.concatenate(y2_val_list, axis=0) if y2_val_list else None

print(f"Hora actual: {strftime('%H:%M:%S')}")

import numpy as np
import os
import math
import random
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------
# MLP2 MODELO
# ------------------------------------------------------------------
input_dim = X_train_mlp2.shape[1]

model2 = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax')
])

model2.compile(
    optimizer=keras.optimizers.Adam(0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------------------------------
# CARGAR DATOS EN MEMORIA
# ------------------------------------------------------------------
history2 = model2.fit(
    X_train_mlp2, y_train_mlp2,
    batch_size=batch_size,
    epochs=300,
    validation_data=(X_val_mlp2, y_val_mlp2) if X_val_mlp2 is not None else None,
    callbacks=callbacks,
    verbose=2
)

# ------------------------------------------------------------------
# GUARDAR MODELO
# ------------------------------------------------------------------
model2.save(os.path.join(SAVED_MODEL_FOLDER, "mlp2_model.h5"))

import pickle
with open(os.path.join(SAVED_MODEL_FOLDER, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(SAVED_MODEL_FOLDER, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# Guardar gráfico de pérdidas de MLP1
plt.figure()
plt.plot(history1.history["loss"], label="MLP1 Train Loss")
plt.plot(history1.history["val_loss"], label="MLP1 Val Loss")
plt.title("MLP1 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, "mlp1_loss.png"))
plt.close()

# Guardar gráfico de pérdidas de MLP2
plt.figure()
plt.plot(history2.history["loss"], label="MLP2 Train Loss")
plt.plot(history2.history["val_loss"], label="MLP2 Val Loss")
plt.title("MLP2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, "mlp2_loss.png"))
plt.close()

# Guardar gráfico de accuracy de MLP1
plt.figure()
plt.plot(history1.history["accuracy"], label="MLP1 Train Acc")
plt.plot(history1.history["val_accuracy"], label="MLP1 Val Acc")
plt.title("MLP1 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, "mlp1_accuracy.png"))
plt.close()

# Guardar gráfico de accuracy de MLP2
plt.figure()
plt.plot(history2.history["accuracy"], label="MLP2 Train Acc")
plt.plot(history2.history["val_accuracy"], label="MLP2 Val Acc")
plt.title("MLP2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, "mlp2_accuracy.png"))
plt.close()

def detectar_mejor_frame_para_signo(csv_path, modelo, encoder, scaler):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None, None, None
    X = scaler.transform(df.drop(columns=["label"]).values)
    y = df["label"].values
    probas = modelo.predict(X)
    label_real = str(y[0])
    class_index = encoder.transform([label_real])[0]
    probas_clase_real = probas[:, class_index]
    idx = np.argmax(probas_clase_real)
    prob = probas_clase_real[idx]
    clase_predicha = np.argmax(probas[idx])
    signo_predicho = str(encoder.inverse_transform([clase_predicha])[0])
    return signo_predicho, idx, prob, X[idx]


# ------------------------------------------------------------------
#  Evaluar sobre test y guardar imágenes
# ------------------------------------------------------------------

import cv2

# Función para dibujar los landmarks en los frames desde CSV original
def draw_frame_with_landmarks_from_csv(csv_proc_file, frame_idx, output_folder,
                                        etiqueta_real, signo_detectado, prob_real,
                                        video_root_folder, mlp, original_csv_folder):
    base_csv_name = os.path.splitext(os.path.basename(csv_proc_file))[0]
    base_csv_name = base_csv_name.replace("_proc", "")

    if '-' not in base_csv_name:
        print(f"[!] Nombre de CSV {base_csv_name} no tiene formato esperado 'id-label'")
        return

    subfolder = base_csv_name
    video_name_base = base_csv_name.split('-')[1]
    video_name = f"{video_name_base}.mp4"
    video_path = os.path.join(video_root_folder, subfolder, video_name)
    csv_original_path = os.path.join(original_csv_folder, base_csv_name + ".csv")

    if not os.path.exists(video_path):
        print(f"[!] Video no encontrado: {video_path}")
        return

    if not os.path.exists(csv_original_path):
        print(f"[!] CSV original no encontrado: {csv_original_path}")
        return

    df = pd.read_csv(csv_original_path)
    if frame_idx >= len(df):
        print(f"[!] Frame index {frame_idx} fuera de rango en CSV original.")
        return

    row = df.iloc[frame_idx]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] No se pudo abrir el video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[!] No se pudo leer el frame {frame_idx} del video {video_path}")
        return

    h, w, _ = frame.shape

    for j in range(33):  # pose
        x = row.get(f"pose_lm{j}_x", np.nan)
        y = row.get(f"pose_lm{j}_y", np.nan)
        if not np.isnan(x) and not np.isnan(y):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    for j in range(21):  # left hand
        x = row.get(f"left_hand_lm{j}_x", np.nan)
        y = row.get(f"left_hand_lm{j}_y", np.nan)
        if not np.isnan(x) and not np.isnan(y):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    for j in range(21):  # right hand
        x = row.get(f"right_hand_lm{j}_x", np.nan)
        y = row.get(f"right_hand_lm{j}_y", np.nan)
        if not np.isnan(x) and not np.isnan(y):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    file_id = base_csv_name.split('-')[0]
    prob_txt = f"p{prob_real:.2f}"
    filename = f"{file_id}-{etiqueta_real}-{signo_detectado}-fr{frame_idx}-{prob_txt}{mlp}.png"
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, frame)
    print(f"[img] Guardada: {out_path}")

aciertos = 0
total = 0
resultados = []
aciertos_mlp1=0

for csv_file in tqdm(csv_test_files, desc="Clasificando test"):
    try:
        etiqueta_real = str(pd.read_csv(csv_file, nrows=1)["label"].values[0]) # para evitar problema tipos etiqueta numérica
    except Exception as e:
        print(f"⚠️ Error leyendo etiqueta de {csv_file}: {e}")
        continue

    # MLP1: encontrar mejor frame
    signo_detectado_1, mejor_idx_1, prob_real_1, mejor_input = detectar_mejor_frame_para_signo(csv_file, model1, le, scaler)
    if mejor_input is None:
        continue

    signo_detectado_1=str(signo_detectado_1)

    # Guardar imagen del mejor frame MLP1
    draw_frame_with_landmarks_from_csv(
        csv_file, mejor_idx_1, output_img_folder,
        etiqueta_real, signo_detectado_1, prob_real_1,
        video_root_folder, 'MLP1', original_csv_folder
    )

    # MLP2: generar datos y clasificar
    df = pd.read_csv(csv_file)
    X = scaler.transform(df.drop(columns=["label"]).values)
    y = df["label"].astype(str).values
    Xi2, yi2 = generar_datos_mlp2(X, y, mejor_input, mejor_idx_1)
    if not len(Xi2):
        continue

    probas2 = model2.predict(Xi2)
    probas_clase_real_2 = probas2[:, le.transform([str(etiqueta_real)])[0]]
    idx_best_2 = np.argmax(probas_clase_real_2)
    prob_best_2 = probas_clase_real_2[idx_best_2]
    pred_best_2 = np.argmax(probas2[idx_best_2])
    signo_detectado_2 = str(le.inverse_transform([pred_best_2])[0])

    # Guardar imagen del mejor frame MLP2
    draw_frame_with_landmarks_from_csv(
        csv_file, idx_best_2, output_img_folder,
        etiqueta_real, signo_detectado_2, prob_best_2,
        video_root_folder, 'MLP2', original_csv_folder
    )

    # Elegir el resultado con mayor probabilidad entre MLP1 y MLP2
    if prob_real_1 >= prob_best_2:
        signo_final = signo_detectado_1
        prob_final = prob_real_1
        mlp_final = "MLP1"
    else:
        signo_final = signo_detectado_2
        prob_final = prob_best_2
        mlp_final = "MLP2"

    resultados.append((os.path.basename(csv_file), etiqueta_real, signo_final, mlp_final, prob_final))

    if signo_detectado_1 == etiqueta_real:
        aciertos_mlp1 += 1
    

    if signo_final == etiqueta_real:
        aciertos += 1
    total += 1


# ------------------------------------------------------------------
#  Mostrar resultados
# ------------------------------------------------------------------
print("\n=== RESUMEN TEST FINAL ===")
accuracy_mlp1 = (aciertos_mlp1 / total) * 100 if total > 0 else 0
print(f"\n Precisión total test (MLP1 solo): {aciertos_mlp1}/{total} = {accuracy_mlp1:.2f}%")
accuracy = (aciertos / total) * 100 if total > 0 else 0
print(f"\n Precisión total test (MLP1+MLP2): {aciertos}/{total} = {accuracy:.2f}%")

for fname, real, pred, mlp, prob in resultados:
    print(f"{fname:<40} -> {pred:<15} ({mlp}, p={prob:.2f}) (real: {real}) {'correct' if real == pred else 'incorrect'}")


# MATRIZ DE CONFUSION
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Recolectar etiquetas reales y predichas
etiquetas_reales = [r for _, r, _, _, _ in resultados]
etiquetas_predichas = [p for _, _, p, _, _ in resultados]

# Asegurarse de que todas las etiquetas estén codificadas con el encoder
labels_texto = sorted(set(etiquetas_reales + etiquetas_predichas))
labels_codificadas = le.transform(labels_texto)
labels_texto_ordenadas = le.inverse_transform(labels_codificadas)

#labels_texto_ordenadas = le.classes_


# Crear matriz de confusión
cm = confusion_matrix(etiquetas_reales, etiquetas_predichas, labels=labels_texto_ordenadas)

# Ajustar tamaño de figura automáticamente
num_clases = len(labels_texto_ordenadas)
lado = max(10, int(num_clases * 0.14))  # escalar dinámicamente
figsize = (lado, lado)


fig, ax = plt.subplots(figsize=figsize)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_texto_ordenadas)
disp.plot(include_values=False, cmap='Blues', ax=ax, xticks_rotation=90)
plt.title("Matriz de Confusión (MLP1+MLP2)")
plt.tight_layout()

output_conf_path = os.path.join(output_img_folder, "matriz_confusion.png")
plt.savefig(output_conf_path, dpi=300)
plt.close()

print(f"Matriz de confusión guardada en: {output_conf_path}")