"""
Preprocesamiento de CSVs de MediaPipe y guardado en CSVs procesados
"""

import os
import glob
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

# CONFIGURACIÓN
INPUT_FOLDER = r"C:\UPC IA\Media Pipe\videos_test\mp17"  # Cambiar según el conjunto (train/val/test)
OUTPUT_PROCESSED_FOLDER = r"C:\UPC IA\Media Pipe\videos_test\mp17_proc"

os.makedirs(OUTPUT_PROCESSED_FOLDER, exist_ok=True)

def angle_between_vectors(v1, v2):
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(v1, v2)
    angle_rad = math.atan2(cross_norm, dot)
    return abs(math.degrees(angle_rad))

def compute_finger_angles(hand_coords):
    finger_defs = {
        "Thumb":  [(4,3,2), (3,2,0)],
        "Index":  [(8,6,5), (6,5,0)],
        "Middle": [(12,10,9), (10,9,0)],
        "Ring":   [(16,14,13), (14,13,0)],
        "Pinky":  [(20,18,17), (18,17,0)]
    }
    angles = []
    for defs in finger_defs.values():
        for (tip, pip, mcp) in defs:
            pa, pb, pc = hand_coords[tip], hand_coords[pip], hand_coords[mcp]
            angles.append(angle_between_vectors(pb - pa, pc - pb))
    return angles

def compute_neighbor_angles(hand_coords):
    neighbor_pairs = [
        ((4,2),(8,5)),
        ((8,5),(12,9)),
        ((12,9),(16,13)),
        ((16,13),(20,17))
    ]
    angles = []
    for (tip1,base1), (tip2,base2) in neighbor_pairs:
        vec1 = hand_coords[tip1] - hand_coords[base1]
        vec2 = hand_coords[tip2] - hand_coords[base2]
        angles.append(angle_between_vectors(vec1, vec2))
    return angles

def normalize_point(point, left_shoulder, shoulder_vec_norm):
    return (point - left_shoulder) / shoulder_vec_norm

def get_palm_normal(hand_coords):
    wrist, index_base, pinky_base = hand_coords[0], hand_coords[5], hand_coords[17]
    normal = np.cross(index_base - wrist, pinky_base - wrist)
    norm = np.linalg.norm(normal)
    return np.zeros(3) if norm < 1e-6 else normal / norm

def angle_with_projection(normal, plane_axes=('y','z'), ref_axis='z'):
    idx = {'x':0,'y':1,'z':2}
    keep = [idx[a] for a in plane_axes]
    ref = np.zeros(3); ref[idx[ref_axis]] = 1.0
    proj = np.zeros(3); proj[keep[0]], proj[keep[1]] = normal[keep[0]], normal[keep[1]]
    proj_norm = np.linalg.norm(proj)
    return 0.0 if proj_norm < 1e-6 else angle_between_vectors(proj/proj_norm, ref)

pose_upper_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,23,24]

def process_csv_and_save(csv_path, output_folder):
    df = pd.read_csv(csv_path)
    X_rows = []
    labels = []

    prev_pose = None
    prev_left = None
    prev_right = None
    prev_time = None

    for _, row in df.iterrows():
        timestamp = row.get("timestamp_ms", 0.0)
        pose_xyz = np.array([[row.get(f"pose_lm{j}_{c}", np.nan) for c in 'xyz'] for j in range(33)])
        left_xyz = np.array([[row.get(f"left_hand_lm{j}_{c}", np.nan) for c in 'xyz'] for j in range(21)])
        right_xyz = np.array([[row.get(f"right_hand_lm{j}_{c}", np.nan) for c in 'xyz'] for j in range(21)])

        left_sh, right_sh = pose_xyz[11], pose_xyz[12]
        shoulder_vec = right_sh - left_sh
        shoulder_norm = np.linalg.norm(shoulder_vec)

        if shoulder_norm < 1e-6:
            pose_norm = [0.0] * (len(pose_upper_indices) * 3)
        else:
            pose_norm = []
            for idx in pose_upper_indices:
                p = pose_xyz[idx]
                pose_norm.extend(
                    normalize_point(p, left_sh, shoulder_norm).tolist()
                    if not np.isnan(p).any() else [0.0, 0.0, 0.0]
                )

        def hand_features(coords):
            if np.isnan(coords).any():
                return [0.0] * 22
            feats = compute_finger_angles(coords)
            feats += compute_neighbor_angles(coords)
            feats += normalize_point(coords[0], left_sh, shoulder_norm).tolist()
            feats += normalize_point(coords[8], left_sh, shoulder_norm).tolist()
            palm_norm = get_palm_normal(coords)
            feats += [
                angle_with_projection(palm_norm, ('y', 'z'), 'y'),
                angle_with_projection(palm_norm, ('x', 'z'), 'z')
            ]
            return feats

        left_feats = hand_features(left_xyz)
        right_feats = hand_features(right_xyz)

        if prev_time is not None:
            delta_t = (timestamp - prev_time) / 1000.0
            delta_t = max(delta_t, 1e-6)
            def velocity(curr, prev):
                delta = curr - prev
                v = (delta / delta_t).flatten()
                v[np.isnan(v)] = 0.0
                return v.tolist()
            v_pose = velocity(pose_xyz, prev_pose) if prev_pose is not None else [0.0] * (33 * 3)
            v_left = velocity(left_xyz, prev_left) if prev_left is not None else [0.0] * (21 * 3)
            v_right = velocity(right_xyz, prev_right) if prev_right is not None else [0.0] * (21 * 3)
        else:
            v_pose = [0.0] * (33 * 3)
            v_left = [0.0] * (21 * 3)
            v_right = [0.0] * (21 * 3)

        prev_pose = pose_xyz
        prev_left = left_xyz
        prev_right = right_xyz
        prev_time = timestamp

        features = left_feats + right_feats + pose_norm + v_pose + v_left + v_right
        X_rows.append(features)
        labels.append(row["label"])

    # Guardar a CSV procesado
    df_out = pd.DataFrame(X_rows)
    df_out["label"] = labels
    output_path = os.path.join(output_folder, os.path.basename(csv_path))
    df_out.to_csv(output_path, index=False)
    print(f"💾 Procesado y guardado: {output_path}")

# PROCESAR TODOS LOS CSV
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
for csv_file in tqdm(csv_files, desc="Procesando CSVs"):
    process_csv_and_save(csv_file, OUTPUT_PROCESSED_FOLDER)