"""
PureCheck — Final Definitive Diagnosis
=======================================
Run this ONCE on your machine. It loads the retrained model and
runs your actual band folders through the EXACT same preprocessing
as retrain.py — then prints the confirmed class mapping.

Usage:
    python final_diagnosis.py \
        --g160 "/path/to/G160_folder" \
        --g300 "/path/to/G300_folder" \
        --l8   "/path/to/L8_folder"

Each folder needs at least 5 sorted band PNG files.
This script will also auto-fix class_mapping.json for you.
"""

import os, sys, json, argparse, glob
import numpy as np
import cv2

IMG_DEPTH  = 5
IMG_HEIGHT = 64
IMG_WIDTH  = 64

def load_one_band(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read: {fpath}')
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.equalizeHist(img)
    return img.astype(np.float32)   # [0,255]


def load_cube_from_folder(folder):
    """
    Load 5 CONSECUTIVE bands from the middle of the spectral range.
    Matches retrain.py sliding window approach — consecutive bands
    from the same hyperspectral cube give the model real spectral
    signatures to work with.
    """
    files = sorted([
        f for f in glob.glob(os.path.join(folder, '*'))
        if f.lower().endswith(('.png','.jpg','.jpeg','.tiff','.bmp'))
    ])
    n = len(files)
    if n == 0:
        raise ValueError(f'No image files found in {folder}')

    # Pad if fewer than 5 files
    while len(files) < IMG_DEPTH:
        files.append(files[-1])
    n = len(files)

    # Take 5 consecutive bands from the middle of the range
    # (same region the sliding window most heavily samples)
    mid = n // 2
    start = max(0, mid - IMG_DEPTH // 2)
    selected = files[start:start + IMG_DEPTH]

    print(f'  Using bands: {[os.path.basename(f) for f in selected]}')
    cube = np.stack([load_one_band(f) for f in selected], axis=0)[..., np.newaxis]
    cube = cube / 255.0
    return cube.astype(np.float32)


def build_inputs(cube, pca_models):
    """Exact copy of retrain.py inference pipeline."""
    from scipy.stats import skew, kurtosis

    D, H, W, C = cube.shape

    pca_spectral    = pca_models['spectral']
    pca_global      = pca_models['global']
    selector        = pca_models['selector']
    scalers         = pca_models['scalers']
    scaler_spectral = scalers['spectral']
    scaler_spatial  = scalers['spatial']
    scaler_global   = scalers['global']
    scaler_final    = scalers['final']

    # 3D input
    input_3d = np.expand_dims(cube, axis=0)   # (1,5,64,64,1)

    # 2D input
    idx = [0, D//3, 2*D//3]
    input_2d = np.expand_dims(
        np.stack([cube[idx[0],:,:,0],
                  cube[idx[1],:,:,0],
                  cube[idx[2],:,:,0]], axis=-1), axis=0)  # (1,64,64,3)

    # Spectral features — matches retrain.py spectral_feat()
    sd = cube.reshape(D, H*W).T             # (HW, D)
    ss = scaler_spectral.transform(sd)
    sp = pca_spectral.transform(ss)
    spectral_feat = np.concatenate([
        sp.mean(0), sp.std(0), sp.max(0), sp.min(0),
        np.percentile(sp,25,0), np.percentile(sp,75,0)
    ])

    # Spatial features — matches retrain.py spatial_feat()
    parts = []
    for band in range(D):
        bd = cube[band,:,:,0].reshape(1,-1)
        parts.append(scaler_spatial.transform(bd))
    spatial_feat = np.concatenate(parts, axis=1)
    spatial_feat = spatial_feat[:, :min(30, spatial_feat.shape[1])]

    # Global features
    flat = cube.reshape(1,-1)
    gf   = pca_global.transform(scaler_global.transform(flat))

    # Statistical features
    s = cube.flatten()
    stat_feat = np.array([[
        np.mean(s), np.std(s), np.median(s),
        np.percentile(s,10), np.percentile(s,25),
        np.percentile(s,75), np.percentile(s,90),
        np.min(s), np.max(s), np.var(s),
        float(skew(s)), float(kurtosis(s)),
        np.sum(s > np.mean(s))   / len(s),
        np.sum(s > np.median(s)) / len(s),
        np.mean(np.abs(s - np.mean(s))),
    ]])

    combined = np.concatenate([
        spectral_feat.reshape(1,-1), spatial_feat, gf, stat_feat
    ], axis=1)
    combined_sel   = selector.transform(combined)
    input_pca      = scaler_final.transform(combined_sel).astype(np.float32)

    return input_3d, input_2d.astype(np.float32), input_pca


def run(args):
    import tensorflow as tf, joblib

    base = os.path.dirname(os.path.abspath(__file__))

    print('Loading model...')
    model = tf.keras.models.load_model(
        os.path.join(base, 'hsi_ultimate_model.keras'))
    print(f'  output shape: {model.output_shape}')

    print('Loading PCA models...')
    pca_models = joblib.load(
        os.path.join(base, 'hsi_ultimate_model_pca_models.pkl'))

    samples = [
        ('G160', args.g160),
        ('G300', args.g300),
        ('L8',   args.l8),
    ]
    samples = [(lbl, path) for lbl, path in samples if path]
    if not samples:
        print('ERROR: provide at least one folder.')
        sys.exit(1)

    print()
    print('=' * 55)
    print('  RESULTS')
    print('=' * 55)

    mapping = {}   # true_label -> model_index
    for true_label, folder in samples:
        print(f'\n{true_label}  ({folder})')
        try:
            cube = load_cube_from_folder(folder)
            i3, i2, ip = build_inputs(cube, pca_models)
            preds  = model.predict([i3, i2, ip], verbose=0)[0]
            scores = [round(float(p)*100, 1) for p in preds]
            idx    = int(np.argmax(preds))
            conf   = float(np.max(preds)) * 100
            mapping[true_label] = idx
            print(f'  model index = {idx}  ({conf:.1f}% confidence)')
            print(f'  all scores  : index0={scores[0]}%  '
                  f'index1={scores[1]}%  index2={scores[2]}%')
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback; traceback.print_exc()

    if len(mapping) < 2:
        print('\nNeed at least 2 classes to build mapping.')
        sys.exit(1)

    # Build correct CLASS_MAP
    # Fill any missing class by exclusion
    all_labels  = {'G160', 'G300', 'L8'}
    all_indices = {0, 1, 2}
    known_idx   = set(mapping.values())
    known_lbl   = set(mapping.keys())

    if len(mapping) == 3:
        class_map = {v: k for k, v in mapping.items()}
    else:
        # Fill the missing label into the missing index
        missing_idx = (all_indices - known_idx).pop() if len(all_indices - known_idx) == 1 else None
        missing_lbl = (all_labels  - known_lbl).pop() if len(all_labels  - known_lbl) == 1 else None
        class_map = {v: k for k, v in mapping.items()}
        if missing_idx is not None and missing_lbl is not None:
            class_map[missing_idx] = missing_lbl
            print(f'\n  (inferred: index {missing_idx} = {missing_lbl})')

    print()
    print('=' * 55)
    print('  CORRECT class_mapping.json:')
    print('=' * 55)
    out = {str(i): class_map.get(i, '???') for i in range(3)}
    print(json.dumps(out, indent=2))

    # Write class_mapping.json automatically
    if '???' not in out.values():
        map_path = os.path.join(base, 'class_mapping.json')
        with open(map_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n✓ Saved automatically to: {map_path}')
        print('\n  Now update app.py CLASS_MAP to:')
        print(f"  CLASS_MAP = {{{', '.join(f'{k}: {repr(v)}' for k,v in class_map.items())}}}")
    else:
        print('\n  Could not infer all 3 — run with all 3 folders.')

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g160', metavar='FOLDER')
    parser.add_argument('--g300', metavar='FOLDER')
    parser.add_argument('--l8',   metavar='FOLDER')
    run(parser.parse_args())