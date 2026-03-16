"""
PureCheck — Class Mapping Diagnostic (5-Band Folder Version)
=============================================================
Pass a FOLDER containing all 5 band images for each class.
The script loads all 5 bands (sorted by filename), builds the
correct model inputs, and prints which output index maps to each class.

Usage — pass a folder per class:
    python find_class_mapping.py \
        --g300 "/path/to/G300_folder" \
        --g160 "/path/to/G160_folder" \
        --l8   "/path/to/L8_folder"

You can omit any class you don't have a folder for.
The folder must contain exactly 5 image files (PNG/JPG/TIFF/BMP).
They are sorted alphabetically — band filenames must sort in band order.
"""

import argparse, os, glob, numpy as np

_ROWS = np.linspace(8, 56, 4, dtype=int)
_COLS = np.linspace(8, 56, 5, dtype=int)
SPATIAL_SAMPLES = [(int(r), int(c)) for r in _ROWS for c in _COLS]
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


def load_single_band(filepath):
    from PIL import Image
    arr = np.array(
        Image.open(filepath).convert('L').resize((64, 64), Image.BILINEAR),
        dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def load_folder_as_5bands(folder):
    files = sorted([
        f for f in glob.glob(os.path.join(folder, '*'))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    if len(files) == 0:
        raise FileNotFoundError(f'No image files found in: {folder}')
    if len(files) != 5:
        print(f'  [WARN] Expected 5 band files, found {len(files)} in {folder}.')
        print(f'  Files: {[os.path.basename(f) for f in files]}')
        if len(files) < 5:
            raise ValueError(f'Need exactly 5 band images, only found {len(files)}.')
        print(f'  Using first 5 only.')
        files = files[:5]
    print(f'  Bands loaded from {os.path.basename(folder)}/:')
    bands = []
    for i, f in enumerate(files):
        print(f'    Band {i+1}: {os.path.basename(f)}')
        bands.append(load_single_band(f))
    return np.stack(bands, axis=-1).astype(np.float32)


def build_inputs(bands5, pca_spectral=None, pca_global=None):
    x2d = bands5[:, :, :3][np.newaxis].astype(np.float32)
    work = bands5.copy()
    if pca_global is not None:
        try:
            flat  = bands5.reshape(1, -1)
            recon = pca_global.inverse_transform(
                        pca_global.transform(flat)).reshape(64, 64, 5).astype(np.float32)
            for b in range(5):
                lo, hi = recon[:,:,b].min(), recon[:,:,b].max()
                if hi > lo:
                    recon[:,:,b] = (recon[:,:,b] - lo) / (hi - lo)
            work = recon
        except Exception as e:
            print(f'  [WARN] global PCA skipped: {e}')
    x3d = np.stack([work[:,:,b][:,:,np.newaxis] for b in range(5)],
                   axis=0)[np.newaxis].astype(np.float32)
    feats = []
    for (r, c) in SPATIAL_SAMPLES:
        spec = bands5[r, c, :]
        if pca_spectral is not None:
            try:
                feat = pca_spectral.transform(spec.reshape(1, -1)).flatten()
            except Exception:
                feat = spec[:4]
        else:
            feat = spec[:4]
        feats.append(feat)
    xpca = np.concatenate(feats)[np.newaxis].astype(np.float32)
    return x3d, x2d, xpca


def run(args):
    import tensorflow as tf, pickle, json

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsi_ultimate_model.keras')
    print('Loading model...')
    model = tf.keras.models.load_model(model_path)
    print(f'  inputs : {[list(i.shape) for i in model.inputs]}')
    print(f'  output : {list(model.output_shape)}')

    pca_spectral = pca_global = None
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsi_ultimate_model_pca_models.pkl')
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, dict):
                pca_spectral = d.get('spectral')
                pca_global   = d.get('global')
            else:
                pca_spectral = d
            print('  PCA models loaded.')
        except Exception as e:
            print(f'  [WARN] PCA skipped ({e})')

    samples = [('G300', args.g300), ('G160', args.g160), ('L8', args.l8)]
    samples = [(label, path) for label, path in samples if path]
    if not samples:
        print('ERROR: provide at least one folder with --g300, --g160, or --l8')
        return

    print()
    print('=' * 60)
    mapping = {}
    for label, folder in samples:
        print(f'\nProcessing {label}  ({folder}):')
        try:
            bands5 = load_folder_as_5bands(folder)
            x3d, x2d, xpca = build_inputs(bands5, pca_spectral, pca_global)
            preds  = model.predict([x3d, x2d, xpca], verbose=0)[0]
            idx    = int(np.argmax(preds))
            conf   = float(np.max(preds)) * 100
            scores = [round(float(p) * 100, 1) for p in preds]
            mapping[label] = idx
            print(f'  → model index {idx}  (confidence {conf:.1f}%)')
            print(f'    scores:  index0={scores[0]}%  index1={scores[1]}%  index2={scores[2]}%')
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback; traceback.print_exc()

    if not mapping:
        print('\nNo samples processed successfully.')
        return

    print()
    print('=' * 60)
    print('  YOUR class_mapping.json should be:')
    print('=' * 60)
    reverse = {v: k for k, v in mapping.items()}
    result = {str(i): reverse.get(i, '???') for i in range(3)}

    print('  {')
    for k, v in result.items():
        comma = ',' if k != '2' else ''
        print(f'    "{k}": "{v}"{comma}')
    print('  }')
    print()
    print('  curl command (run while server is running):')
    safe = {k: v for k, v in result.items() if v != '???'}
    print(f"  curl -X POST http://localhost:5001/api/remap -H 'Content-Type: application/json' -d '{json.dumps(safe)}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find correct class index mapping using 5-band folders')
    parser.add_argument('--g300', metavar='FOLDER', help='Folder with 5 band images for a G300 sample')
    parser.add_argument('--g160', metavar='FOLDER', help='Folder with 5 band images for a G160 sample')
    parser.add_argument('--l8',   metavar='FOLDER', help='Folder with 5 band images for a L8 sample')
    run(parser.parse_args())