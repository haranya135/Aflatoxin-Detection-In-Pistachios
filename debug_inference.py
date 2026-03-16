"""
Run this on your machine to compare final_diagnosis.py vs app.py preprocessing.
It processes the same files both ways and prints the raw model scores.

Usage:
    python debug_inference.py \
        --g160 "/Users/haranyas/Downloads/dataset/train/G160" \
        --g300 "/Users/haranyas/Downloads/dataset/train/G300" \
        --l8   "/Users/haranyas/Downloads/dataset/train/L8"
"""

import os, glob, argparse
import numpy as np
import cv2

IMG_DEPTH  = 5
IMG_HEIGHT = 64
IMG_WIDTH  = 64


def load_band_equalize(fpath):
    """With equalizeHist — as used in final_diagnosis.py"""
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.equalizeHist(img)
    return img.astype(np.float32)          # [0,255]


def load_band_plain(fpath):
    """Without equalizeHist — plain grayscale"""
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img.astype(np.float32)          # [0,255]


def get_middle_5(folder):
    files = sorted([
        f for f in glob.glob(os.path.join(folder, '*'))
        if f.lower().endswith(('.png','.jpg','.jpeg','.tiff','.bmp'))
    ])
    while len(files) < IMG_DEPTH:
        files.append(files[-1])
    n     = len(files)
    mid   = n // 2
    start = max(0, mid - IMG_DEPTH // 2)
    return files[start:start + IMG_DEPTH]


def build_cube(files, loader):
    cube = np.stack([loader(f) for f in files], axis=0)[..., np.newaxis]
    return (cube / 255.0).astype(np.float32)


def build_inputs(cube, pca_models):
    from scipy.stats import skew, kurtosis

    D, H, W, C = cube.shape
    pca_spectral    = pca_models['spectral']
    pca_global      = pca_models['global']
    selector        = pca_models['selector']
    scalers         = pca_models['scalers']
    spectral_scaler = scalers['spectral']
    spatial_scaler  = scalers['spatial']
    global_scaler   = scalers['global']
    final_scaler    = scalers['final']

    # 3D
    input_3d = np.expand_dims(cube, 0)

    # 2D
    idx = [0, D//3, 2*D//3]
    input_2d = np.expand_dims(
        np.stack([cube[idx[0],:,:,0], cube[idx[1],:,:,0], cube[idx[2],:,:,0]], axis=-1), 0)

    # Spectral
    sd = cube.reshape(D, H*W).T
    sp = pca_spectral.transform(spectral_scaler.transform(sd))
    sf = np.concatenate([sp.mean(0), sp.std(0), sp.max(0), sp.min(0),
                         np.percentile(sp,25,0), np.percentile(sp,75,0)])

    # Spatial
    parts = []
    for b in range(D):
        parts.append(spatial_scaler.transform(cube[b,:,:,0].reshape(1,-1)))
    spat = np.concatenate(parts, axis=1)[:, :min(30, D*H*W)]

    # Global
    gf = pca_global.transform(global_scaler.transform(cube.reshape(1,-1)))

    # Stats
    s = cube.flatten()
    stat = np.array([[np.mean(s), np.std(s), np.median(s),
                      np.percentile(s,10), np.percentile(s,25),
                      np.percentile(s,75), np.percentile(s,90),
                      np.min(s), np.max(s), np.var(s),
                      float(skew(s)), float(kurtosis(s)),
                      np.sum(s>np.mean(s))/len(s),
                      np.sum(s>np.median(s))/len(s),
                      np.mean(np.abs(s-np.mean(s)))]])

    comb = np.concatenate([sf.reshape(1,-1), spat, gf, stat], axis=1)
    pca_input = final_scaler.transform(selector.transform(comb)).astype(np.float32)

    return input_3d, input_2d.astype(np.float32), pca_input


def test(model, pca_models, label, folder, loader_fn, loader_name):
    files = get_middle_5(folder)
    cube  = build_cube(files, loader_fn)
    i3, i2, ip = build_inputs(cube, pca_models)
    preds  = model.predict([i3, i2, ip], verbose=0)[0]
    scores = [round(float(p)*100,1) for p in preds]
    idx    = int(np.argmax(preds))
    print(f'  [{loader_name}] {label}: index={idx}  '
          f'scores: 0={scores[0]}%  1={scores[1]}%  2={scores[2]}%')
    return idx, scores


def run(args):
    import tensorflow as tf, joblib

    base = os.path.dirname(os.path.abspath(__file__))
    model      = tf.keras.models.load_model(
        os.path.join(base, 'hsi_ultimate_model.keras'))
    pca_models = joblib.load(
        os.path.join(base, 'hsi_ultimate_model_pca_models.pkl'))

    samples = [('G160', args.g160), ('G300', args.g300), ('L8', args.l8)]
    samples = [(l, p) for l, p in samples if p]

    print('\n=== WITH equalizeHist (final_diagnosis.py method) ===')
    for label, folder in samples:
        test(model, pca_models, label, folder, load_band_equalize, 'equalize')

    print('\n=== WITHOUT equalizeHist (plain grayscale) ===')
    for label, folder in samples:
        test(model, pca_models, label, folder, load_band_plain, 'plain')

    print('\nThe method that gives clearly DIFFERENT indices per class is correct.')
    print('Use that loader in app.py.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g160', metavar='FOLDER')
    parser.add_argument('--g300', metavar='FOLDER')
    parser.add_argument('--l8',   metavar='FOLDER')
    run(parser.parse_args())