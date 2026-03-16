"""
PureCheck — Model Input Diagnostic
====================================
This script helps figure out:
1. What band naming/numbering convention your model expects
2. Whether the model is actually working (not collapsed)
3. What a "good" input looks like vs a random one

Run from the food_adulteration folder:
    python diagnose_model.py --folder "/path/to/any_sample_folder"
"""

import os, glob, argparse
import numpy as np

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

_ROWS = np.linspace(8, 56, 4, dtype=int)
_COLS = np.linspace(8, 56, 5, dtype=int)
SPATIAL_SAMPLES = [(int(r), int(c)) for r in _ROWS for c in _COLS]


def load_band(filepath):
    from PIL import Image
    arr = np.array(
        Image.open(filepath).convert('L').resize((64, 64), Image.BILINEAR),
        dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def build_inputs(bands5):
    x2d = bands5[:, :, :3][np.newaxis].astype(np.float32)
    x3d = np.stack([bands5[:,:,b][:,:,np.newaxis] for b in range(5)],
                   axis=0)[np.newaxis].astype(np.float32)
    feats = []
    for (r, c) in SPATIAL_SAMPLES:
        feats.append(bands5[r, c, :4])
    xpca = np.concatenate(feats)[np.newaxis].astype(np.float32)
    return x3d, x2d, xpca


def predict(model, bands5):
    x3d, x2d, xpca = build_inputs(bands5)
    preds = model.predict([x3d, x2d, xpca], verbose=0)[0]
    return [round(float(p)*100, 2) for p in preds]


def run(args):
    import tensorflow as tf

    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, 'hsi_ultimate_model.keras')

    print('Loading model...')
    model = tf.keras.models.load_model(model_path)
    print(f'  inputs: {[list(i.shape) for i in model.inputs]}')
    print(f'  output: {list(model.output_shape)}')

    # ── TEST 1: All-zeros input ───────────────────────────────────────────────
    print('\n[TEST 1] All-zeros input (blank image):')
    zeros = np.zeros((64, 64, 5), dtype=np.float32)
    s = predict(model, zeros)
    print(f'  scores: index0={s[0]}%  index1={s[1]}%  index2={s[2]}%')

    # ── TEST 2: All-ones input ────────────────────────────────────────────────
    print('\n[TEST 2] All-ones input (saturated image):')
    ones = np.ones((64, 64, 5), dtype=np.float32)
    s = predict(model, ones)
    print(f'  scores: index0={s[0]}%  index1={s[1]}%  index2={s[2]}%')

    # ── TEST 3: Random noise ──────────────────────────────────────────────────
    print('\n[TEST 3] Random noise input (3 different seeds):')
    for seed in [42, 99, 7]:
        np.random.seed(seed)
        noise = np.random.rand(64, 64, 5).astype(np.float32)
        s = predict(model, noise)
        print(f'  seed={seed}: index0={s[0]}%  index1={s[1]}%  index2={s[2]}%')

    # ── TEST 4: Real folder if provided ──────────────────────────────────────
    if args.folder:
        files = sorted([
            f for f in glob.glob(os.path.join(args.folder, '*'))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        print(f'\n[TEST 4] Real folder: {args.folder}')
        print(f'  Found {len(files)} image files:')
        for f in files:
            print(f'    {os.path.basename(f)}')

        if len(files) >= 5:
            # Try all consecutive 5-band windows across the files
            print(f'\n  Trying every consecutive 5-band window:')
            best_entropy = -1
            best_window = None
            best_scores = None
            for start in range(len(files) - 4):
                window = files[start:start+5]
                bands5 = np.stack([load_band(f) for f in window], axis=-1)
                s = predict(model, bands5)
                import math
                # Entropy: higher = more uncertain/mixed, lower = more decisive
                entropy = -sum((p/100)*math.log(p/100+1e-9) for p in s)
                names = [os.path.basename(f) for f in window]
                print(f'    bands {start+1}-{start+5}: scores={s}  entropy={entropy:.3f}')
                # We want the LEAST entropy (most decisive/confident prediction)
                if best_entropy < 0 or entropy < best_entropy:
                    best_entropy = entropy
                    best_window = names
                    best_scores = s

            print(f'\n  Most decisive window (lowest entropy):')
            for n in best_window:
                print(f'    {n}')
            print(f'  Scores: {best_scores}')
        else:
            print(f'  Need at least 5 files, found {len(files)}')

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('  INTERPRETATION:')
    print('=' * 60)
    print("""
  If Tests 1-3 ALL give index1=100%:
    → The model output layer is saturated/collapsed. The .keras
      file may be corrupted or saved in an incompatible way.
      Try re-exporting it with: model.save('new_model.keras')

  If Tests 1-3 give DIFFERENT scores:
    → The model is working. Your band selection is just wrong.
      Use Test 4 results to find the right 5 bands.

  If scores change between random seeds in Test 3:
    → Model is alive and responsive to input.
      The issue is purely band selection / class mapping.
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose model input sensitivity')
    parser.add_argument('--folder', metavar='FOLDER',
                        help='Optional: folder of band images to test all 5-band windows')
    run(parser.parse_args())