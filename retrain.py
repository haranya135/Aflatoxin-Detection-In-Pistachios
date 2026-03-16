"""
PureCheck — Retrain on Local PNG Band Data
==========================================
This script retrains the model from scratch using your actual .png band
images, so G300 and L8 are properly distinguishable.

Folder structure expected:
    dataset/
        train/
            G160/   ← all G160 band PNGs
            G300/   ← all G300 band PNGs
            L8/     ← all L8 band PNGs
        test/
            G160/
            G300/
            L8/

Each class folder contains sorted grayscale band images.
A sliding window of 5 consecutive bands forms one training sample.

Usage:
    python retrain.py --data /path/to/dataset --out /path/to/output

Output:
    hsi_ultimate_model.keras
    hsi_ultimate_model_pca_models.pkl
    training_report.png
"""

import os, argparse, gc
import numpy as np
import cv2
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout,
    Concatenate, BatchNormalization, GlobalAveragePooling3D,
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Add,
    Reshape, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# ── Config ────────────────────────────────────────────────────────────────────
IMG_DEPTH  = 5
IMG_HEIGHT = 64
IMG_WIDTH  = 64
PCA_COMPONENTS = 25
LABELS = {'G160': 0, 'G300': 1, 'L8': 2}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_one_band(img_path):
    """Load single band: grayscale -> resize -> equalizeHist -> float32 [0,255]."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.equalizeHist(img)
    return img.astype(np.float32)


def load_data(data_path):
    """
    Correct data loading based on actual filename structure:
        {CLASS}_{TOTAL_BANDS}band_{BAND_INDEX}.png

    Each class folder contains ALL spectral bands of that class type as
    a single hyperspectral cube — not separate samples per subfolder.

    e.g. G300/ has 607 files = 607 spectral bands of one G300 cube
         G160/ has 202 files = 202 spectral bands of one G160 cube
         L8/   has 136 files = 136 spectral bands of one L8 cube

    Strategy: slide a window of IMG_DEPTH=5 consecutive bands across the
    full spectral range with step=1, creating many training samples from
    each class. This gives the model real consecutive spectral signatures.
    """
    X, y = [], []

    for class_label, class_idx in LABELS.items():
        folder = os.path.join(data_path, class_label)
        if not os.path.exists(folder):
            print(f'  [WARN] Folder not found, skipping: {folder}')
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ])
        n = len(files)
        print(f'  {class_label}: {n} spectral bands → ', end='')

        if n < IMG_DEPTH:
            # Pad by repeating last band
            while len(files) < IMG_DEPTH:
                files.append(files[-1])
            n = len(files)

        cubes = []
        # Slide window of 5 consecutive bands across all bands
        step = max(1, IMG_DEPTH // 2)
        for i in range(0, n - IMG_DEPTH + 1, step):
            bands = []
            for j in range(IMG_DEPTH):
                img = load_one_band(os.path.join(folder, files[i + j]))
                if img is not None:
                    bands.append(img)
            if len(bands) == IMG_DEPTH:
                cube_arr = np.stack(bands, axis=0)[..., np.newaxis]
                cubes.append(cube_arr)

        base_count = len(cubes)
        for cube_arr in cubes:
            X.append(cube_arr)
            y.append(class_idx)
            # Augmentation: flip + rotate
            X.append(np.flip(cube_arr, axis=2))
            y.append(class_idx)
            cube_rot = np.array([
                cv2.rotate(cube_arr[k,:,:,0], cv2.ROTATE_90_CLOCKWISE)
                for k in range(IMG_DEPTH)
            ])[..., np.newaxis]
            X.append(cube_rot)
            y.append(class_idx)

        print(f'{base_count} windows → {base_count*3} with augmentation')

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── PCA feature extraction ────────────────────────────────────────────────────
def build_pca_features(X_train, X_test, y_train):
    """
    Extracts spectral + spatial + global + statistical features.
    Fits all scalers/PCA on training data only.
    Returns (X_pca_train, X_pca_test, pca_models_dict, n_features)
    """
    print('  Building PCA features...')
    N_tr, D, H, W, C = X_train.shape
    N_te = X_test.shape[0]

    # ── Spectral PCA ──────────────────────────────────────────────────────────
    subset_size = min(2000, N_tr)
    subset_idx  = np.random.choice(N_tr, subset_size, replace=False)
    subset_data = X_train[subset_idx].reshape(subset_size, D, H * W)
    subset_data = subset_data.transpose(0, 2, 1).reshape(-1, D)  # (N*HW, D)

    scaler_spectral = RobustScaler()
    subset_scaled   = scaler_spectral.fit_transform(subset_data)
    n_spec = min(PCA_COMPONENTS, D - 1)
    pca_spectral = PCA(n_components=n_spec, random_state=42)
    pca_spectral.fit(subset_scaled)
    print(f'    Spectral PCA: {n_spec} components, '
          f'{np.sum(pca_spectral.explained_variance_ratio_):.3f} variance explained')

    def spectral_feat(X):
        N = X.shape[0]
        feats = []
        for i in range(N):
            sd = X[i].reshape(D, H * W).T          # (HW, D)
            ss = scaler_spectral.transform(sd)
            sp = pca_spectral.transform(ss)
            feats.append(np.concatenate([
                sp.mean(0), sp.std(0), sp.max(0), sp.min(0),
                np.percentile(sp, 25, 0), np.percentile(sp, 75, 0)
            ]))
        return np.array(feats)

    sf_train = spectral_feat(X_train)
    sf_test  = spectral_feat(X_test)

    # ── Spatial PCA (one scaler fitted on all bands combined) ─────────────────
    all_band_data = X_train[:, :, :, :, 0].reshape(N_tr * D, H * W)
    scaler_spatial = RobustScaler()
    scaler_spatial.fit(all_band_data)

    def spatial_feat(X):
        N = X.shape[0]
        parts = []
        for band in range(D):
            bd = X[:, band, :, :, 0].reshape(N, -1)
            bs = scaler_spatial.transform(bd)
            parts.append(bs)
        combined = np.concatenate(parts, axis=1)
        return combined[:, :min(30, combined.shape[1])]

    spf_train = spatial_feat(X_train)
    spf_test  = spatial_feat(X_test)

    # ── Global PCA ────────────────────────────────────────────────────────────
    train_flat    = X_train.reshape(N_tr, -1)
    test_flat     = X_test.reshape(N_te, -1)
    scaler_global = RobustScaler()
    train_flat_sc = scaler_global.fit_transform(train_flat)
    test_flat_sc  = scaler_global.transform(test_flat)
    n_glob = min(PCA_COMPONENTS * 2, train_flat_sc.shape[1] - 1)
    pca_global = PCA(n_components=n_glob, random_state=42)
    gf_train = pca_global.fit_transform(train_flat_sc)
    gf_test  = pca_global.transform(test_flat_sc)

    # ── Statistical features ──────────────────────────────────────────────────
    def stat_feat(X):
        feats = []
        for i in range(X.shape[0]):
            s = X[i].flatten()
            feats.append([
                np.mean(s), np.std(s), np.median(s),
                np.percentile(s, 10), np.percentile(s, 25),
                np.percentile(s, 75), np.percentile(s, 90),
                np.min(s), np.max(s), np.var(s),
                float(skew(s)), float(kurtosis(s)),
                np.sum(s > np.mean(s)) / len(s),
                np.sum(s > np.median(s)) / len(s),
                np.mean(np.abs(s - np.mean(s))),
            ])
        return np.array(feats)

    stf_train = stat_feat(X_train)
    stf_test  = stat_feat(X_test)

    # ── Combine → select → scale ──────────────────────────────────────────────
    comb_train = np.concatenate([sf_train, spf_train, gf_train, stf_train], axis=1)
    comb_test  = np.concatenate([sf_test,  spf_test,  gf_test,  stf_test],  axis=1)

    k_best   = min(80, comb_train.shape[1] - 1)
    selector = SelectKBest(f_classif, k=k_best)
    cs_train = selector.fit_transform(comb_train, y_train)
    cs_test  = selector.transform(comb_test)

    scaler_final = RobustScaler()
    cf_train = scaler_final.fit_transform(cs_train)
    cf_test  = scaler_final.transform(cs_test)

    print(f'    Final PCA shape: train={cf_train.shape}, test={cf_test.shape}')

    pca_models = {
        'spectral': pca_spectral,
        'global':   pca_global,
        'selector': selector,
        'scalers': {
            'spectral': scaler_spectral,
            'spatial':  scaler_spatial,
            'global':   scaler_global,
            'final':    scaler_final,
        }
    }
    return cf_train.astype(np.float32), cf_test.astype(np.float32), pca_models, cf_train.shape[1]


# ── 2D projection ─────────────────────────────────────────────────────────────
def make_2d(X):
    D = X.shape[1]
    idx = [0, D // 3, 2 * D // 3]
    return np.stack([X[:, idx[0], :, :, 0],
                     X[:, idx[1], :, :, 0],
                     X[:, idx[2], :, :, 0]], axis=-1).astype(np.float32)


# ── Model architecture (unchanged from original) ──────────────────────────────
def build_resnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same',
               kernel_regularizer=l1_l2(1e-5, 1e-3))(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    for i, filters in enumerate([64, 128, 256, 512]):
        stride = 2 if i > 0 else 1
        shortcut = x
        if stride != 1 or x.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride)(x)
            shortcut = BatchNormalization()(shortcut)
        x = Conv2D(filters // 4, 1, strides=stride,
                   kernel_regularizer=l1_l2(1e-5, 1e-3))(x)
        x = BatchNormalization()(x); x = tf.keras.activations.relu(x)
        x = Conv2D(filters // 4, 3, padding='same',
                   kernel_regularizer=l1_l2(1e-5, 1e-3))(x)
        x = BatchNormalization()(x); x = tf.keras.activations.relu(x)
        x = Conv2D(filters, 1, kernel_regularizer=l1_l2(1e-5, 1e-3))(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x]); x = tf.keras.activations.relu(x)
        gap = GlobalAveragePooling2D()(x)
        att = Dense(filters // 16, activation='relu')(gap)
        att = Dense(filters, activation='sigmoid')(att)
        att = Reshape((1, 1, filters))(att)
        x = tf.keras.layers.multiply([x, att])
        x = Dropout(0.1 * (i + 1))(x)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs, x)


def build_3dcnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, (2, 3, 3), activation='relu', padding='same',
               kernel_regularizer=l1_l2(1e-5, 1e-3))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1, 2, 2))(x)
    sc = x
    x = Conv3D(64, (2, 3, 3), activation='relu', padding='same',
               kernel_regularizer=l1_l2(1e-5, 1e-3))(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, (2, 3, 3), activation='relu', padding='same',
               kernel_regularizer=l1_l2(1e-5, 1e-3))(x)
    x = BatchNormalization()(x)
    if sc.shape[-1] != 64:
        sc = Conv3D(64, 1, padding='same')(sc)
        sc = BatchNormalization()(sc)
    x = Add()([sc, x]); x = tf.keras.activations.relu(x)
    x = MaxPooling3D((1, 2, 2))(x); x = Dropout(0.1)(x)
    x = Conv3D(96, (2, 3, 3), activation='relu', padding='same',
               kernel_regularizer=l1_l2(1e-5, 1e-3))(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling3D()(x); x = Dropout(0.15)(x)
    return Model(inputs, x)


def build_model(pca_features):
    input_3d  = Input(shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1), name='input_3d')
    input_2d  = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3),            name='input_2d')
    input_pca = Input(shape=(pca_features,),                       name='input_pca')

    x3d = build_3dcnn((IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1))(input_3d)
    x3d = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-3))(x3d)
    x3d = BatchNormalization()(x3d); x3d = Dropout(0.3)(x3d)

    x2d = build_resnet((IMG_HEIGHT, IMG_WIDTH, 3))(input_2d)
    x2d = Dense(160, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-3))(x2d)
    x2d = BatchNormalization()(x2d); x2d = Dropout(0.35)(x2d)

    xp = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-3))(input_pca)
    xp = BatchNormalization()(xp); xp = Dropout(0.2)(xp)
    xp = Dense(64,  activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-3))(xp)
    xp = BatchNormalization()(xp); xp = Dropout(0.25)(xp)
    xp = Dense(48,  activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-3))(xp)
    xp = BatchNormalization()(xp); xp = Dropout(0.3)(xp)

    combined = Concatenate()([x3d, x2d, xp])
    cr = Reshape((1, combined.shape[-1]))(combined)
    att = MultiHeadAttention(num_heads=4, key_dim=combined.shape[-1] // 4)(cr, cr)
    att = Flatten()(att)

    f = Dense(192, activation='relu', kernel_regularizer=l1_l2(1e-4, 2e-3))(att)
    f = BatchNormalization()(f); f = Dropout(0.4)(f)
    fr = Dense(192, activation='linear', kernel_regularizer=l1_l2(1e-5, 1e-3))(combined)
    f = Add()([f, fr]); f = tf.keras.activations.relu(f); f = Dropout(0.3)(f)

    f = Dense(96, activation='relu', kernel_regularizer=l1_l2(1e-4, 2e-3))(f)
    f = BatchNormalization()(f); f = Dropout(0.45)(f)
    f = Dense(48, activation='relu', kernel_regularizer=l1_l2(1e-4, 2e-3))(f)
    f = BatchNormalization()(f); f = Dropout(0.4)(f)
    f = Dense(24, activation='relu', kernel_regularizer=l1_l2(1e-4, 1e-3))(f)
    f = BatchNormalization()(f); f = Dropout(0.3)(f)

    out = Dense(3, activation='softmax', name='output')(f)
    return Model(inputs=[input_3d, input_2d, input_pca], outputs=out)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    print('=' * 60)
    print('  PureCheck Retraining')
    print(f'  Data path : {args.data}')
    print(f'  Output    : {args.out}')
    print('=' * 60)

    os.makedirs(args.out, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(args.data, 'train')
    test_path  = os.path.join(args.data, 'test')

    print('\nLoading TRAIN data:')
    X_train_raw, y_train = load_data(train_path)
    print('\nLoading TEST data:')
    X_test_raw,  y_test  = load_data(test_path)

    print(f'\nTrain samples: {len(X_train_raw)}  |  Test samples: {len(X_test_raw)}')
    for lbl, idx in LABELS.items():
        print(f'  {lbl}: train={np.sum(y_train==idx)}  test={np.sum(y_test==idx)}')

    # ── Normalise ─────────────────────────────────────────────────────────────
    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test  = X_test_raw.astype(np.float32)  / 255.0
    # Mild noise for regularisation
    X_train += np.random.normal(0, 0.003, X_train.shape).astype(np.float32)
    X_train  = np.clip(X_train, 0, 1)
    del X_train_raw, X_test_raw; gc.collect()

    # ── PCA features ──────────────────────────────────────────────────────────
    print('\nExtracting PCA features:')
    X_pca_train, X_pca_test, pca_models, n_pca = build_pca_features(
        X_train, X_test, y_train)

    # ── 2D projections ────────────────────────────────────────────────────────
    X_2d_train = make_2d(X_train)
    X_2d_test  = make_2d(X_test)

    # ── Train/val split ───────────────────────────────────────────────────────
    (X3_tr, X3_va, X2_tr, X2_va,
     Xp_tr, Xp_va, y_tr, y_va) = train_test_split(
        X_train, X_2d_train, X_pca_train,
        to_categorical(y_train, 3),
        test_size=0.15, random_state=42,
        stratify=y_train
    )
    print(f'Train: {len(X3_tr)}  Val: {len(X3_va)}')
    del X_train, X_2d_train, X_pca_train; gc.collect()

    # Class weights
    y_tr_lbl = np.argmax(y_tr, axis=1)
    cw = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_tr_lbl), y=y_tr_lbl)
    cw = dict(enumerate(cw))
    print(f'Class weights: {cw}')

    # ── Build & train model ───────────────────────────────────────────────────
    print('\nBuilding model...')
    model = build_model(n_pca)
    print(f'Parameters: {model.count_params():,}')

    best_acc = 0.0
    best_weights = None

    for phase, (lr, bs, ep, pat) in enumerate([
        (1e-4, 12, 60, 12),
        (5e-5, 8,  40, 15),
    ], 1):
        print(f'\n── Phase {phase}  lr={lr}  batch={bs}  epochs={ep} ──')
        model.compile(
            optimizer=Adam(lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        callbacks = [
            EarlyStopping('val_accuracy', patience=pat,
                          restore_best_weights=True, mode='max'),
            ReduceLROnPlateau('val_accuracy', factor=0.7,
                              patience=pat // 2, min_lr=1e-7, mode='max'),
            ModelCheckpoint(os.path.join(args.out, 'best_temp.keras'),
                            monitor='val_accuracy', save_best_only=True, mode='max'),
        ]
        history = model.fit(
            [X3_tr, X2_tr, Xp_tr], y_tr,
            batch_size=bs, epochs=ep,
            validation_data=([X3_va, X2_va, Xp_va], y_va),
            callbacks=callbacks,
            class_weight=cw,
            verbose=1
        )
        _, val_acc = model.evaluate([X3_va, X2_va, Xp_va], y_va, verbose=0)
        print(f'Val accuracy after phase {phase}: {val_acc*100:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.get_weights()

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_test_cat = to_categorical(y_test, 3)
    _, test_acc = model.evaluate(
        [X_test, X_2d_test, X_pca_test], y_test_cat, verbose=0)
    preds = model.predict([X_test, X_2d_test, X_pca_test], verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    class_names = ['G160', 'G300', 'L8']
    print('\n' + '=' * 60)
    print(f'  FINAL TEST ACCURACY: {test_acc*100:.2f}%')
    print('=' * 60)
    print(classification_report(y_test, pred_labels, target_names=class_names))

    cm = confusion_matrix(y_test, pred_labels)
    print('Confusion matrix:')
    print('       ' + '  '.join(f'{n:>6}' for n in class_names))
    for i, n in enumerate(class_names):
        print(f'{n:>6} ' + '  '.join(f'{cm[i,j]:>6d}' for j in range(3)))

    # ── Save model + PCA ──────────────────────────────────────────────────────
    model_out = os.path.join(args.out, 'hsi_ultimate_model.keras')
    pca_out   = os.path.join(args.out, 'hsi_ultimate_model_pca_models.pkl')
    # Remove old files first to avoid PermissionError on locked files
    for path in [model_out, pca_out]:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f'  Removed old: {path}')
        except Exception as e:
            print(f'  [WARN] Could not remove {path}: {e}')
    model.save(model_out)
    joblib.dump(pca_models, pca_out)
    print(f'\n✓ Model saved  → {model_out}')
    print(f'✓ PCA saved    → {pca_out}')
    print(f'\n  class_mapping.json should be:')
    print('  {"0": "G160", "1": "G300", "2": "L8"}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True)

    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = axes[2].imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(class_names)
    axes[2].set_yticks(range(3)); axes[2].set_yticklabels(class_names)
    axes[2].set_title(f'Confusion Matrix ({test_acc*100:.1f}%)')
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f'{cm_n[i,j]:.2f}',
                         ha='center', va='center', fontsize=12,
                         color='white' if cm_n[i,j] > 0.5 else 'black')
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    report_out = os.path.join(args.out, 'training_report.png')
    plt.savefig(report_out, dpi=150)
    print(f'✓ Report saved → {report_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain PureCheck on PNG band data')
    parser.add_argument('--data', required=True,
                        help='Path to dataset folder containing train/ and test/ subfolders')
    parser.add_argument('--out', default='.',
                        help='Output folder for saved model and PCA (default: current dir)')
    main(parser.parse_args())