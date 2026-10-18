"""Controles de compatibilidad temporal y ausencia sin usar Test."""
import numpy as np

from data import POSE_DIR, secuencia_a_170, augment_sequence
from model import BPFrontPoseCNN


def test_grid_historica_en_video_completo():
    raw = np.load(POSE_DIR / "bp_001.npy", allow_pickle=False)
    nuevo, mask, centro, escala = secuencia_a_170(raw)
    normalizado = raw.astype(np.float32).copy()
    normalizado[..., :2] = (normalizado[..., :2] - centro) / escala
    old_grid = np.linspace(0, 1, len(raw))
    new_grid = np.linspace(0, 1, 170)
    esperado = np.empty_like(nuevo)
    for k in range(6):
        for c in range(3):
            esperado[:, k, c] = np.interp(new_grid, old_grid,
                                           normalizado[:, k, c])
    assert mask.all() and np.allclose(nuevo, esperado, atol=1e-5)


def test_no_interpola_hueco_y_augmentation_conserva_ausencia():
    raw = np.load(POSE_DIR / "bp_406.npy", allow_pickle=False)
    assert np.isnan(raw[7:16]).all() and len(raw) == 61
    nuevo, mask, _, _ = secuencia_a_170(raw)
    assert (~mask.astype(bool)).any()
    assert np.all(nuevo[~mask.astype(bool)] == 0)
    augmented = augment_sequence(nuevo, mask, np.random.default_rng(42))
    assert np.all(augmented[~mask.astype(bool)] == 0)


def test_embedding_128_y_misma_clasificacion():
    import torch
    model = BPFrontPoseCNN().eval()
    x = torch.zeros((2, 3, 170, 6))
    with torch.no_grad():
        emb = model.extract_embeddings(x)
        p = model(x)
        p_emb = model.classifier(emb)
    assert emb.shape == (2, 128) and p.shape == (2, 1)
    assert torch.allclose(p, p_emb)


if __name__ == "__main__":
    test_grid_historica_en_video_completo()
    test_no_interpola_hueco_y_augmentation_conserva_ausencia()
    test_embedding_128_y_misma_clasificacion()
    print("PoseCNN Common: preprocesamiento y embedding OK")
