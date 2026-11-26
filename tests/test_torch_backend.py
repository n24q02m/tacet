"""Tests for PyTorch KGE backend availability and factory fallback."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.slow


@dataclass
class _KGESettingsStub:
    kge_backend: str = "auto"
    kge_dim: int = 64
    kge_epochs: int = 80
    kge_device: str = "cpu"
    kge_model: str = "complex"


class TestTorchBackend(unittest.TestCase):
    def test_torch_backend_import_signals_correctly(self) -> None:
        from tacet.kge.kge_torch import _HAS_TORCH, TorchComplEx, TorchKGEConfig

        if _HAS_TORCH:
            m = TorchComplEx(TorchKGEConfig(dim=8, epochs=2, device="cpu"))
            m.fit([("a", "rel", "b"), ("b", "rel", "c"), ("c", "rel", "d")])
            self.assertIn("a", m.ent)
        else:
            with self.assertRaises(ImportError):
                TorchComplEx(TorchKGEConfig(dim=8, epochs=2, device="cpu"))

    def test_build_kge_from_settings_falls_back_to_numpy(self) -> None:
        from tacet.kge.kge_torch import build_kge_from_settings

        model = build_kge_from_settings(_KGESettingsStub(kge_backend="numpy"))
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict_tail"))


if __name__ == "__main__":
    unittest.main()
