import numpy as np
import pytest
import torch

from geovision.evaluate import classification_metrics, expected_calibration_error
from geovision.explain import GradCAM
from geovision.models import build_model, param_groups, set_backbone_trainable


@pytest.mark.parametrize("name", ["baseline_cnn", "efficientnet_b0", "resnet50"])
def test_forward_shape(name):
    model = build_model(name, 10, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10)


def test_freeze_backbone_keeps_head_trainable():
    model = build_model("efficientnet_b0", 10, pretrained=False)
    set_backbone_trainable(model, "efficientnet_b0", False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("classifier") for n in trainable)


def test_param_groups_cover_all_params():
    model = build_model("resnet50", 10, pretrained=False)
    groups = param_groups(model, "resnet50", 1e-3, 0.1)
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    assert groups[0]["lr"] == pytest.approx(1e-4)


def test_gradcam_output():
    model = build_model("baseline_cnn", 3, pretrained=False)
    cam, idx = GradCAM(model, "baseline_cnn")(torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 and 0 <= idx < 3


def test_metrics_perfect_predictions():
    labels = np.array([0, 1, 2, 1])
    probs = np.eye(3)[labels]
    m = classification_metrics(probs, labels)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
