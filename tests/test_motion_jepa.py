from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from module import SIGReg
from train import lewm_forward


def test_training_loss_includes_motion_and_backpropagates_through_both_branches():
    with initialize_config_dir(str(Path(__file__).resolve().parents[1] / "config/train"), version_base=None):
        cfg = compose("lewm_motion", overrides=[
            "img_size=28", "model.diff_size=28", "model.action_encoder.input_dim=10",
            "+model.encoder.num_hidden_layers=1", "+model.diff_encoder.num_hidden_layers=1",
            "model.predictor.depth=1",
        ])
    torch.manual_seed(0)
    model = instantiate(cfg.model).train()
    batch = {"pixels": torch.randn(2, 4, 3, 28, 28), "action": torch.randn(2, 4, 10)}
    trainer = SimpleNamespace(model=model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs), log_dict=lambda *a, **k: None)
    output = lewm_forward(trainer, batch, "fit", cfg)
    base = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
    motion = (cfg.loss.motion.diff_pred_weight * output["diff_pred_loss"]
              + cfg.loss.motion.diff_sigreg_weight * output["diff_sigreg_loss"])
    torch.testing.assert_close(output["loss"], base + motion)

    # The difference embedding is a learned target, so its prediction loss must train its encoder too.
    target_grad, = torch.autograd.grad(
        output["diff_pred_loss"], next(model.diff_encoder.parameters()), retain_graph=True, allow_unused=True
    )
    assert target_grad is not None and torch.isfinite(target_grad).all() and target_grad.abs().sum() > 0
    output["loss"].backward()
    for branch in (model.diff_encoder, model.diff_predictor):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
