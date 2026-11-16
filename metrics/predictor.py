"""Whether the predictor beats copying: its error against the last context embedding's.

Chunks of ``history + 1`` consecutive frames are slid over each held-out window. The
predictor reads the first ``history`` frames with their actions, and at every position
its output is scored against the next frame's embedding. The copy baseline outputs the
embedding at that position instead. Squared errors are averaged over positions and
dimensions, as in the validation ``pred_loss`` and ``copy_loss`` of the training loop.
"""

import torch


@torch.inference_mode()
def measure(model, lat, batch_size=512):
    """Mean squared error of the predictor and of copying, on the held-out windows.

    ``ratio`` is pred_mse / copy_mse; below 1 the predictor beats copying.
    """
    device = next(model.parameters()).device
    history = model.predictor.pos_embedding.shape[1]
    held = ~lat.train
    z, action, valid = lat.z[held], lat.action[held], lat.valid[held]

    start = torch.arange(z.shape[1] - history)[:, None]
    frames = start + torch.arange(history + 1)  # (S, history + 1)
    # windows are padded by repeating the last frame, which would make copying exact
    ok = valid[:, history:].reshape(-1)
    z = z[:, frames].flatten(0, 1)[ok]  # (M, history + 1, D)
    action = action[:, frames[:, :history]].flatten(0, 1)[ok]

    se = {"pred": 0.0, "copy": 0.0}
    for i in range(0, len(z), batch_size):
        zb = z[i : i + batch_size].to(device)
        ab = action[i : i + batch_size].to(device)
        pred = model.predict(zb[:, :history], model.action_encoder(ab))
        target = zb[:, 1:]
        se["pred"] += (pred - target).double().pow(2).sum().item()
        se["copy"] += (zb[:, :history] - target).double().pow(2).sum().item()
    n = z[:, 1:].numel()
    pred_mse, copy_mse = se["pred"] / n, se["copy"] / n
    return {
        "pred_mse": pred_mse,
        "copy_mse": copy_mse,
        "ratio": pred_mse / copy_mse,
        "n_chunks": len(z),
        "history": history,
    }
