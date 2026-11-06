"""Small shared-trunk outcome predictor for frozen-controller branch labels."""
import torch
from torch import nn


class ConnectionPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, inputs):
        return self.network(inputs)


def fit_predictor(inputs, labels, times, validate, *, seed, epochs=100, patience=20, brier_tiebreak=False):
    """One equal-root full batch; caller supplies observed validation outcomes."""
    torch.manual_seed(seed)
    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    t = torch.tensor(times, dtype=torch.float32)
    success = y.bool()
    mean, scale = x.mean(0), x.std(0, correction=0)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    time_mean = t[success].mean() if success.any() else torch.tensor(0.)
    time_scale = t[success].std(correction=0).clamp_min(.02) if success.any() else torch.tensor(1.)
    model = ConnectionPredictor(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best_key, best, stale, history = None, None, 0, []
    for epoch in range(1, epochs + 1):
        prediction = model((x - mean) / scale)
        classification = nn.functional.binary_cross_entropy_with_logits(prediction[:, 0], y)
        regression = nn.functional.huber_loss(prediction[success, 1], (t[success] - time_mean) / time_scale) if success.any() else prediction[:, 1].sum() * 0.
        optimizer.zero_grad()
        (classification + regression).backward()
        optimizer.step()
        score = validate(model, mean, scale, time_mean, time_scale)
        key = (-score['complete'], score['time_s'] if score['complete'] else float('inf'))
        if brier_tiebreak:
            key += (score['brier'],)
        history.append(dict(epoch=epoch, classification_loss=float(classification.detach()),
                            time_loss=float(regression.detach()), validation=score))
        if best_key is None or key < best_key:
            best_key, stale = key, 0
            best = dict(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
                        input_mean=mean, input_scale=scale, time_mean=time_mean,
                        time_scale=time_scale, epoch=epoch, validation=score)
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best['state_dict'])
    return model, best, history
