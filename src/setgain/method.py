import math

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from torch import nn
from torch.nn import functional as F

from .network import Decoder, Encoder


POLICIES = (
    "setgain",
    "random",
    "uniform",
    "fewest",
    "disagreement",
    "entropy",
    "diversity_entropy",
    "current_loss",
    "feature_lal",
    "bald",
    "epig",
    "empirical",
    "oracle",
    "no_count",
    "no_disagreement",
    "dice_target",
    "shuffled",
    "no_exploration",
    "no_set",
)


def dice(a, b):
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def pair_distance(masks):
    return (
        float(
            np.mean(
                [
                    1 - dice(masks[i], masks[j])
                    for i in range(len(masks))
                    for j in range(i)
                ]
            )
        )
        if len(masks) > 1
        else 0.0
    )


def energy_score(samples, hidden):
    samples = np.asarray(samples, dtype=np.float32).reshape(len(samples), -1)
    hidden = np.asarray(hidden, dtype=np.float32).reshape(len(hidden), -1)
    if len(samples) < 2 or not len(hidden):
        raise ValueError(
            "Energy score requires at least two samples and one hidden reader"
        )
    first = np.sqrt(((samples[:, None] - hidden[None]) ** 2).mean(-1)).mean()
    distance = np.sqrt(((samples[:, None] - samples[None]) ** 2).mean(-1))
    second = distance.sum() / (len(samples) * (len(samples) - 1))
    return float(first - 0.5 * second)


def scalars(masks):
    values = masks.float()
    areas = values.mean((-2, -1))
    return values.new_tensor(
        [
            float(areas.std(unbiased=False) / areas.mean().clamp_min(1e-8)),
            pair_distance(values.detach().cpu().numpy()),
            float(values.var(0, unbiased=False).mean()),
            float(len(values) < 2),
        ]
    )


class ConditionalModel(nn.Module):
    def __init__(self, latent=8, no_set=False):
        super().__init__()
        self.image_encoder = Encoder()
        self.mask_encoder = Encoder(inputs=2)
        self.contour = nn.Linear(256, 128)
        self.prior = nn.Sequential(
            nn.Linear(513, 128), nn.ReLU(), nn.Linear(128, 2 * latent)
        )
        self.posterior = nn.Sequential(
            nn.Linear(641, 128), nn.ReLU(), nn.Linear(128, 2 * latent)
        )
        self.condition = nn.Linear(257, 32)
        self.decoder = Decoder(32 + latent)
        self.latent = latent
        self.no_set = no_set

    def encode_image(self, image):
        return self.image_encoder(image)

    def embed_masks(self, image, masks):
        if masks.ndim == 3:
            masks = masks[:, None]
        paired = torch.cat([image.expand(len(masks), -1, -1, -1), masks], 1)
        return self.contour(self.mask_encoder(paired)[-1].mean((-2, -1)))

    def statistics(self, embeddings):
        n = len(embeddings)
        state = torch.cat(
            [
                embeddings.mean(0),
                embeddings.square().mean(0),
                embeddings.new_tensor([math.log1p(n)]),
            ]
        )
        return state * 0 if self.no_set else state

    def gaussian(self, features, state, target_embedding=None):
        vector = torch.cat([features[-1].mean((-2, -1))[0], state])
        parameters = (
            self.prior(vector)
            if target_embedding is None
            else self.posterior(torch.cat([vector, target_embedding]))
        )
        mean, logvar = parameters.chunk(2)
        return mean, logvar.clamp(-10, 8)

    def probabilities(self, features, state, normal, posterior=None):
        mean, logvar = posterior or self.gaussian(features, state)
        latent = mean[None] + (0.5 * logvar).exp()[None] * normal
        context = torch.cat([self.condition(state).expand(len(normal), -1), latent], 1)[
            :, :, None, None
        ]
        expanded = [f.expand(len(normal), -1, -1, -1) for f in features]
        return self.decoder(expanded, context).sigmoid()[:, 0]

    def loss(self, image, visible, target, beta=0.001):
        features = self.encode_image(image)
        state = self.statistics(self.embed_masks(image, visible))
        prior = self.gaussian(features, state)
        posterior = self.gaussian(
            features, state, self.embed_masks(image, target[None])[0]
        )
        normal = torch.randn(1, self.latent, device=image.device)
        probabilities = self.probabilities(features, state, normal, posterior)[0]
        nll = F.binary_cross_entropy(probabilities.clamp(1e-6, 1 - 1e-6), target)
        mean_q, variance_q = posterior
        mean_p, variance_p = prior
        kl = (
            0.5
            * (
                variance_p
                - variance_q
                + (variance_q.exp() + (mean_q - mean_p).square()) / variance_p.exp()
                - 1
            ).sum()
        )
        return nll + beta * kl

    def sample(self, features, state, normal, uniforms):
        probabilities = self.probabilities(features, state, normal)
        return (uniforms < probabilities).float().cpu().numpy(), probabilities.mean(
            0
        ).cpu().numpy()

    def gain_features(self, features, state, visible, remaining, variant="setgain"):
        _, logvar = self.gaussian(features, state)
        values = scalars(visible)
        if variant == "no_count":
            state = state.clone()
            state[-1] = 0
            remaining = 0
        if variant == "no_disagreement":
            values = torch.zeros_like(values)
            state = torch.cat([state[:128], state[:128].square(), state[-1:]])
        return torch.cat(
            [
                features[-1].mean((-2, -1))[0],
                state,
                state.new_tensor([remaining]),
                values,
                logvar,
            ]
        )


class GainHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(526, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )


def predictive_metrics(samples, probability, masks, hidden_index, initial_mask=None):
    hidden = masks[hidden_index]
    distances = np.zeros((len(masks), len(masks)))
    for i in range(len(masks)):
        for j in range(i):
            distances[i, j] = distances[j, i] = 1 - dice(masks[i], masks[j])
    clusters = (
        fcluster(
            linkage(squareform(distances, checks=False), method="complete"),
            0.20,
            criterion="distance",
        )
        if len(masks) > 1
        else np.ones(1)
    )
    counts = np.bincount(clusters)
    minority = counts[int(clusters[hidden_index])] < counts.max()
    ece = 0.0
    for low in np.arange(0, 1, 0.1):
        selected = (probability >= low) & (
            (probability < low + 0.1) if low < 0.9 else (probability <= 1)
        )
        if selected.any():
            ece += selected.mean() * abs(
                float(probability[selected].mean() - hidden[selected].mean())
            )
    band_metrics = {}
    if initial_mask is not None:
        band = ndi.binary_dilation(
            initial_mask.astype(bool), iterations=10
        ) ^ ndi.binary_erosion(initial_mask.astype(bool), iterations=10)
        band_metrics["band_brier"] = (
            float(((probability[band] - hidden[band]) ** 2).mean())
            if band.any()
            else None
        )
        band_ece = 0.0
        for index in range(10):
            selected = (
                band
                & (probability >= index / 10)
                & ((probability < (index + 1) / 10) if index < 9 else probability <= 1)
            )
            if selected.any():
                band_ece += (
                    selected.sum()
                    / band.sum()
                    * abs(float(probability[selected].mean() - hidden[selected].mean()))
                )
        band_metrics["band_ece"] = float(band_ece) if band.any() else None
    return {
        **band_metrics,
        "energy_score": energy_score(samples, hidden[None]),
        "brier": float(((probability - hidden) ** 2).mean()),
        "ece": float(ece),
        "minority_mask_coverage": 100 * max(dice(s, hidden) for s in samples)
        if minority
        else None,
        "disagreement_gap": abs(pair_distance(samples) - pair_distance(masks)),
    }
