"""Differentiable, dimensionless QM9 physical residuals after inverse scaling."""

import torch
from torch import nn

from molgap.data.qm9_properties import PROPERTY_UNITS

TASKS7 = tuple(PROPERTY_UNITS)
TASKS10 = TASKS7 + ("u0", "u298", "h298")
UNITS10 = dict(PROPERTY_UNITS, u0="hartree", u298="hartree", h298="hartree")


class QM9PhysicsLoss(nn.Module):
    def __init__(self, scalers, settings):
        super().__init__()
        self.settings = dict(settings)
        self.register_buffer(
            "means", torch.tensor([scalers[t]["mean"] for t in TASKS10], dtype=torch.float64)
        )
        self.register_buffer(
            "scales", torch.tensor([scalers[t]["scale"] for t in TASKS10], dtype=torch.float64)
        )
        if (
            not torch.isfinite(self.means).all()
            or not torch.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("Finite training means and positive training scales are required")
        names = {name: i for i, name in enumerate(TASKS10)}
        self.ix = names
        s = self.scales
        normalizers = torch.stack(
            [
                (s[names["delta_e"]] ** 2 + s[names["homo"]] ** 2 + s[names["lumo"]] ** 2).sqrt(),
                (s[names["h298"]] ** 2 + s[names["u298"]] ** 2).sqrt(),
                (s[names["u0"]] ** 2 + s[names["u298"]] ** 2).sqrt(),
            ]
        ).clamp_min(settings["scale_floor_hartree"])
        self.register_buffer("normalizers", normalizers)

    def weight(self, epoch):
        # Epoch is one-based in logs and in the public loss interface.
        warm, ramp = self.settings["warmup_epochs"], self.settings["ramp_epochs"]
        return self.settings["weight"] * min(1.0, max(0.0, (epoch - warm) / max(1, ramp)))

    def forward(self, prediction, target, epoch):
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 10:
            raise ValueError("PhysicsThermo10 requires aligned [batch, 10] tensors")
        supervised = (prediction - target).square().mean()
        y = prediction.double() * self.scales + self.means
        p = {t: y[:, j] for t, j in self.ix.items()}
        rg = p["delta_e"] - (p["lumo"] - p["homo"])
        rh = p["h298"] - p["u298"] - self.settings["kbt_hartree"]
        vt = (p["u0"] - p["u298"]).clamp_min(0)
        violations = torch.stack(
            [
                (rg.abs() - self.settings["gap_tolerance_hartree"]).clamp_min(0),
                (rh.abs() - self.settings["enthalpy_tolerance_hartree"]).clamp_min(0),
                vt,
            ],
            dim=-1,
        )
        physical = (violations / self.normalizers).square().mean(0)
        total = supervised + self.weight(epoch) * physical.mean()
        with torch.no_grad():
            diagnostics = dict(
                physical_weight=self.weight(epoch),
                gap_residual_mae=float(rg.abs().mean()),
                enthalpy_residual_mae=float(rh.abs().mean()),
                thermal_violation_rate=float((vt > 0).double().mean()),
            )
            for name, residual in [("gap", rg.abs()), ("enthalpy", rh.abs()), ("thermal", vt)]:
                diagnostics[f"{name}_residual_p95"] = float(torch.quantile(residual, 0.95))
        return (
            total,
            dict(
                supervised=supervised,
                orbital_gap=physical[0],
                enthalpy_internal=physical[1],
                thermal_nonnegative=physical[2],
            ),
            diagnostics,
        )
