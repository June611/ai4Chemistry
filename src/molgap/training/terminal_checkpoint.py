"""Full epoch-boundary checkpoints with RNG and deterministic-loader recovery."""

import copy
import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Callback


def rng_state(include_cuda=True):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all()
        if include_cuda and torch.cuda.is_initialized()
        else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda"):
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology differs; use the same visible GPU count on resume")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def loader_state(loader):
    state = {}
    if loader.generator is not None:
        state["generator"] = loader.generator.get_state()
    sampler = loader.sampler
    if hasattr(sampler, "rg") and hasattr(sampler, "idxs"):
        state["sampler_rng"] = copy.deepcopy(sampler.rg.bit_generator.state)
        state["sampler_indices"] = sampler.idxs.copy()
    return state


def restore_loader(loader, state):
    if "generator" in state:
        if loader.generator is None:
            loader.generator = torch.Generator()
        loader.generator.set_state(state["generator"].cpu())
    if "sampler_rng" in state:
        if not hasattr(loader.sampler, "rg") or len(loader.sampler.idxs) != len(
            state["sampler_indices"]
        ):
            raise ValueError("Training sampler differs from checkpoint")
        loader.sampler.rg.bit_generator.state = copy.deepcopy(state["sampler_rng"])
        loader.sampler.idxs = state["sampler_indices"].copy()


def require_full_checkpoint(payload):
    if not all(
        payload.get(k) for k in ("state_dict", "optimizer_states", "lr_schedulers", "loops")
    ):
        raise ValueError("Full training checkpoint required; an inference model.pt is insufficient")
    if "rng_snapshot" not in payload or "loader_states" not in payload:
        raise ValueError(
            "Checkpoint lacks new-protocol RNG/loader state; use historical resume tooling"
        )


class TerminalFitCheckpoint(Callback):
    def __init__(self, *, loaders=None, metadata=None, every_n_epochs=50, seed=0):
        super().__init__()
        self.metadata = metadata or {}
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        self.loaders = {}
        self.pending_rng = None
        if loaders:
            self.bind_loaders(loaders)

    def bind_loaders(self, loaders):
        self.loaders = {k: v for k, v in loaders.items() if v is not None}
        for i, loader in enumerate(self.loaders.values()):
            # Worker/base-seed creation must not perturb the model's global RNG.
            if loader.generator is None:
                loader.generator = torch.Generator().manual_seed(self.seed + i)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Legacy 100->200 tooling has no loader/context contract. It retains its
        # documented statistical (rather than exact-RNG) continuation semantics.
        if "rng_snapshot" not in checkpoint and not self.metadata and not self.loaders:
            return
        require_full_checkpoint(checkpoint)
        if checkpoint.get("run_context", {}) != self.metadata:
            raise ValueError("Resume configuration/data/scalers differ from checkpoint")
        if set(checkpoint["loader_states"]) != set(self.loaders):
            raise ValueError("Resume loader splits differ from checkpoint")
        for split, state in checkpoint["loader_states"].items():
            restore_loader(self.loaders[split], state)
        self.pending_rng = checkpoint["rng_snapshot"]
        restore_rng(self.pending_rng)
        # Preserve best/early-stop history when a complete run bundle is relocated.
        for key, state in checkpoint.get("callbacks", {}).items():
            if "ModelCheckpoint" not in key or not isinstance(state, dict):
                continue
            dest = Path(trainer.checkpoint_callback.dirpath)
            if str(state.get("dirpath")) == str(dest):
                continue

            def moved(path, dest=dest):
                if not path:
                    return path
                target = dest / Path(path).name
                if not target.exists():
                    raise FileNotFoundError(f"Resume requires saved best/last assets: {target}")
                return str(target)

            for name in ("best_model_path", "kth_best_model_path", "last_model_path"):
                state[name] = moved(state.get(name, ""))
            state["best_k_models"] = {
                moved(k): v for k, v in state.get("best_k_models", {}).items()
            }
            state["dirpath"] = str(dest)

    def on_train_start(self, trainer, pl_module):
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if self.every_n_epochs and epoch % self.every_n_epochs == 0:
            path = Path(trainer.checkpoint_callback.dirpath) / f"epoch{epoch:04d}.ckpt"
            trainer.save_checkpoint(path)

    def on_train_end(self, trainer, pl_module):
        path = Path(trainer.checkpoint_callback.dirpath) / "terminal_fit.ckpt"
        trainer.save_checkpoint(path)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["rng_snapshot"] = rng_state()
        checkpoint["loader_states"] = {k: loader_state(v) for k, v in self.loaders.items()}
        checkpoint["run_context"] = self.metadata
