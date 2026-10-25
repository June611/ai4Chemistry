import torch

from molgap.qm9_models.runtime import dataloader_kwargs, move_batch


def test_cpu_loader_options_do_not_pin_by_default():
    options = dataloader_kwargs({"num_workers": 0}, device=torch.device("cpu"))
    assert options == {"num_workers": 0, "pin_memory": False}


def test_cuda_loader_enables_overlap_when_workers_exist():
    options = dataloader_kwargs(
        {"num_workers": 4, "prefetch_factor": 3}, device=torch.device("cuda")
    )
    assert options == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 3,
    }


def test_smoke_forces_single_process_loader():
    options = dataloader_kwargs(
        {"num_workers": 4, "prefetch_factor": 3},
        device=torch.device("cuda"),
        smoke=True,
    )
    assert options == {"num_workers": 0, "pin_memory": True}


def test_move_batch_preserves_metadata():
    tensor = torch.tensor([1.0])
    moved = move_batch({"target": tensor, "sample_id": ["x"]}, torch.device("cpu"))
    assert moved["target"] is tensor
    assert moved["sample_id"] == ["x"]
