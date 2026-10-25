import copy
from pathlib import Path

import pytest
import yaml

from molgap.training.protocol import validate_qm9_protocol


def property_config():
    config = yaml.safe_load(
        Path("configs/qm9_six_properties/models/dimenet_pp.yaml").read_text()
    )
    config["training"]["formal_protocol"] = True
    return config


def test_formal_qm9_property_protocol_accepts_frozen_budget():
    validate_qm9_protocol(property_config(), property_run=True)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("training", "batch_size"), 128, "batch_size=256"),
        (("training", "epochs"), 101, "1..100"),
        (("training", "seed"), 7, "matched"),
        (("training", "early_stopping", "patience"), 10, "patience=20"),
        (("training", "early_stopping", "min_delta"), 1e-5, "min_delta=0"),
    ],
)
def test_formal_qm9_property_protocol_rejects_drift(path, value, message):
    config = copy.deepcopy(property_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_qm9_protocol(config, property_run=True)
