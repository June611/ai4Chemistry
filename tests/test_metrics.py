import pytest

from molgap.metrics import regression_metrics


def test_regression_metrics() -> None:
    result = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])

    assert result["n"] == 3
    assert result["mae"] == pytest.approx(1 / 3)
    assert result["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert result["mape"] == pytest.approx(100 / 9)
    assert result["mape_n"] == 3
    assert result["r2"] == pytest.approx(0.5)


def test_regression_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        regression_metrics([1.0], [1.0, 2.0])


def test_regression_metrics_mape_excludes_zero_targets() -> None:
    result = regression_metrics([0.0, 2.0], [1.0, 1.0])

    assert result["mape"] == pytest.approx(50.0)
    assert result["mape_n"] == 1
    assert result["mape_excluded_zero_targets"] == 1
