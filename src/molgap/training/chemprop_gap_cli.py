"""Run the official Chemprop CLI with a project-local gap metric callback."""

from __future__ import annotations

import lightning.pytorch as pl

from molgap.training.gap_metric import GapMAE, GapMetricScale


class _GapTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", None) or [])
        kwargs["callbacks"] = [GapMetricScale(), *callbacks]
        super().__init__(*args, **kwargs)


class _LightningProxy:
    Trainer = _GapTrainer

    def __getattr__(self, name):
        return getattr(pl, name)


def main():
    import chemprop.cli.train as cli_train
    from chemprop.cli.main import main as official_main

    # Only this subprocess's CLI namespace changes; no installed file is edited.
    cli_train.pl = _LightningProxy()
    evaluate_test = cli_train.evaluate_and_save_predictions

    def evaluate_without_selection_metric(preds, loader, metrics, output_dir, args):
        # The CLI's report loops over individual columns, then averages tasks.
        # gap_mae is validation-only; project evaluation reports raw test gap MAE.
        metrics = [metric for metric in metrics if not isinstance(metric, GapMAE)]
        return evaluate_test(preds, loader, metrics, output_dir, args)

    cli_train.evaluate_and_save_predictions = evaluate_without_selection_metric
    official_main()


if __name__ == "__main__":
    main()
