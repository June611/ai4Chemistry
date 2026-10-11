"""Train-only adapter for the official SmilesPE algorithm."""

from __future__ import annotations

import io
from collections.abc import Iterable
from importlib.metadata import version
from typing import Any

from SmilesPE.learner import learn_SPE
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.tokenizer import SPE_Tokenizer

from molgap.transformer.tokenizer import SmilesTokenizer, TokenizationError


class SpeSmilesTokenizer(SmilesTokenizer):
    """Keep official merge semantics and the project's encoding/artifact contract."""

    def __init__(
        self,
        pattern: str,
        max_length: int,
        special_tokens: dict[str, str],
        num_merges: int = 3000,
        min_frequency: int = 100,
        vocabulary: dict[str, int] | None = None,
        merge_codes: str = "",
    ) -> None:
        super().__init__(pattern, max_length, special_tokens, vocabulary)
        if num_merges < 1 or min_frequency < 2:
            raise ValueError("SPE requires num_merges >= 1 and min_frequency >= 2.")
        self.num_merges = int(num_merges)
        self.min_frequency = int(min_frequency)
        self.merge_codes = merge_codes
        self._spe = SPE_Tokenizer(io.StringIO(merge_codes))

    def _atoms(self, smiles: str) -> list[str]:
        # Upstream findall silently drops unmatched characters; enforce full coverage.
        super().tokenize(smiles)
        atoms = atomwise_tokenizer(smiles)
        if "".join(atoms) != smiles:
            raise TokenizationError(f"Official SPE pretokenizer does not cover {smiles!r}.")
        return atoms

    def tokenize(self, smiles: str) -> list[str]:
        self._atoms(smiles)
        tokens = self._spe.tokenize(smiles).split()
        if "".join(tokens) != smiles:
            raise TokenizationError(f"SPE segmentation does not preserve {smiles!r}.")
        return tokens

    def fit(self, smiles_values: Iterable[str]) -> None:
        smiles = [str(value) for value in smiles_values]
        if not smiles:
            raise TokenizationError("Cannot learn SPE on an empty training corpus.")
        atoms = [self._atoms(value) for value in smiles]
        codes = io.StringIO()
        # Upstream learner calls max() on an empty pair table for single-atom corpora.
        if any(len(value) > 1 for value in atoms):
            learn_SPE(
                smiles, codes, num_symbols=self.num_merges,
                min_frequency=self.min_frequency, augmentation=0,
                verbose=False, total_symbols=False,
            )
        self.merge_codes = codes.getvalue()
        self._spe = SPE_Tokenizer(io.StringIO(self.merge_codes))
        # Include train-observed atoms and every learned merge, including intermediate
        # symbols: these can be emitted in new contexts at validation/inference time.
        observed = {atom for molecule in atoms for atom in molecule}
        observed.update("".join(line.split()) for line in self.merge_codes.splitlines())
        specials = [self.special_tokens[key] for key in ("pad", "unk", "cls")]
        if observed.intersection(specials):
            raise TokenizationError("A special token also occurs as a regular SMILES token.")
        self.vocabulary = {
            token: index for index, token in enumerate([*specials, *sorted(observed)])
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "type": "spe",
            "implementation": "SmilesPE",
            "implementation_version": version("SmilesPE"),
            "num_merges": self.num_merges,
            "min_frequency": self.min_frequency,
            "merge_codes": self.merge_codes,
            "learned_merges": len(self.merge_codes.splitlines()),
            "fit_split": "train",
            "augmentation": 0,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpeSmilesTokenizer:
        return cls(
            pattern=payload["pattern"], max_length=int(payload["max_length"]),
            special_tokens=payload["special_tokens"],
            num_merges=int(payload["num_merges"]),
            min_frequency=int(payload["min_frequency"]),
            vocabulary=payload["vocabulary"], merge_codes=payload["merge_codes"],
        )
