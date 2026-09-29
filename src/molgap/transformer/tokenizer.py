"""Deterministic regex tokenizer for canonical SMILES strings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


class TokenizationError(ValueError):
    """Raised when the token pattern does not cover a full SMILES string."""


@dataclass(frozen=True)
class EncodedSmiles:
    """A fixed-length encoded SMILES sequence."""

    input_ids: list[int]
    attention_mask: list[bool]
    token_count: int
    unknown_count: int


class SmilesTokenizer:
    """Tokenize full SMILES strings and maintain a train-only vocabulary."""

    def __init__(
        self,
        pattern: str,
        max_length: int,
        special_tokens: dict[str, str],
        vocabulary: dict[str, int] | None = None,
    ) -> None:
        self.pattern = pattern
        self.regex = re.compile(pattern)
        self.max_length = int(max_length)
        self.special_tokens = dict(special_tokens)
        self.vocabulary = dict(vocabulary or {})

    @property
    def pad_id(self) -> int:
        return self.vocabulary[self.special_tokens["pad"]]

    @property
    def unk_id(self) -> int:
        return self.vocabulary[self.special_tokens["unk"]]

    @property
    def cls_id(self) -> int:
        return self.vocabulary[self.special_tokens["cls"]]

    def tokenize(self, smiles: str) -> list[str]:
        """Return tokens only when the regex covers every input character."""
        if not isinstance(smiles, str) or not smiles:
            raise TokenizationError("SMILES must be a non-empty string.")
        tokens: list[str] = []
        position = 0
        for match in self.regex.finditer(smiles):
            if match.start() != position:
                fragment = smiles[position : match.start()]
                raise TokenizationError(
                    f"Unmatched SMILES fragment {fragment!r} at position {position} in {smiles!r}."
                )
            tokens.append(match.group(0))
            position = match.end()
        if position != len(smiles):
            raise TokenizationError(
                f"Unmatched SMILES fragment {smiles[position:]!r} at position {position} "
                f"in {smiles!r}."
            )
        return tokens

    def fit(self, smiles_values: Iterable[str]) -> None:
        """Build a deterministic vocabulary from training SMILES only."""
        observed: set[str] = set()
        for smiles in smiles_values:
            observed.update(self.tokenize(str(smiles)))
        ordered_specials = [
            self.special_tokens["pad"],
            self.special_tokens["unk"],
            self.special_tokens["cls"],
        ]
        if observed.intersection(ordered_specials):
            raise TokenizationError("A special token also occurs as a regular SMILES token.")
        self.vocabulary = {
            token: index for index, token in enumerate([*ordered_specials, *sorted(observed)])
        }

    def encode(self, smiles: str) -> EncodedSmiles:
        """Add [CLS], map to IDs, and pad without silently truncating."""
        if not self.vocabulary:
            raise RuntimeError("Tokenizer vocabulary has not been fitted.")
        tokens = self.tokenize(smiles)
        sequence_length = len(tokens) + 1
        if sequence_length > self.max_length:
            raise TokenizationError(
                f"SMILES requires {sequence_length} positions including [CLS], exceeding "
                f"max_length={self.max_length}: {smiles!r}"
            )
        unknown_count = sum(token not in self.vocabulary for token in tokens)
        ids = [self.cls_id, *(self.vocabulary.get(token, self.unk_id) for token in tokens)]
        mask = [True] * sequence_length
        padding = self.max_length - sequence_length
        return EncodedSmiles(
            input_ids=[*ids, *([self.pad_id] * padding)],
            attention_mask=[*mask, *([False] * padding)],
            token_count=len(tokens),
            unknown_count=unknown_count,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a portable tokenizer representation."""
        return {
            "pattern": self.pattern,
            "max_length": self.max_length,
            "special_tokens": self.special_tokens,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SmilesTokenizer:
        """Restore a tokenizer from a saved artifact."""
        return cls(
            pattern=str(payload["pattern"]),
            max_length=int(payload["max_length"]),
            special_tokens=dict(payload["special_tokens"]),  # type: ignore[arg-type]
            vocabulary={
                str(token): int(index)
                for token, index in dict(payload["vocabulary"]).items()  # type: ignore[arg-type]
            },
        )
