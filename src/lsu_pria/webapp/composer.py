from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComposeConfig:
    confidence_threshold: float = 0.75
    stable_frames_min: int = 6
    pause_ms_min: int = 350
    cooldown_ms: int = 800


@dataclass
class ComposeMode:
    """
    words: each confirmed label is appended as a token (word)
    spelling: labels are treated as letters; accumulated into current_word
    both: if label is 1-char A-Z -> spelling, else -> words
    """

    name: str = "both"  # words | spelling | both


@dataclass
class ComposeState:
    config: ComposeConfig = field(default_factory=ComposeConfig)
    mode: ComposeMode = field(default_factory=ComposeMode)
    tokens: list[str] = field(default_factory=list)
    text: str = ""
    current_word: str = ""

    _current_label: Optional[str] = None
    _stable_count: int = 0
    _candidate_label: Optional[str] = None
    _candidate_ready: bool = False
    _last_seen_ms: Optional[int] = None
    _pause_start_ms: Optional[int] = None
    _last_confirmed_label: Optional[str] = None
    _last_confirmed_ms: Optional[int] = None

    def debug_state(self) -> dict:
        return {
            "mode": self.mode.name,
            "tokens": list(self.tokens),
            "current_word": self.current_word,
            "text": self.text,
            "current_label": self._current_label,
            "stable_count": self._stable_count,
            "candidate_label": self._candidate_label,
            "candidate_ready": self._candidate_ready,
            "pause_start_ms": self._pause_start_ms,
            "last_confirmed_label": self._last_confirmed_label,
            "last_confirmed_ms": self._last_confirmed_ms,
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "stable_frames_min": self.config.stable_frames_min,
                "pause_ms_min": self.config.pause_ms_min,
                "cooldown_ms": self.config.cooldown_ms,
            },
        }

    def reset(self) -> None:
        self.tokens.clear()
        self.text = ""
        self.current_word = ""
        self._current_label = None
        self._stable_count = 0
        self._candidate_label = None
        self._candidate_ready = False
        self._last_seen_ms = None
        self._pause_start_ms = None
        self._last_confirmed_label = None
        self._last_confirmed_ms = None

    def rebuild_text(self) -> None:
        if self.current_word:
            self.text = " ".join(self.tokens + [self.current_word])
        else:
            self.text = " ".join(self.tokens)

    def add_space(self) -> None:
        if self.current_word:
            self.tokens.append(self.current_word)
            self.current_word = ""
        self.rebuild_text()

    def backspace(self) -> None:
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.tokens:
            self.tokens.pop()
        self.rebuild_text()

    def _accept_token(self, tok: str) -> None:
        tok = tok.strip()
        if not tok:
            return
        m = self.mode.name
        if m == "words":
            self.tokens.append(tok)
        elif m == "spelling":
            self.current_word += tok
        else:  # both
            if len(tok) == 1 and tok.isalpha():
                self.current_word += tok
            else:
                self.tokens.append(tok)
        self.rebuild_text()

    def update(self, label: str, confidence: float, no_hand: bool, ts_ms: int) -> Optional[str]:
        """
        Returns newly confirmed token if any.
        Confirmation rule:
        - build a candidate when the same label is stable for N frames above threshold
        - when a pause (no_hand) lasts pause_ms_min, confirm the last stable candidate
        - apply cooldown to avoid repeats
        """
        self._last_seen_ms = ts_ms

        if no_hand or confidence < self.config.confidence_threshold or label == "no_hand":
            if self._pause_start_ms is None:
                self._pause_start_ms = ts_ms
            pause_ms = ts_ms - self._pause_start_ms
            if self._candidate_ready and pause_ms >= self.config.pause_ms_min:
                tok = self._candidate_label
                self._candidate_ready = False
                self._candidate_label = None
                self._current_label = None
                self._stable_count = 0
                self._pause_start_ms = None
                if tok:
                    if self._last_confirmed_label == tok and self._last_confirmed_ms is not None:
                        if ts_ms - self._last_confirmed_ms < self.config.cooldown_ms:
                            return None
                    self._last_confirmed_label = tok
                    self._last_confirmed_ms = ts_ms
                    self._accept_token(tok)
                    return tok
            return None

        # Not paused, have a valid prediction.
        self._pause_start_ms = None
        if label != self._current_label:
            self._current_label = label
            self._stable_count = 1
        else:
            self._stable_count += 1

        if self._stable_count >= self.config.stable_frames_min:
            self._candidate_label = self._current_label
            self._candidate_ready = True
        return None
