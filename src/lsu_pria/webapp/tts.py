from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TtsEngine:
    """
    Simple offline TTS wrapper (pyttsx3).
    Notes:
    - Runs speak requests serially in a background thread.
    - Uses OS voices; language selection is best-effort.
    """

    _engine: Optional[object] = None
    _lock: threading.Lock = threading.Lock()

    def _ensure(self) -> object:
        if self._engine is not None:
            return self._engine
        import pyttsx3

        self._engine = pyttsx3.init()
        return self._engine

    def speak(self, text: str) -> None:
        def run() -> None:
            with self._lock:
                engine = self._ensure()
                engine.say(text)
                engine.runAndWait()

        t = threading.Thread(target=run, daemon=True)
        t.start()

