import json
import os
from datetime import datetime
from typing import Set

from .config import (
    BAD_WORDS_CACHE,
    CURRENT_GAME_ID_FILE,
    LAST_SUCCESSFUL_GAME_ID_FILE,
    RESULTS_DIR,
)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_current_game_id() -> int:
    if os.path.exists(CURRENT_GAME_ID_FILE):
        with open(CURRENT_GAME_ID_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except Exception:
                return 0
    return 0


def load_last_successful_game_id() -> int:
    if os.path.exists(LAST_SUCCESSFUL_GAME_ID_FILE):
        with open(LAST_SUCCESSFUL_GAME_ID_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except Exception:
                return 0
    return 0


def bump_and_store_game_id() -> int:
    current = load_current_game_id()
    new = current + 1
    with open(CURRENT_GAME_ID_FILE, "w") as f:
        f.write(str(new))
    return new


def set_current_game_id(value: int) -> None:
    with open(CURRENT_GAME_ID_FILE, "w") as f:
        f.write(str(value))


def set_last_successful_game_id(value: int) -> None:
    with open(LAST_SUCCESSFUL_GAME_ID_FILE, "w") as f:
        f.write(str(value))


def result_filepath(game_id: int, timestamp: str) -> str:
    ensure_dir(RESULTS_DIR)
    return os.path.join(RESULTS_DIR, f"game_{game_id}_{timestamp}.jsonl")


def load_bad_words() -> Set[str]:
    if os.path.exists(BAD_WORDS_CACHE):
        try:
            with open(BAD_WORDS_CACHE, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_bad_words(bad: Set[str]):
    with open(BAD_WORDS_CACHE, "w") as f:
        json.dump(sorted(list(bad)), f)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
