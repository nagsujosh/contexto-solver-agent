"""Contexto Solver package.

Public API is exposed lazily to keep imports lightweight.
"""

__all__ = ["play_game_and_record", "load_current_game_id", "load_last_successful_game_id"]


def __getattr__(name):  # pragma: no cover
    if name == "play_game_and_record":
        from .pipeline import play_game_and_record

        return play_game_and_record
    if name == "load_current_game_id":
        from .io import load_current_game_id

        return load_current_game_id
    if name == "load_last_successful_game_id":
        from .io import load_last_successful_game_id

        return load_last_successful_game_id
    raise AttributeError(name)


