"""HomeGrid-style language streaming for multi-object Fetch (gym <=0.17).

Streams one T5 token per env step in parallel with actions (no frozen preread).
Token ids / embeddings come from an offline cache built by scripts/embed_offline.py.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
from gym import spaces

from multi_object_fetch.utils.viz import overlay_language


class LanguageWrapper(gym.Wrapper):
    """Stream task instructions one token per timestep while the agent acts.

    Observation keys added:
      - token (uint32): current T5 token id
      - token_embed (float32[embed_dim]): T5 encoder embedding for that token
      - is_read_step (bool): always False (parallel streaming; kept for Dynalang compat)

    Info keys added:
      - log_language_info (str): full utterance currently being streamed
      - log_stream_prefix (str): decoded text revealed up to the current token
      - log_token_piece (str): decoded text of the token emitted this step
      - instruction (str): episode instruction string

    render(mode='rgb_array') overlays progressive token streaming when visualize=True.
    """

    PAD = "<pad>"

    def __init__(
        self,
        env: gym.Env,
        embeds_path: Optional[Union[str, pathlib.Path]] = None,
        repeat_task_every: int = 20,
        embed_size: int = 512,
        visualize: bool = True,
    ) -> None:
        super().__init__(env)
        self.repeat_task_every = repeat_task_every
        self.embed_size = embed_size
        self.visualize = visualize

        if embeds_path is None:
            embeds_path = pathlib.Path(__file__).resolve().parent / "embeds.pkl"
        embeds_path = pathlib.Path(embeds_path)
        if not embeds_path.exists():
            raise FileNotFoundError(
                f"Embedding cache not found at {embeds_path}. "
                "Generate it with: python scripts/embed_offline.py"
            )
        with open(embeds_path, "rb") as f:
            payload = pickle.load(f)
        if len(payload) == 2:
            raise ValueError(
                f"{embeds_path} is missing per-token string cache. "
                "Re-run: python scripts/embed_offline.py"
            )
        self.token_cache, self.embed_cache, self.token_str_cache = payload

        if self.PAD not in self.token_cache or self.PAD not in self.embed_cache:
            raise KeyError(f"Embedding cache must contain '{self.PAD}'")

        self.empty_token = np.asarray(self.token_cache[self.PAD], dtype=np.uint32).reshape(())
        pad_embed = np.asarray(self.embed_cache[self.PAD], dtype=np.float32)
        if pad_embed.ndim != 1:
            raise ValueError(f"Pad embed must be 1-D, got shape {pad_embed.shape}")
        self.embed_size = int(pad_embed.shape[0])
        self.empty_embed = pad_embed

        base_spaces = dict(env.observation_space.spaces)
        self.observation_space = spaces.Dict(
            {
                **base_spaces,
                "token": spaces.Box(0, 32100, shape=(), dtype=np.uint32),
                "token_embed": spaces.Box(
                    -np.inf, np.inf, shape=(self.embed_size,), dtype=np.float32
                ),
                "is_read_step": spaces.Box(
                    low=np.array(False), high=np.array(True), shape=(), dtype=bool
                ),
            }
        )

        self.string = self.PAD
        self.tokens = [self.empty_token]
        self.token_embeds = [self.empty_embed]
        self.token_pieces: List[str] = [self.PAD]
        self.cur_token = 0
        self._step_cnt = 0
        self._last_task_repeat = 0
        self._last_token = int(self.empty_token)
        self._last_token_index = 0
        self._last_token_count = 1
        self._stream_prefix = self.PAD
        self._current_piece = self.PAD
        self._last_info: Dict[str, Any] = {}

    def _instruction(self) -> str:
        instruction = getattr(self.env, "instruction", None)
        if instruction is None:
            raise AttributeError(
                "Wrapped env must expose `.instruction` (set on reset by MultiObjectFetchEnv)."
            )
        return instruction

    def _tokenize(self, string: str) -> np.ndarray:
        if string not in self.token_cache:
            raise KeyError(
                f"String not in embedding cache: {string!r}. "
                "Append it to multi_object_fetch/language/sentences.txt and re-run "
                "scripts/embed_offline.py"
            )
        tokens = np.asarray(self.token_cache[string])
        if tokens.ndim == 0:
            return tokens.reshape(1).astype(np.uint32)
        return tokens.astype(np.uint32)

    def _embed(self, string: str) -> np.ndarray:
        if string not in self.embed_cache:
            raise KeyError(
                f"String not in embedding cache: {string!r}. "
                "Append it to multi_object_fetch/language/sentences.txt and re-run "
                "scripts/embed_offline.py"
            )
        embeds = np.asarray(self.embed_cache[string], dtype=np.float32)
        if string == self.PAD:
            return embeds.reshape(self.embed_size)
        if embeds.ndim == 1:
            return embeds.reshape(1, self.embed_size)
        return embeds

    def _pieces(self, string: str) -> List[str]:
        if string not in self.token_str_cache:
            raise KeyError(
                f"String not in token-string cache: {string!r}. "
                "Re-run: python scripts/embed_offline.py"
            )
        pieces = list(self.token_str_cache[string])
        if string == self.PAD:
            return pieces
        return pieces

    def _set_current_string(self, string: str) -> None:
        self.string = string
        if string == self.PAD:
            self.tokens = [self.empty_token]
            self.token_embeds = [self.empty_embed]
            self.token_pieces = [self.PAD]
        else:
            self.tokens = list(self._tokenize(string))
            self.token_embeds = list(self._embed(string))
            self.token_pieces = self._pieces(string)
            if not (len(self.tokens) == len(self.token_embeds) == len(self.token_pieces)):
                raise ValueError(
                    f"Token/embed/piece length mismatch for {string!r}: "
                    f"{len(self.tokens)}, {len(self.token_embeds)}, {len(self.token_pieces)}"
                )
        self.cur_token = 0

    def _lang_is_empty(self) -> bool:
        return self.string == self.PAD

    def _increment_token(self) -> None:
        if self._lang_is_empty():
            return
        self.cur_token += 1
        if self.cur_token >= len(self.tokens):
            self._set_current_string(self.PAD)

    def _add_language(self, obs: Dict[str, Any], info: Optional[Dict[str, Any]] = None):
        instruction = self._instruction()

        # Start / restart instruction stream.
        if self._step_cnt == 0 or (
            self.repeat_task_every > 0
            and self._lang_is_empty()
            and self._step_cnt - self._last_task_repeat >= self.repeat_task_every
        ):
            self._set_current_string(instruction)
            self._last_task_repeat = self._step_cnt

        idx = self.cur_token
        token = np.asarray(self.tokens[idx], dtype=np.uint32).reshape(())
        piece = self.token_pieces[idx]
        prefix = "".join(self.token_pieces[: idx + 1])

        obs = dict(obs)
        obs.update(
            {
                "token": token,
                "token_embed": np.asarray(self.token_embeds[idx], dtype=np.float32),
                "is_read_step": np.asarray(False),
            }
        )
        self._last_token = int(token)
        self._last_token_index = idx
        self._last_token_count = len(self.token_pieces)
        self._current_piece = piece
        self._stream_prefix = prefix

        if info is not None:
            info = dict(info)
            info["log_language_info"] = self.string
            info["log_stream_prefix"] = prefix
            info["log_token_piece"] = piece
            info["instruction"] = instruction

        self._increment_token()
        return obs, info

    def render(self, mode="human", **kwargs):
        frame = self.env.render(mode=mode, **kwargs)
        if (
            self.visualize
            and mode == "rgb_array"
            and frame is not None
            and isinstance(frame, np.ndarray)
            and frame.ndim == 3
        ):
            return overlay_language(
                frame,
                instruction=self._instruction(),
                current_piece=self._current_piece,
                token=self._last_token,
            )
        return frame

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._step_cnt = 0
        self._last_task_repeat = 0
        self._set_current_string(self.PAD)
        obs, info = self._add_language(obs, {})
        # gym<=0.17 reset returns obs only; stash language info on the env for callers that want it.
        self._last_info = info
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._step_cnt += 1
        obs, info = self._add_language(obs, info)
        self._last_info = info
        return obs, reward, done, info
