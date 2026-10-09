"""Synthetic training data: commentator-style paraphrases of real event descriptions (+ ASR noise)
and hard negatives (keyword mentions that are NOT an event happening now).

Backends
--------
* ``template`` - offline, rule-based commentator phrases. Always available (used in CPU smoke tests).
* ``hf``       - local instruction model through ``transformers`` (default
  ``Qwen/Qwen2.5-1.5B-Instruct``, fits a Colab T4 in fp16). Free, no API key.
* ``anthropic``- Claude API (``pip install anthropic`` + ``ANTHROPIC_API_KEY``).

Each sample is a 3-part window ``before / event / after`` so it matches the shape of the Echoes
windows (context + current segment). Every sample records ``source`` ("synthetic"/"hardneg") and
``meta`` (backend, seed id, category) so it can be traced and ablated.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from ..data.types import Sample
from .text_utils import asr_noise

# ---------------------------------------------------------------- template backend
_PLAYERS = ["Silva", "Muller", "Rodriguez", "Kane", "Benzema", "Lewandowski", "Salah", "Griezmann",
            "Modric", "Hazard", "Costa", "Neymar", "Ramos", "Pique", "Reus", "Immobile", "Mertens"]
_TEAMS = ["the home side", "the visitors", "Madrid", "Dortmund", "Chelsea", "Juventus", "Bayern", "Arsenal",
          "Barcelona", "Liverpool", "Inter", "Napoli", "Paris"]

_TEMPLATES = {
    "Goal": {
        "before": ["{p} picks it up on the edge of the box", "here comes {t} again", "cross into the middle",
                   "{p} cuts inside", "it falls for {p}", "a quick break from {t}"],
        "event": ["and it's in! {p} scores!", "goal! {p} finds the bottom corner", "{p} puts it away, what a finish",
                  "it's in the back of the net, {p}!", "{p} smashes it home", "and {p} makes no mistake",
                  "that's a goal for {t}", "{p} heads it in"],
        "after": ["the keeper had no chance", "what a moment for {t}", "the crowd erupts",
                  "{p} runs off to celebrate", "they have the lead now", "a superb strike"],
    },
    "Card": {
        "before": ["that's a late challenge from {p}", "{p} goes right through the back of him",
                   "the referee has seen that", "a cynical foul to stop the break", "{p} catches him high"],
        "event": ["and he's going in the book", "the referee shows {p} the yellow card", "{p} is booked",
                  "that's a yellow card for {p}", "straight red! {p} is sent off", "second yellow, {p} has to go",
                  "out comes the card for {p}", "{p} gets a caution"],
        "after": ["he can have no complaints", "he'll have to be careful now", "{t} are down to ten men",
                  "the free kick to come", "that was reckless", "a deserved booking"],
    },
    "Penalty": {
        "before": ["{p} goes down in the area", "he's clipped inside the box", "{p} is brought down",
                   "contact in the box", "the referee is pointing"],
        "event": ["penalty! the referee points to the spot", "it's a penalty to {t}", "he's given the penalty",
                  "{p} steps up to take the penalty", "the spot kick from {p}", "penalty kick for {t}",
                  "and the referee awards a penalty"],
        "after": ["{p} places the ball on the spot", "big chance from twelve yards", "the keeper waits",
                  "the players surround the referee", "this could decide the game"],
    },
}

# Hard negatives: mention the class keyword but nothing happens *now*.
HARDNEG_TEMPLATES = {
    "replay": ["let's see that goal again", "another look at the goal from {p}", "on the replay you can see the finish",
               "watch the replay of the penalty incident", "here is the yellow card from earlier on the replay"],
    "past_reference": ["{p} scored the opening goal earlier", "that first half goal changed the game",
                       "he already has a yellow card from the first half", "they scored from a penalty last week",
                       "{t} have conceded a goal in every game", "his goal against {t} last season"],
    "near_miss": ["oh that was almost a goal", "so close to a goal there", "{p} hits the post", "inches wide of the goal",
                  "what a save, that was nearly a goal", "it's cleared off the line"],
    "should_be": ["that should have been a card", "he was lucky not to get a yellow there", "surely that's a penalty? no",
                  "the referee waves play on, no penalty", "they wanted a penalty but nothing given",
                  "he could have been booked for that"],
    "statistics": ["{p} has eleven goals this season", "no cards so far in this game", "{t} need a goal",
                   "they have not won a penalty all season", "the goal difference matters here"],
    "goal_area": ["goal kick for {t}", "the goalkeeper takes it short", "it's played into the penalty area",
                  "a shot from the edge of the penalty area, blocked", "the ball goes out for a goal kick",
                  "he's inside the penalty box but loses it"],
    "disallowed": ["the goal is ruled out for offside", "no goal, the flag is up", "VAR checking, and it's no goal",
                   "they thought they had scored but it's disallowed"],
}


def _fill(s: str, rng: random.Random) -> str:
    return s.format(p=rng.choice(_PLAYERS), t=rng.choice(_TEAMS))


def template_event(cls: str, rng: random.Random) -> tuple[str, str, str]:
    t = _TEMPLATES[cls]
    return _fill(rng.choice(t["before"]), rng), _fill(rng.choice(t["event"]), rng), _fill(rng.choice(t["after"]), rng)


def template_hardneg(category: str, rng: random.Random) -> tuple[str, str, str]:
    neutral = ["the ball is played back", "they keep possession", "a long ball forward", "throw in to {t}",
               "{p} on the ball", "patient build up from {t}"]
    return _fill(rng.choice(neutral), rng), _fill(rng.choice(HARDNEG_TEMPLATES[category]), rng), _fill(rng.choice(neutral), rng)


# ---------------------------------------------------------------- LLM backends
EVENT_PROMPT = """You write realistic live TV football commentary as it would be transcribed by speech recognition.
Below is a written description of a real match event ({cls}).
Write {n} different short commentary snippets of that moment, spoken style (short broken phrases, excitement,
no names of sources, no score lines). Each snippet has three parts:
"before" (what the commentator says just before), "event" (the line when it happens), "after" (reaction).
Return ONLY a JSON list of objects with keys before, event, after.

Description: {desc}"""

HARDNEG_PROMPT = """You write realistic live TV football commentary as transcribed by speech recognition.
Write {n} short commentary snippets that MENTION a {cls_lower} but where NO {cls_lower} actually happens at this moment.
Category: {category} ({category_hint}). Each snippet has keys "before", "event", "after" (the "event" line contains the
mention). Return ONLY a JSON list of objects."""

CATEGORY_HINTS = {
    "replay": "talking over a replay of an earlier incident",
    "past_reference": "recalling an earlier goal/card/penalty or another match",
    "near_miss": "almost a goal, hits the post, saved",
    "should_be": "arguing that it should have been a card/penalty but nothing is given",
    "statistics": "season statistics or needs",
    "goal_area": "goal kick, goalkeeper, penalty area/box used as a location",
    "disallowed": "goal ruled out, offside, VAR overturns",
}


def _parse_json_list(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("event")]


class LLMBackend:
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class HFBackend(LLMBackend):
    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 512, temperature: float = 0.9):
        from transformers import pipeline

        self.pipe = pipeline("text-generation", model=model, torch_dtype="auto", device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        out = self.pipe([{"role": "user", "content": prompt}], max_new_tokens=self.max_new_tokens,
                        do_sample=True, temperature=self.temperature, return_full_text=False)
        return out[0]["generated_text"]


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def generate(self, prompt: str) -> str:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            return "[]"
        return "".join(b.text for b in resp.content if b.type == "text")


def make_backend(name: str, **kw) -> LLMBackend | None:
    if name == "template":
        return None
    if name == "hf":
        return HFBackend(**kw)
    if name == "anthropic":
        return AnthropicBackend(**kw)
    raise ValueError(f"unknown synthetic backend {name!r}")


# ---------------------------------------------------------------- generation
@dataclass
class SynthConfig:
    backend: str = "template"
    targets: dict | None = None  # class -> number of event samples, e.g. {"Penalty": 3000, ...}
    hardneg_per_category: int = 300
    per_call: int = 5
    noise: bool = True
    seed: int = 42
    backend_kwargs: dict | None = None


def _mk(parts: tuple[str, str, str], label: str, source: str, rng: random.Random, noise: bool, meta: dict) -> Sample:
    b, e, a = parts
    if noise:
        b, e, a = (asr_noise(x, rng) for x in (b, e, a))
    return Sample(text=e, label=label, source=source, ctx_before=b, ctx_after=a, game=f"{source}:{meta.get('seed_id', '')}",
                  split="train", meta=meta)


def generate_synthetic(seeds: dict[str, list[str]], cfg: SynthConfig) -> list[Sample]:
    """``seeds``: class -> real event descriptions from TRAIN games (Echoes event windows, captions)."""
    rng = random.Random(cfg.seed)
    backend = make_backend(cfg.backend, **(cfg.backend_kwargs or {}))
    targets = cfg.targets or {"Penalty": 3000, "Goal": 1000, "Card": 1000}
    out: list[Sample] = []
    for cls, n in targets.items():
        pool = seeds.get(cls) or [""]
        made = 0
        tries = 0
        while made < n and tries < n * 3:
            tries += 1
            sid = rng.randrange(len(pool))
            if backend is None:
                items = [template_event(cls, rng)]
            else:
                txt = backend.generate(EVENT_PROMPT.format(cls=cls, n=cfg.per_call, desc=pool[sid][:600]))
                items = [(d.get("before", ""), d["event"], d.get("after", "")) for d in _parse_json_list(txt)]
            for it in items[: n - made]:
                out.append(_mk(it, cls, "synthetic", rng, cfg.noise,
                               {"backend": cfg.backend, "seed_id": f"{cls}-{sid}"}))
                made += 1
    return out


def generate_hard_negatives(cfg: SynthConfig) -> list[Sample]:
    rng = random.Random(cfg.seed + 1)
    backend = make_backend(cfg.backend, **(cfg.backend_kwargs or {}))
    out: list[Sample] = []
    for cat in HARDNEG_TEMPLATES:
        made = 0
        tries = 0
        while made < cfg.hardneg_per_category and tries < cfg.hardneg_per_category * 3:
            tries += 1
            if backend is None:
                items = [template_hardneg(cat, rng)]
            else:
                cls = rng.choice(["goal", "card", "penalty"])
                txt = backend.generate(HARDNEG_PROMPT.format(n=cfg.per_call, cls_lower=cls, category=cat,
                                                             category_hint=CATEGORY_HINTS[cat]))
                items = [(d.get("before", ""), d["event"], d.get("after", "")) for d in _parse_json_list(txt)]
            for it in items[: cfg.hardneg_per_category - made]:
                out.append(_mk(it, "No-Event", "hardneg", rng, cfg.noise,
                               {"backend": cfg.backend, "category": cat, "seed_id": f"{cat}-{made}"}))
                made += 1
    return out


def mine_echoes_hard_negatives(samples: list[Sample], max_n: int | None = None, seed: int = 42,
                               event_times: dict | None = None, min_gap: float = 120.0) -> list[Sample]:
    """No-Event Echoes TRAIN windows that mention a class keyword (natural hard negatives). Returned as
    copies with source="hardneg" (meta.category="mined") so they can be oversampled.

    ``event_times`` maps (game, half) -> [(time, cls)]; windows closer than ``min_gap`` seconds to an
    event of a mentioned class are skipped, because they are often late reactions/replays of the real
    event (label noise) rather than true negatives."""
    from ..data.align import mentions_any

    rng = random.Random(seed)

    def far(s: Sample) -> bool:
        if not event_times:
            return True
        cls = set(mentions_any(s.text))
        return all(abs(s.start - t) > min_gap for t, c in event_times.get((s.game, s.half), []) if c in cls)

    mined = [s for s in samples if s.split == "train" and s.source == "echoes" and s.label == "No-Event"
             and mentions_any(s.text) and far(s)]
    rng.shuffle(mined)
    if max_n:
        mined = mined[:max_n]
    out = []
    for s in mined:
        d = s.to_dict()
        d.update(source="hardneg", meta={"category": "mined", "orig_game": s.game})
        out.append(Sample(**d))
    return out
