"""Thread-safe token counts, rolling live speed, and measured decoding averages."""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass(frozen=True)
class GenerationSnapshot:
    """Token counts, wall elapsed, and independently estimated generation speed."""

    tokens: int
    estimated: bool
    elapsed: float
    rate: Optional[float] = None
    rate_estimated: bool = False


@dataclass
class _Generation:
    characters: int = 0
    active: bool = True
    completed: bool = False
    actual_tokens: Optional[int] = None
    actual_duration: Optional[int] = None
    samples: deque = field(default_factory=lambda: deque(maxlen=512))
    sampled_interval: bool = False

    WINDOW = 1.0

    def sample(self, now):
        cumulative = self.characters / 4.0
        if self.samples and now == self.samples[-1][0]:
            self.samples[-1] = (now, cumulative)
        elif not self.samples or now > self.samples[-1][0]:
            self.sampled_interval = self.sampled_interval or bool(self.samples)
            self.samples.append((now, cumulative))
        self._prune(now)

    def _prune(self, now):
        cutoff = now - self.WINDOW
        # Keep the last sample at/before the boundary as the baseline. Its
        # cumulative count excludes earlier bursts from the current window.
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()

    def live_rate(self, now):
        if not self.sampled_interval:
            return None
        self._prune(now)
        first_time, first_count = self.samples[0]
        last_count = self.samples[-1][1]
        interval = now - max(first_time, now - self.WINDOW)
        if interval <= 0:
            return None
        return max(0.0, (last_count - first_count) / interval)

    @property
    def tokens(self):
        if self.actual_tokens is not None:
            return self.actual_tokens
        # Round once over cumulative text, independent of streamed chunk sizes.
        return (self.characters + 3) // 4

    @property
    def estimated(self):
        return not self.completed or self.actual_tokens is None


class GenerationMetrics:
    """Track completed model calls plus the current call, without doing UI work.

    Live counts use four emitted characters per token until a completed response
    supplies ``eval_count``. Thinking, answer text, and serialized tool calls all
    contribute to this approximation. Live speed uses a rolling one-second
    window of emitted text, starting with its first nonempty chunk. Same-time
    samples coalesce; at most 512 distinct samples are kept, shortening the
    effective window only at unusually high update frequency. This is an
    estimate of streamed throughput, not exact tokenizer or decoding speed.

    Final speed uses only paired ``eval_count`` / ``eval_duration`` metadata
    from completed model calls, weighted by decoding time in nanoseconds.
    Startup, prompt processing, and tool time affect elapsed but not this rate.

    A new generation discards an unfinished prior attempt, so retries count
    completed responses and the active attempt rather than abandoned work.
    ``begin_turn`` starts a fresh total and ``finish_turn`` freezes its timer.
    """

    def __init__(self, clock=None):
        # Look up monotonic dynamically so terminal tests can patch the clock.
        self._clock = clock if clock is not None else lambda: time.monotonic()
        self._lock = Lock()
        self._started = None
        self._finished = None
        self._tokens = 0
        self._estimated = False
        self._current = None
        self._generation_seen = False
        self._completed_generations = 0
        self._actual_total = 0
        self._duration_total = 0
        self._durations_valid = True
        self._final_rate = None

    @staticmethod
    def _actual_count(value):
        return value if type(value) is int and value >= 0 else None

    @staticmethod
    def _actual_duration(value):
        return value if type(value) is int and value > 0 else None

    @staticmethod
    def _json_default(value):
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            return dump()
        attributes = getattr(value, "__dict__", None)
        return attributes if attributes is not None else str(value)

    @classmethod
    def _characters(cls, chunk):
        message = getattr(chunk, "message", None)
        count = sum(len(text) for text in (
            getattr(message, "thinking", None), getattr(message, "content", None)
        ) if isinstance(text, str))
        # Serialize each call independently so splitting a list across chunks
        # does not repeatedly count list punctuation or change the estimate.
        for call in getattr(message, "tool_calls", None) or ():
            count += len(json.dumps(call, ensure_ascii=False, separators=(",", ":"),
                                    default=cls._json_default))
        return count

    def begin_turn(self, started=None):
        with self._lock:
            self._started = self._clock() if started is None else started
            self._finished = None
            self._tokens = 0
            self._estimated = False
            self._current = None
            self._generation_seen = False
            self._completed_generations = 0
            self._actual_total = 0
            self._duration_total = 0
            self._durations_valid = True
            self._final_rate = None

    def begin_generation(self):
        with self._lock:
            if self._started is None:
                self._started = self._clock()
            self._current = _Generation()
            self._generation_seen = True

    def progress(self, chunk):
        # Serialization can be comparatively expensive; never block a renderer
        # taking a snapshot while doing it.
        characters = self._characters(chunk)
        done = getattr(chunk, "done", False) is True
        actual = self._actual_count(getattr(chunk, "eval_count", None)) if done else None
        duration = self._actual_duration(getattr(chunk, "eval_duration", None)) if done else None
        with self._lock:
            if (self._current is None or not self._current.active or self._current.completed
                    or self._finished is not None):
                return
            self._current.characters += characters
            if characters:
                self._current.sample(self._clock())
            if done:
                self._current.completed = True
                self._current.actual_tokens = actual
                self._current.actual_duration = duration

    def _commit_completed(self):
        if self._current is not None and self._current.completed:
            self._tokens += self._current.tokens
            self._estimated = self._estimated or self._current.estimated
            self._completed_generations += 1
            if self._current.actual_tokens is None or self._current.actual_duration is None:
                self._durations_valid = False
            else:
                self._actual_total += self._current.actual_tokens
                self._duration_total += self._current.actual_duration
            self._current = None

    def end_generation(self):
        with self._lock:
            # An interrupted call remains provisional until a retry replaces it
            # or the turn finishes; repeated cleanup does not double its count.
            if self._current is not None:
                self._current.active = False
            self._commit_completed()

    def finish_turn(self, generated_tokens=None):
        with self._lock:
            if self._finished is not None:
                return
            unfinished = self._current is not None and not self._current.completed
            self._commit_completed()
            actual = self._actual_count(generated_tokens)
            if (not unfinished and self._completed_generations and self._durations_valid
                    and self._duration_total and (actual is None or actual == self._actual_total)):
                self._final_rate = self._actual_total * 1e9 / self._duration_total
            if actual is not None:
                self._tokens = actual
                self._estimated = False
                self._current = None
            elif not self._generation_seen:
                # A failure before generation provides no measured zero.
                self._estimated = True
            self._finished = self._clock()

    def snapshot(self):
        with self._lock:
            now = self._finished if self._finished is not None else self._clock()
            elapsed = max(0.0, now - self._started) if self._started is not None else 0.0
            tokens = self._tokens + (self._current.tokens if self._current is not None else 0)
            estimated = self._estimated or (self._current is not None and self._current.estimated)
            rate = (self._final_rate if self._finished is not None else
                    self._current.live_rate(now)
                    if self._current is not None and self._current.active else None)
            return GenerationSnapshot(tokens=tokens, estimated=estimated, elapsed=elapsed,
                                      rate=rate, rate_estimated=rate is not None and self._finished is None)
