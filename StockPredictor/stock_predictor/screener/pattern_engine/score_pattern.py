from __future__ import annotations

from .types import Candle, PatternDetection


def score_pattern(detection: PatternDetection, candles: list[Candle]) -> float:
    volumes = [c.volume for c in candles[-20:] if c.volume is not None]
    avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
    vol_boost = 5.0 if volumes and volumes[-1] > avg_vol else 0.0
    breakout_boost = 8.0 if detection.status == "confirmed" else 0.0
    penalty = 10.0 if detection.status == "failed" else 0.0
    return max(0.0, min(100.0, detection.score + vol_boost + breakout_boost - penalty))
