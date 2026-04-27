from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from ..swing_points import compute_atr, find_swing_points
from ..types import Candle, PatternDetection


@dataclass(frozen=True)
class CupAndHandleOptions:
    min_cup_bars: int = 20
    max_cup_bars: int = 180
    min_handle_bars: int = 3
    max_handle_bars: int = 40
    rim_tolerance_percent: float = 0.06
    min_cup_depth_percent: float = 0.08
    max_cup_depth_percent: float = 0.50
    max_handle_retrace_of_cup: float = 0.50
    ideal_max_handle_retrace_of_cup: float = 0.33
    require_rounded_bottom: bool = True
    require_volume_confirmation: bool = False
    min_confidence: float = 60.0
    breakout_buffer_percent: float = 0.002
    scan_window_min: int = 80
    scan_window_max: int = 250
    swing_left_bars: int = 3
    swing_right_bars: int = 3


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denom = sum((idx - mean_x) ** 2 for idx in range(n))
    if denom == 0:
        return 0.0
    num = sum((idx - mean_x) * (val - mean_y) for idx, val in enumerate(values))
    return num / denom


def _avg_volume(candles: list[Candle], start: int, end: int) -> float:
    if end < start:
        return 0.0
    vols = [candles[idx].volume for idx in range(start, end + 1)]
    return float(mean(vols)) if vols else 0.0


def _timestamp_text(candle: Candle) -> str:
    return candle.timestamp.isoformat(sep=" ", timespec="minutes")


def _build_explanation(detection: dict[str, float | int | str | None], candles: list[Candle]) -> str:
    status = str(detection["status"])
    start = candles[int(detection["startIndex"])]
    end = candles[int(detection["rightRimIndex"])]
    base = f"Rounded cup formed between {_timestamp_text(start)} and {_timestamp_text(end)}."
    if status == "confirmed":
        return f"Confirmed Cup and Handle: price closed above handle resistance. {base}"
    if status == "forming":
        return f"Forming Cup and Handle: handle is valid but breakout above resistance has not occurred. {base}"
    return f"Failed Cup and Handle: structure formed but fell below invalidation. {base}"


def _status_rank(status: str) -> int:
    return {"confirmed": 3, "forming": 2, "failed": 1}.get(status, 0)


def _score_detection(
    *,
    rim_diff_percent: float,
    cup_depth_percent: float,
    rounded_bottom_ok: bool,
    side_balance: float,
    left_slope: float,
    right_slope: float,
    v_shaped: bool,
    handle_low_ok: bool,
    handle_retrace: float,
    handle_bars_ok: bool,
    status: str,
    close_to_breakout: bool,
    breakout_volume_ratio: float | None,
    lower_handle_volume: bool,
    illiquid: bool,
    gap_distorted: bool,
) -> float:
    score = 0.0
    score += 20
    if rim_diff_percent <= 0.03:
        score += 10
    elif rim_diff_percent <= 0.06:
        score += 5

    if 0.12 <= cup_depth_percent <= 0.35:
        score += 10
    elif 0.08 <= cup_depth_percent <= 0.50:
        score += 5

    if rounded_bottom_ok:
        score += 15
    if side_balance >= 0.5:
        score += 10
    elif side_balance >= 0.3:
        score += 5
    if left_slope < 0 and right_slope > 0:
        score += 10

    if handle_low_ok:
        score += 15
    if handle_retrace <= 0.33:
        score += 10
    elif handle_retrace <= 0.50:
        score += 5
    if handle_bars_ok:
        score += 5

    if status == "confirmed":
        score += 15
    elif close_to_breakout:
        score += 5

    if breakout_volume_ratio is not None:
        if breakout_volume_ratio >= 1.5:
            score += 15
        elif breakout_volume_ratio >= 1.3:
            score += 10
        elif breakout_volume_ratio < 0.8:
            score -= 10

    if lower_handle_volume:
        score += 5

    if v_shaped:
        score -= 20
    if not handle_low_ok:
        score -= 20
    if rim_diff_percent > 0.06:
        score -= 15
    if illiquid:
        score -= 10
    if gap_distorted:
        score -= 10

    return max(0.0, min(100.0, score * 0.75))


def _dedupe_overlapping(detections: list[dict[str, float | int | str | None]]) -> list[dict[str, float | int | str | None]]:
    if not detections:
        return []

    ranked = sorted(
        detections,
        key=lambda d: (
            _status_rank(str(d["status"])),
            float(d["confidence"]),
            int(d["endIndex"]),
        ),
        reverse=True,
    )

    kept: list[dict[str, float | int | str | None]] = []
    for cand in ranked:
        overlap_found = False
        for existing in kept:
            cand_start = int(cand["startIndex"])
            cand_end = int(cand["endIndex"])
            ex_start = int(existing["startIndex"])
            ex_end = int(existing["endIndex"])
            overlap = max(0, min(cand_end, ex_end) - max(cand_start, ex_start) + 1)
            cand_len = max(1, cand_end - cand_start + 1)
            ex_len = max(1, ex_end - ex_start + 1)
            overlap_ratio = overlap / min(cand_len, ex_len)
            similar_rims = (
                abs(int(cand["leftRimIndex"]) - int(existing["leftRimIndex"])) <= 3
                and abs(int(cand["rightRimIndex"]) - int(existing["rightRimIndex"])) <= 3
            )
            if overlap_ratio > 0.70 and similar_rims:
                overlap_found = True
                break
        if not overlap_found:
            kept.append(cand)
    return sorted(kept, key=lambda d: int(d["endIndex"]))


def detect_cup_and_handle_candidates(
    candles: list[Candle],
    options: CupAndHandleOptions | None = None,
) -> list[dict[str, float | int | str | None]]:
    cfg = options or CupAndHandleOptions()
    if len(candles) < max(cfg.min_cup_bars + cfg.min_handle_bars + 5, 40):
        return []

    recent_count = min(len(candles), cfg.scan_window_max)
    start_offset = max(0, len(candles) - recent_count)
    working = candles[start_offset:]

    swings = find_swing_points(
        working,
        left_bars=cfg.swing_left_bars,
        right_bars=cfg.swing_right_bars,
        min_move_atr=0.2,
    )
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if not highs or not lows:
        return []

    detections: list[dict[str, float | int | str | None]] = []
    atr = compute_atr(working)
    closes = [c.close for c in working]

    for left in highs:
        for bottom in lows:
            if bottom.index <= left.index:
                continue
            for right in highs:
                if right.index <= bottom.index:
                    continue
                cup_bars = right.index - left.index
                if cup_bars < cfg.min_cup_bars or cup_bars > cfg.max_cup_bars:
                    continue

                left_rim_price = working[left.index].high
                right_rim_price = working[right.index].high
                cup_bottom_price = working[bottom.index].low
                rim_price = min(left_rim_price, right_rim_price)
                rim_diff_percent = abs(left_rim_price - right_rim_price) / max(left_rim_price, 1e-9)
                if rim_diff_percent > cfg.rim_tolerance_percent:
                    continue

                cup_depth = rim_price - cup_bottom_price
                if cup_depth <= 0:
                    continue
                cup_depth_percent = cup_depth / max(rim_price, 1e-9)
                if not (cfg.min_cup_depth_percent <= cup_depth_percent <= cfg.max_cup_depth_percent):
                    continue

                left_side = bottom.index - left.index
                right_side = right.index - bottom.index
                side_balance = min(left_side, right_side) / max(left_side, right_side)
                v_shaped = left_side < cfg.min_cup_bars * 0.25 or right_side < cfg.min_cup_bars * 0.25
                if v_shaped:
                    continue
                if side_balance < 0.30:
                    continue

                bottom_threshold = cup_bottom_price + cup_depth * 0.20
                bottom_zone_bars = sum(
                    1
                    for idx in range(left.index, right.index + 1)
                    if working[idx].low <= bottom_threshold or working[idx].close <= bottom_threshold
                )
                bottom_zone_needed = max(3, int(cup_bars * 0.10))
                rounded_bottom_ok = bottom_zone_bars >= bottom_zone_needed
                if cfg.require_rounded_bottom and not rounded_bottom_ok:
                    continue

                left_slope = _slope(closes[left.index : bottom.index + 1])
                right_slope = _slope(closes[bottom.index : right.index + 1])
                if left_slope >= 0 or right_slope <= 0:
                    continue

                strength_lookback = min(20, left.index)
                if strength_lookback >= 5:
                    prior_strength = closes[left.index] / max(closes[left.index - strength_lookback], 1e-9) - 1
                    if prior_strength < -0.03:
                        continue

                handle_search_start = right.index
                handle_limit = min(len(working) - 1, right.index + cfg.max_handle_bars - 1)
                if handle_limit - handle_search_start + 1 < cfg.min_handle_bars:
                    continue

                rim_breakout_level = max(left_rim_price, right_rim_price)
                earliest_breakout_after_rim: int | None = None
                for idx in range(handle_search_start + cfg.min_handle_bars - 1, len(working)):
                    if working[idx].close > rim_breakout_level * (1 + cfg.breakout_buffer_percent):
                        earliest_breakout_after_rim = idx
                        break
                if earliest_breakout_after_rim is not None:
                    handle_end_min = handle_search_start + cfg.min_handle_bars - 1
                    handle_end_max = min(handle_limit, earliest_breakout_after_rim - 1)
                else:
                    handle_end_min = len(working) - 1
                    handle_end_max = len(working) - 1
                if handle_end_max < handle_end_min:
                    continue

                for handle_end in range(handle_end_min, handle_end_max + 1):
                    handle_bars = handle_end - handle_search_start + 1
                    if handle_bars > cup_bars * 0.50:
                        break

                    handle_high = max(c.high for c in working[handle_search_start : handle_end + 1])
                    handle_low = min(c.low for c in working[handle_search_start : handle_end + 1])
                    handle_depth = handle_high - handle_low
                    handle_retrace = handle_depth / max(cup_depth, 1e-9)
                    if handle_retrace > cfg.max_handle_retrace_of_cup:
                        continue

                    cup_midpoint = cup_bottom_price + cup_depth * 0.50
                    handle_low_ok = handle_low >= cup_midpoint
                    if not handle_low_ok:
                        continue

                    handle_bars_ok = (
                        cfg.min_handle_bars <= handle_bars <= cfg.max_handle_bars and handle_bars <= cup_bars * 0.40
                    )
                    if not handle_bars_ok:
                        continue

                    handle_slope = _slope(closes[handle_search_start : handle_end + 1])
                    if handle_slope > 0.03:
                        continue

                    breakout_level = max(left_rim_price, right_rim_price, handle_high)
                    breakout_index: int | None = None
                    for idx in range(handle_end + 1, len(working)):
                        if working[idx].close > breakout_level * (1 + cfg.breakout_buffer_percent):
                            breakout_index = idx
                            break

                    invalidation_level = min(handle_low, right_rim_price - cup_depth * 0.50)
                    failed = any(working[idx].close < invalidation_level for idx in range(handle_end, len(working)))
                    if failed:
                        status = "failed"
                    elif breakout_index is not None:
                        status = "confirmed"
                    else:
                        status = "forming"

                    average_volume_cup = _avg_volume(working, left.index, right.index)
                    average_volume_handle = _avg_volume(working, handle_search_start, handle_end)
                    breakout_volume_ratio: float | None = None
                    if breakout_index is not None:
                        vol_lookback_start = max(0, breakout_index - 20)
                        avg20 = _avg_volume(working, vol_lookback_start, breakout_index - 1)
                        if avg20 > 0:
                            breakout_volume_ratio = working[breakout_index].volume / avg20
                    lower_handle_volume = average_volume_handle > 0 and average_volume_handle < average_volume_cup

                    close_to_breakout = (
                        breakout_index is None
                        and working[-1].close >= breakout_level * (1 - 0.005)
                        and working[-1].close <= breakout_level * (1 + cfg.breakout_buffer_percent)
                    )
                    illiquid = average_volume_cup < 50_000
                    gap_distorted = False
                    if atr > 0:
                        for idx in range(left.index + 1, right.index + 1):
                            gap = abs(working[idx].open - working[idx - 1].close)
                            if gap > 2.5 * atr:
                                gap_distorted = True
                                break

                    confidence = _score_detection(
                        rim_diff_percent=rim_diff_percent,
                        cup_depth_percent=cup_depth_percent,
                        rounded_bottom_ok=rounded_bottom_ok,
                        side_balance=side_balance,
                        left_slope=left_slope,
                        right_slope=right_slope,
                        v_shaped=v_shaped,
                        handle_low_ok=handle_low_ok,
                        handle_retrace=handle_retrace,
                        handle_bars_ok=handle_bars_ok,
                        status=status,
                        close_to_breakout=close_to_breakout,
                        breakout_volume_ratio=breakout_volume_ratio,
                        lower_handle_volume=lower_handle_volume,
                        illiquid=illiquid,
                        gap_distorted=gap_distorted,
                    )
                    if confidence < cfg.min_confidence:
                        continue
                    if cfg.require_volume_confirmation and breakout_index is not None:
                        if breakout_volume_ratio is None or breakout_volume_ratio < 1.0:
                            continue

                    result: dict[str, float | int | str | None] = {
                        "patternType": "Cup and Handle",
                        "direction": "Bullish",
                        "status": status,
                        "confidence": confidence,
                        "startIndex": left.index + start_offset,
                        "endIndex": (breakout_index if breakout_index is not None else handle_end) + start_offset,
                        "leftRimIndex": left.index + start_offset,
                        "cupBottomIndex": bottom.index + start_offset,
                        "rightRimIndex": right.index + start_offset,
                        "handleStartIndex": handle_search_start + start_offset,
                        "handleEndIndex": handle_end + start_offset,
                        "breakoutIndex": breakout_index + start_offset if breakout_index is not None else None,
                        "neckline": breakout_level,
                        "breakoutLevel": breakout_level,
                        "invalidationLevel": invalidation_level,
                        "cupDepthPercent": cup_depth_percent,
                        "handleDepthPercent": handle_retrace,
                        "averageVolumeCup": average_volume_cup,
                        "averageVolumeHandle": average_volume_handle,
                        "breakoutVolumeRatio": breakout_volume_ratio,
                        "explanation": "",
                    }
                    result["explanation"] = _build_explanation(result, candles)
                    detections.append(result)

    return _dedupe_overlapping(detections)


def detect_cup_and_handle_details(
    candles: list[Candle],
    options: CupAndHandleOptions | None = None,
) -> dict[str, float | int | str | None] | None:
    candidates = detect_cup_and_handle_candidates(candles, options)
    if not candidates:
        return None
    best = sorted(
        candidates,
        key=lambda d: (
            _status_rank(str(d["status"])),
            float(d["confidence"]),
            int(d["endIndex"]),
        ),
        reverse=True,
    )[0]
    return best


def detect_cup_and_handle(candles: list[Candle]) -> PatternDetection | None:
    details = detect_cup_and_handle_details(candles)
    if details is None:
        return None
    return PatternDetection(
        pattern_type="Cup and Handle",
        status=str(details["status"]),
        direction="bullish",
        score=float(details["confidence"]),
        start_index=int(details["startIndex"]),
        end_index=int(details["endIndex"]),
        breakout_level=float(details["breakoutLevel"]),
        invalidation_level=float(details["invalidationLevel"]),
        neckline_level=float(details["neckline"]),
    )
