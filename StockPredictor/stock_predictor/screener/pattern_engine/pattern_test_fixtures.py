from __future__ import annotations

from datetime import datetime, timedelta

from .types import Candle


def _candles_from_close(close_values: list[float]) -> list[Candle]:
    base = datetime(2025, 1, 1)
    out: list[Candle] = []
    for i, c in enumerate(close_values):
        out.append(
            Candle(
                timestamp=base + timedelta(hours=i),
                open=c,
                high=c * 1.01,
                low=c * 0.99,
                close=c,
                volume=1_000_000 + i * 500,
            )
        )
    return out


def create_double_bottom_fixture() -> list[Candle]:
    return _candles_from_close([130, 127, 124, 120, 116, 112, 108, 104, 101, 103, 106, 109, 106, 103, 101.2, 103, 106, 109, 112, 115])


def create_double_top_fixture() -> list[Candle]:
    return _candles_from_close([90, 92, 95, 99, 103, 107, 111, 114, 117, 115, 112, 109, 112, 116.5, 114, 111, 108, 104, 101, 99, 97])


def create_triple_bottom_fixture() -> list[Candle]:
    return _candles_from_close([140, 136, 132, 127, 121, 115, 110, 106, 101, 104, 108, 111, 106, 102, 105, 109, 112, 107, 103, 106, 111, 116, 120])


def create_triple_top_fixture() -> list[Candle]:
    return _candles_from_close([80, 84, 88, 92, 97, 102, 107, 111, 116, 112, 108, 104, 109, 114, 111, 107, 103, 109, 113, 110, 106, 101, 96])


def create_head_and_shoulders_fixture() -> list[Candle]:
    return _candles_from_close([100, 104, 108, 112, 116, 121, 118, 114, 120, 127, 123, 117, 120, 124, 119, 113, 108, 103])


def create_inverted_head_and_shoulders_fixture() -> list[Candle]:
    return _candles_from_close([130, 126, 122, 118, 113, 108, 112, 116, 110, 103, 108, 114, 111, 107, 112, 118, 123])


def create_ascending_triangle_fixture() -> list[Candle]:
    return _candles_from_close([100, 101, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 110, 112])


def create_descending_triangle_fixture() -> list[Candle]:
    return _candles_from_close([120, 118, 116, 117, 115, 114, 113, 114, 112, 111, 110, 111, 109, 108, 107, 108, 104, 101])


def create_flag_fixture() -> list[Candle]:
    return _candles_from_close(
        [
            94,
            95,
            96,
            96.5,
            97,
            97.5,
            98,
            98.5,
            99,
            100,
            101,
            103,
            105,
            107,
            109,
            111,
            113,
            115,
            117,
            119,
            121,
            120.5,
            120,
            119.5,
            119,
            118.5,
            118,
            117.5,
            117,
            118.5,
        ]
    )


def create_bearish_flag_fixture() -> list[Candle]:
    return _candles_from_close(
        [
            156,
            155,
            154,
            153.5,
            153,
            152.5,
            152,
            151.5,
            151,
            150,
            148,
            146,
            144,
            142,
            140,
            138,
            136,
            134,
            132,
            130,
            128,
            128.5,
            129,
            129.5,
            130,
            130.5,
            131,
            131.5,
            132,
            130.5,
        ]
    )


def create_pennant_fixture() -> list[Candle]:
    return _candles_from_close([100, 104, 108, 112, 116, 121, 126, 130, 132, 131, 130, 129, 130, 129.5, 129, 129.3, 129.8, 130.5])


def create_channel_fixture() -> list[Candle]:
    return _candles_from_close([100, 103, 101, 104, 102, 105, 103, 106, 104, 107, 105, 108, 106, 109, 107, 110, 108, 111, 109, 112])


def create_channel_up_fixture() -> list[Candle]:
    return _candles_from_close([100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110])


def create_channel_down_fixture() -> list[Candle]:
    return _candles_from_close([130, 128, 129, 127, 128, 126, 127, 125, 126, 124, 125, 123, 124, 122, 123, 121, 122, 120])


def create_cup_and_handle_fixture() -> list[Candle]:
    return _candles_from_close([100, 98, 95, 92, 90, 89, 88, 89, 91, 93, 95, 97, 99, 101, 103, 104, 103, 102, 101, 102, 103, 105])


def create_diamond_fixture() -> list[Candle]:
    return _candles_from_close([100, 103, 97, 106, 94, 110, 92, 107, 95, 104, 97, 102, 99, 101, 98, 100, 99, 97, 95, 93, 91, 89])
