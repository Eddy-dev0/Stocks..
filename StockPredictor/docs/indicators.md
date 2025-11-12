# Technical Indicators

This document summarises the technical indicators exposed by the feature assembler. Every indicator is computed using vectorised pandas operations with optional TA-Lib fallbacks when available.

## Trend Indicators

### Supertrend
- **Formula:**
  - Compute the Average True Range (ATR) over *n* periods.
  - Calculate the basic upper/lower bands as `HL2 ± multiplier × ATR`, where `HL2 = (High + Low) / 2`.
  - Derive the final bands recursively by constraining them to prior values and switch the active band when price closes through the opposite band.
- **Outputs:** Trend line and a direction flag (`+1` bullish / `-1` bearish).

### Ichimoku Cloud
- **Components:**
  - Conversion line (Tenkan-sen): `(Highest high + Lowest low) / 2` over the conversion window.
  - Base line (Kijun-sen): `(Highest high + Lowest low) / 2` over the base window.
  - Leading span A: `(Tenkan-sen + Kijun-sen) / 2` shifted forward by the displacement.
  - Leading span B: `(Highest high + Lowest low) / 2` over the span-B window shifted forward.
  - Lagging span (Chikou): closing price shifted backwards by the displacement.

### ADX / DMI
- **Formula:**
  - Compute directional movement (`+DM`, `-DM`) and true range.
  - Smooth directional movement and calculate `+DI = 100 × (+DM / ATR)` and `-DI = 100 × (-DM / ATR)`.
  - The Average Directional Index (ADX) is the smoothed absolute difference between `+DI` and `-DI` normalised by their sum.
- **Outputs:** ADX, +DI, -DI.

### Parabolic SAR
- **Formula:** Iteratively update the stop-and-reverse value using an acceleration factor that increases while price makes new extremes.
- **Outputs:** Single SAR series that trails price.

## Volatility

### Average True Range (ATR)
- **Formula:** Rolling mean of the true range, where `TR = max(High-Low, High-Previous Close, Previous Close-Low)`.

## Momentum Oscillators

### Stochastic Oscillator
- **Formula:**
  - `%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low)` over the lookback window.
  - Fast %D is a moving average of %K; slow variants take additional smoothing windows.
- **Outputs:** Fast %K/%D and slow %K/%D pairs.

### WaveTrend
- **Formula:**
  - Compute the typical price `(High + Low + Close) / 3`.
  - Apply an exponential moving average (ESA) and absolute deviation filter to produce the channel index (`CI`).
  - Smooth `CI` to obtain WaveTrend line (WT1), apply a short moving average for the signal line (WT2), and difference them for the histogram.

### Composite Score
- **Formula:** Rolling z-score normalisation of selected indicators (default: RSI, MACD histogram, ADX, WaveTrend histogram) averaged into a composite momentum score.

## Volume & Liquidity

### VWAP
- **Formula:** Cumulative `(Typical Price × Volume) / Cumulative Volume`.

### Anchored VWAP
- **Formula:** Same as VWAP but the cumulative sums restart from the anchor date. Requires datetime indexed prices when a timestamp is supplied.

### On-Balance Volume (OBV)
- **Formula:** Cumulative volume added when price closes higher and subtracted when it closes lower.

### Money Flow Index (MFI)
- **Formula:** Typical price times volume yields raw money flow. Positive/negative flows are separated by price direction and used to compute the money flow ratio and final oscillator `100 - 100 / (1 + MFR)`.

### Liquidity Proxies
- **Metrics:**
  - Average dollar volume over the chosen window.
  - Rolling standard deviation of volume.
  - Turnover ratio (volume divided by its rolling mean).
  - Impact proxy (`ATR / Dollar Volume`).
  - Sentiment proxy (rolling correlation of returns and volume changes).

## Support & Volatility Structures

### Pivot Points
- **Formula:** Uses prior period high, low, and close.
  - Pivot = `(High + Low + Close) / 3`
  - Resistance/Support levels follow the classical relationships (R1 = `2×Pivot - Low`, S1 = `2×Pivot - High`, etc.).

## Implementation Notes

- TA-Lib functions are used when present for ATR, ADX/DMI, stochastic oscillators, MFI, and Parabolic SAR. Vectorised pandas fallbacks are provided otherwise.
- All outputs are forward and backward filled before being returned to avoid gaps in downstream feature construction.
- Feature assembler exposes indicator parameters through the `technical_indicator_config` argument so experiments can override lookbacks, multipliers, anchors, and component selections.
