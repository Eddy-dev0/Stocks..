# Stock Predictor UI frontend

The Streamlit dashboard communicates with the FastAPI service exposed from
`ui/api/app.py`. The following payload conventions allow the UI to forward user
preferences when calling the API endpoints:

## Feature toggles

`POST /forecasts/{ticker}`, `POST /train/{ticker}`, `POST /backtests/{ticker}`
and `POST /buy-zone/{ticker}` accept an optional `feature_toggles` object. This
map should use feature registry names (`elliott`, `fundamental`, `macro`,
`sentiment`, `technical`, `volume_liquidity`) as keys with boolean values to
indicate whether each group should be enabled for the request.

Example payload:

```json
{
  "targets": ["close", "direction"],
  "refresh": false,
  "horizon": 5,
  "feature_toggles": {
    "technical": true,
    "macro": false,
    "sentiment": true
  }
}
```

When omitted, server defaults are used for the feature groups. The UI can use
this shape directly to submit user selections to the backend.

## Screener tab

Der neue **Screener**-Tab scannt automatisch den 1h-Markt (Aktien/Futures) nach gewähltem Pattern, ohne manuelle Symbol-Eingabe.

### Provider-Konfiguration

- Der Screener verwendet aktuell den bestehenden API-Endpunkt `GET /data/{symbol}` als Datenabstraktion (`FrontendAPIMarketDataProvider`).
- Keine API-Keys werden hartcodiert; Schlüssel bleiben in Umgebungsvariablen (`STOCK_PREDICTOR_UI_API_KEY`, Backend-Provider-Konfiguration).
- Wenn kein externer Marktprovider hinter der API konfiguriert ist, zeigt die UI einen Hinweis auf eingeschränkte/fehlende Live-Daten.

### Features

- Pattern-Dropdown mit 16 Chart-Pattern.
- Filter: Markt, Mindestscore, Minimum Sample Size, Mindestvolumen, Pattern-Status, Sortierung.
- Tabelle mit Trade-Qualität im Format `Rating successes/occurrences`, Durchschnitts- und Median-Move.
- CSV-Export der Ergebnisse.
