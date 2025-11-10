"""Convenience entry point for running the Stocksight utilities.

This module helps with the first start of the project by making sure the
configuration file exists, offering an optional download of the required NLTK
data, and delegating to the original ``sentiment.py`` and ``stockprice.py``
scripts.

Example usage::

    $ python main.py sentiment -s TSLA -k "Tesla"

    $ python main.py stockprice -s TSLA

    $ python main.py --download-nltk
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.py"
CONFIG_SAMPLE_FILE = PROJECT_ROOT / "config.py.sample"


def ensure_config_file() -> bool:
    """Ensure that ``config.py`` exists.

    If the configuration file is missing but the sample file is available, the
    sample is copied to ``config.py`` so that the user has a starting point.
    A short message is displayed so users know they still need to edit the
    configuration before running the pipelines for real data collection.
    """

    if CONFIG_FILE.exists():
        return True

    if not CONFIG_SAMPLE_FILE.exists():
        print(
            "Es wurde keine config.py gefunden und die Beispiel-Datei "
            "config.py.sample existiert nicht. Bitte lege eine Konfigurationsdatei "
            "manuell an, bevor du das Projekt startest."
        )
        return False

    shutil.copy(CONFIG_SAMPLE_FILE, CONFIG_FILE)
    print(
        "Es wurde keine config.py gefunden. Die config.py.sample wurde als "
        "Ausgangspunkt kopiert. Bitte bearbeite die neue config.py und trage "
        "deine Zugangsdaten (z. B. Twitter API) ein, bevor du einen Dienst startest."
    )
    return False


def download_nltk_data() -> None:
    """Download the NLTK data required by the original scripts."""

    try:
        import nltk
    except ImportError as exc:  # pragma: no cover - defensive path
        print(
            "Die Installation von NLTK wurde nicht gefunden."
            " Installiere zuerst die Abhängigkeiten (siehe requirements.txt)."
        )
        raise SystemExit(1) from exc

    for resource in ("punkt", "stopwords"):
        print(f"Lade NLTK Ressource '{resource}' herunter ...")
        nltk.download(resource)


def run_legacy_script(script_name: str, args: Iterable[str]) -> int:
    """Run one of the legacy entry-point scripts with the given arguments."""

    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        print(
            f"Das Skript '{script_name}' wurde nicht gefunden."
            " Bitte überprüfe deine Installation."
        )
        return 1

    command: List[str] = [sys.executable, str(script_path), *args]
    print("Starte:", " ".join(command))
    return subprocess.call(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hilfsprogramm für den Projektstart. Stelle sicher, dass die"
            " Konfiguration vorhanden ist und delegiere bei Bedarf an"
            " sentiment.py oder stockprice.py."
        )
    )

    parser.add_argument(
        "--download-nltk",
        action="store_true",
        help="Lade die erforderlichen NLTK-Daten (punkt, stopwords) herunter.",
    )

    subparsers = parser.add_subparsers(dest="command")

    sentiment_parser = subparsers.add_parser(
        "sentiment",
        help="Starte das Sammeln und Analysieren von Tweets.",
    )
    sentiment_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help=(
            "Weitere Argumente, die an sentiment.py weitergereicht werden."
            " Beispiel: -s TSLA -k Tesla,SpaceX"
        ),
    )

    stock_parser = subparsers.add_parser(
        "stockprice",
        help="Starte das Sammeln von Börsenkursen.",
    )
    stock_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Weitere Argumente, die an stockprice.py weitergereicht werden.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.download_nltk:
        download_nltk_data()

    config_ready = ensure_config_file()

    if args.command is None:
        if not config_ready:
            # When the configuration file was just created we terminate so the
            # user can edit it before running a worker.
            return 0

        parser.print_help()
        print(
            "\nBeispiele:\n"
            "  python main.py sentiment -s TSLA -k 'Elon Musk',Musk\n"
            "  python main.py stockprice -s TSLA\n"
        )
        return 0

    if not config_ready:
        # We already displayed a message in ensure_config_file().
        return 0

    extra_args: List[str] = getattr(args, "extra_args", [])

    if args.command == "sentiment":
        return run_legacy_script("sentiment.py", extra_args)

    if args.command == "stockprice":
        return run_legacy_script("stockprice.py", extra_args)

    parser.error(f"Unbekannter Befehl: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
