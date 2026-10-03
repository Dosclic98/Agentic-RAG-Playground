"""Download local Tesseract language data used by PyMuPDF's OCR engine."""

import argparse
import re
from pathlib import Path
from urllib.request import urlopen


def language_code(value):
    if not re.fullmatch(r"[a-z]{3}(?:_[A-Za-z0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use a language code such as eng, ita, or deu.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", type=language_code, default=["eng", "ita"])
    parser.add_argument("--directory", default="data/tessdata", help="Destination relative to the project root.")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = (root / arguments.directory).resolve()
    if destination != root and root not in destination.parents:
        parser.error("Language data must stay inside the project.")
    destination.mkdir(parents=True, exist_ok=True)
    for code in arguments.languages:
        target = destination / f"{code}.traineddata"
        temporary = target.with_suffix(".traineddata.part")
        if target.resolve().parent != destination or temporary.resolve().parent != destination:
            parser.error("Language data paths must stay inside the destination directory.")
        if target.is_file() and target.stat().st_size:
            print(f"Already available: {target}")
            continue
        url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{code}.traineddata"
        print(f"Downloading {code} from tesseract-ocr/tessdata_fast...")
        try:
            with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if not temporary.stat().st_size:
                raise RuntimeError(f"Empty language data download for {code}.")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Installed: {target}")


if __name__ == "__main__":
    main()
