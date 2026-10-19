#!/usr/bin/env python3
"""Personal, rate-limited Studio Ghibli official stills collector.

Usage: python collector.py --root ./archive --max-process 30 --max-new 12
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from urllib.robotparser import RobotFileParser

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, UnidentifiedImageError

LOG = logging.getLogger("ghibli_collector")
BASE = "https://www.ghibli.jp"
WORKS = BASE + "/works/"
COLOURS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")
IMAGE_PATH = re.compile(r"^/gallery/[a-z0-9_./%+-]+\.(?:jpe?g|png|webp)$", re.I)
RAW_GALLERY = re.compile(r"(?<![a-z0-9_.:/])(?:https://www\.ghibli\.jp)?/gallery/[a-z0-9_./%+-]+\.(?:jpe?g|png|webp)", re.I)
SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  url TEXT PRIMARY KEY, film TEXT NOT NULL,
  discovered_at TEXT NOT NULL, checked_at TEXT,
  status TEXT NOT NULL DEFAULT 'pending', reason TEXT,
  colour TEXT, majority REAL, grey_fraction REAL,
  brown_fraction REAL, entropy REAL, saturation REAL,
  sha256 TEXT, dhash TEXT, mean_rgb TEXT, saved_path TEXT, attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_status ON images(status,discovered_at);
CREATE INDEX IF NOT EXISTS idx_sha ON images(sha256);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_url(url: str, kind: str = "image") -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.ghibli.jp" or parsed.port is not None:
        return False
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    path = unquote(parsed.path)
    if kind == "image":
        return bool(IMAGE_PATH.fullmatch(path)) and ".." not in path
    return bool(re.fullmatch(r"/works/(?:[a-z0-9-]+/)?", path))


def find_film_links(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    links: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        url = urljoin(WORKS, a["href"])
        if valid_url(url, "page") and url != WORKS:
            film = urlparse(url).path.strip("/").split("/")[-1]
            links[film] = url
    return links


def find_still_urls(html: str, page_url: str) -> set[str]:
    """Extract only official /gallery/ image URLs, never arbitrary posters/assets."""
    candidates: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(True):
        for key in ("href", "src", "data-src", "data-original", "data-lazy-src", "srcset", "data-srcset"):
            value = element.get(key)
            if not value or not isinstance(value, str):
                continue
            if key.endswith("srcset"):
                raw_values = [v.strip().split()[0] for v in value.split(",") if v.strip()]
            else:
                raw_values = [value.strip()]
            for raw in raw_values:
                full = urljoin(page_url, raw)
                if valid_url(full):
                    candidates.add(full)
    # Handles gallery links embedded as JS or CSS literals without broad scraping.
    for raw in RAW_GALLERY.findall(html.replace("\\/", "/")):
        full = urljoin(BASE, raw)
        if valid_url(full):
            candidates.add(full)
    return candidates


class PoliteHttp:
    def __init__(self, delay: float, timeout: float = 25.0):
        self.delay = delay
        self.timeout = timeout
        self.last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GhibliColourCollector/1.0 (personal non-commercial; one-at-a-time)"})
        self.robots = RobotFileParser()
        self.robots.set_url(BASE + "/robots.txt")
        self._load_robots()

    def _load_robots(self) -> None:
        # Fail closed if robots.txt cannot be checked.
        try:
            response = self.session.get(BASE + "/robots.txt", timeout=self.timeout)
            if response.status_code == 404:
                self.robots.parse([])
            else:
                response.raise_for_status()
                self.robots.parse(response.text.splitlines())
        except requests.RequestException as exc:
            raise RuntimeError(f"Cannot verify robots.txt; refusing to crawl: {exc}") from exc

    def get(self, url: str, maximum: int) -> bytes:
        if not (valid_url(url, "page") or valid_url(url)):
            raise ValueError(f"Non-allowlisted source: {url}")
        if not self.robots.can_fetch(self.session.headers["User-Agent"], url):
            raise PermissionError(f"Robots.txt disallows {url}")
        wait = self.delay - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()
        with self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=False) as response:
            response.raise_for_status()
            if response.is_redirect:
                raise ValueError("Redirect refused for source security")
            if int(response.headers.get("Content-Length", 0)) > maximum:
                raise ValueError("Remote file exceeds size limit")
            chunks, consumed = [], 0
            for chunk in response.iter_content(65536):
                consumed += len(chunk)
                if consumed > maximum:
                    raise ValueError("Downloaded file exceeds size limit")
                chunks.append(chunk)
            return b"".join(chunks)


def dhash(image: Image.Image) -> str:
    pixels = np.asarray(ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    result = sum(int(bit) << pos for pos, bit in enumerate(bits.flat))
    return f"{result:016x}"


def hamming(x: str, y: str) -> int:
    return (int(x, 16) ^ int(y, 16)).bit_count()


def mean_rgb(image: Image.Image) -> str:
    thumbnail = image.convert("RGB").resize((32, 32), Image.Resampling.BOX)
    values = np.asarray(thumbnail, dtype=np.float32).mean(axis=(0, 1)).astype(np.uint8)
    return bytes(values).hex()


def rgb_distance(x: str, y: str) -> float:
    a, b = np.frombuffer(bytes.fromhex(x), dtype=np.uint8).astype(np.int16), np.frombuffer(bytes.fromhex(y), dtype=np.uint8).astype(np.int16)
    return float(np.linalg.norm(a - b))


def analyse(image: Image.Image, *, majority_threshold: float = 0.60,
            max_neutral: float = 0.30, max_brown: float = 0.20,
            min_saturation: float = 0.30, max_entropy: float = 1.70) -> dict:
    """Whole-frame HSV histogram; neutral and brown pixels stay in denominator."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 640 or image.height < 360:
        return {"reason": "too_small", "colour": None, "majority": 0.0}
    img = image.copy()
    img.thumbnail((320, 320), Image.Resampling.LANCZOS)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * (360 / 255), hsv[..., 1] / 255, hsv[..., 2] / 255
    neutral = (s < 0.22) | (v < 0.14)
    # Dark desaturated orange/yellows usually read as brown rather than orange.
    brown = (~neutral) & (h >= 12) & (h < 55) & (v < 0.64)
    eligible = ~(neutral | brown)
    grey_fraction, brown_fraction = float(neutral.mean()), float(brown.mean())
    edges = [12, 43, 72, 165, 195, 255, 305, 345]
    classes = np.searchsorted(edges, h, side="right")
    # 0 red, 1 orange, ... 7 pink, 8 wraps to red
    classes[classes == 8] = 0
    fractions = np.bincount(classes[eligible].astype(np.int32), minlength=8) / h.size
    i = int(np.argmax(fractions))
    majority = float(fractions[i])
    nonzero = fractions[fractions > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    avg_sat = float(s.mean())
    significant = int(np.sum(fractions >= 0.12))
    reason = "accepted"
    if grey_fraction > max_neutral:
        reason = "too_grey_or_desaturated"
    elif brown_fraction > max_brown:
        reason = "too_brown"
    elif avg_sat < min_saturation:
        reason = "low_average_saturation"
    elif majority < majority_threshold:
        reason = "no_dominant_colour"
    elif significant > 2 or entropy > max_entropy:
        reason = "too_multicoloured"
    return {"reason": reason, "colour": COLOURS[i] if reason == "accepted" else None,
            "majority": round(majority, 4), "grey_fraction": round(grey_fraction, 4),
            "brown_fraction": round(brown_fraction, 4), "entropy": round(entropy, 4),
            "saturation": round(avg_sat, 4)}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def discover(http: PoliteHttp, db: sqlite3.Connection, max_film_pages: int = 8) -> int:
    page = http.get(WORKS, 2_000_000).decode("utf-8", errors="replace")
    films = find_film_links(page)
    LOG.info("Discovered %s film pages", len(films))
    total = 0
    items = sorted(films.items())
    offsetrow = db.execute("SELECT value FROM settings WHERE key='film_offset'").fetchone()
    offset = int(offsetrow[0]) % len(items) if offsetrow and items else 0
    selected = [items[(offset + i) % len(items)] for i in range(min(max_film_pages, len(items)))]
    for film, film_url in selected:
        try:
            content = http.get(film_url, 4_000_000).decode("utf-8", errors="replace")
            urls = find_still_urls(content, film_url)
            for url in sorted(urls):
                db.execute("INSERT OR IGNORE INTO images(url,film,discovered_at) VALUES (?,?,?)", (url, film, now()))
            total += len(urls)
            db.commit()
        except Exception as exc:
            LOG.warning("Skipping %s: %s", film, exc)
    if not total:
        raise RuntimeError("No official gallery image links discovered; site markup may have changed")
    db.execute("INSERT INTO settings(key,value) VALUES ('film_offset', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str((offset + len(selected)) % max(1,len(items))),))
    db.execute("INSERT INTO settings(key,value) VALUES ('discovery_at', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (now(),))
    db.commit()
    return total


def export_csv(db: sqlite3.Connection, path: Path) -> None:
    rows = db.execute("SELECT url,film,status,reason,colour,majority,grey_fraction,brown_fraction,entropy,saturation,saved_path,checked_at FROM images ORDER BY film,url")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([x[0] for x in rows.description])
        writer.writerows(rows)


def save_contact_sheets(root: Path, colours: set[str]) -> None:
    for colour in colours:
        folder = root / "by_colour" / colour
        paths = [p for p in sorted(folder.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.name != "_contact_sheet.jpg"]
        if not paths:
            continue
        tile_w, tile_h, columns = 240, 152, 4
        rows = math.ceil(len(paths) / columns)
        sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (245, 245, 245))
        for n, path in enumerate(paths):
            try:
                with Image.open(path) as src:
                    thumbnail = ImageOps.fit(src.convert("RGB"), (tile_w, tile_h), method=Image.Resampling.LANCZOS)
                sheet.paste(thumbnail, ((n % columns) * tile_w, (n // columns) * tile_h))
            except (OSError, UnidentifiedImageError):
                continue
        sheet.save(folder / "_contact_sheet.jpg", quality=85, optimize=True)


def run(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    db = connect(root / "index.sqlite3")
    http = PoliteHttp(args.delay)
    last = db.execute("SELECT value FROM settings WHERE key='discovery_at'").fetchone()
    should_discover = not last or (datetime.now(timezone.utc) - datetime.fromisoformat(last[0])).total_seconds() >= args.discover_hours * 3600
    if should_discover:
        try:
            count = discover(http, db, getattr(args, 'max_film_pages', 8))
            LOG.info("Found %s official still references", count)
        except Exception as exc:
            LOG.warning("Discovery failed: %s", exc)
            # Previously indexed images remain processable.
    count_accepted, count_rejected, count_duplicate, count_errors = 0, 0, 0, 0
    touched = set()
    pending = db.execute("SELECT url,film,attempts FROM images WHERE status IN ('pending','retry') AND attempts<3 ORDER BY discovered_at,url LIMIT ?", (args.max_process,)).fetchall()
    existing_hashes = {row[0] for row in db.execute("SELECT sha256 FROM images WHERE sha256 IS NOT NULL")}
    existing_dhashes = [(row[0], row[1]) for row in db.execute("SELECT dhash,mean_rgb FROM images WHERE status='accepted' AND dhash IS NOT NULL AND mean_rgb IS NOT NULL")]
    for url, film, attempts in pending:
        try:
            payload = http.get(url, 12_000_000)
            with Image.open(io.BytesIO(payload)) as im:
                im.verify()
            with Image.open(io.BytesIO(payload)) as im:
                digest = hashlib.sha256(payload).hexdigest()
                if digest in existing_hashes:
                    outcome = {"reason": "exact_duplicate", "colour": None, "majority": 0}
                else:
                    hash64 = dhash(im)
                    mean = mean_rgb(im)
                    if any(hamming(hash64, old) <= args.dhash_distance and rgb_distance(mean, old_mean) <= 24 for old, old_mean in existing_dhashes):
                        outcome = {"reason": "near_duplicate", "colour": None, "majority": 0}
                    else:
                        outcome = analyse(im, majority_threshold=args.majority,
                                          max_neutral=args.max_neutral, max_brown=args.max_brown,
                                          min_saturation=args.min_saturation, max_entropy=args.max_entropy)
            reason, colour = outcome["reason"], outcome["colour"]
            status, relative_path = "rejected", None
            if reason == "accepted":
                suffix = Path(urlparse(url).path).suffix.lower()
                # Preserve official original file data; no image modification.
                relative_path = f"by_colour/{colour}/{film}-{digest[:14]}{suffix}"
                destination = root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                os.replace(temporary, destination)
                touched.add(colour)
                existing_hashes.add(digest)
                existing_dhashes.append((hash64, mean))
                status = "accepted"
                count_accepted += 1
            else:
                count_rejected += 1
                if reason in ("exact_duplicate", "near_duplicate"):
                    count_duplicate += 1
            db.execute("""UPDATE images SET checked_at=?,status=?,reason=?,colour=?,majority=?,grey_fraction=?,brown_fraction=?,entropy=?,saturation=?,sha256=?,dhash=?,mean_rgb=?,saved_path=?,attempts=attempts+1 WHERE url=?""",
                (now(), status, reason, colour, outcome.get("majority"), outcome.get("grey_fraction"),
                 outcome.get("brown_fraction"), outcome.get("entropy"), outcome.get("saturation"),
                 digest, hash64 if reason not in ("exact_duplicate",) else None, mean if reason not in ("exact_duplicate",) else None, relative_path, url))
            db.commit()
            existing_hashes.add(digest)
            LOG.info("%s %s %s", status, colour or reason, url)
            if count_accepted >= args.max_new:
                break
        except (requests.RequestException, ValueError, OSError, UnidentifiedImageError, PermissionError) as exc:
            count_errors += 1
            next_status = "retry" if attempts < 2 else "error"
            db.execute("UPDATE images SET checked_at=?,status=?,reason=?,attempts=attempts+1 WHERE url=?", (now(), next_status, str(exc)[:300], url))
            db.commit()
            LOG.warning("%s on %s: %s", next_status, url, exc)
    save_contact_sheets(root, touched)
    export_csv(db, root / "manifest.csv")
    totals = dict(db.execute("SELECT status,COUNT(*) FROM images GROUP BY status"))
    summary = {"run_at": now(), "processed": count_accepted + count_rejected + count_errors,
               "accepted": count_accepted, "rejected": count_rejected, "duplicates": count_duplicate,
               "errors": count_errors, "total": totals, "root": str(root)}
    (root / "latest_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("SUMMARY %s", json.dumps(summary))
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./archive"))
    parser.add_argument("--majority", type=float, default=0.60)
    parser.add_argument("--max-neutral", type=float, default=0.30)
    parser.add_argument("--max-brown", type=float, default=0.20)
    parser.add_argument("--min-saturation", type=float, default=0.30)
    parser.add_argument("--max-entropy", type=float, default=1.70)
    parser.add_argument("--dhash-distance", type=int, default=5)
    parser.add_argument("--max-film-pages", type=int, default=8)
    parser.add_argument("--max-process", type=int, default=30)
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--delay", type=float, default=2.5, help="Seconds between requests")
    parser.add_argument("--discover-hours", type=float, default=24)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.majority <= 1 or args.max_process < 1 or args.max_new < 1 or args.delay < 2 or args.max_film_pages < 1:
        parser.error("Threshold must be in [0,1], limits positive and delay >= 2 seconds")
    return args


if __name__ == "__main__":
    options = parse_args()
    logging.basicConfig(level=logging.DEBUG if options.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run(options)