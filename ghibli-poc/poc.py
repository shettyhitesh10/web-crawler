#!/usr/bin/env python3
"""Read-only, rate-limited POC. Original accepted stills go to a temporary CI artifact, never git."""
from __future__ import annotations
import csv, hashlib, io, json, os, re, time
from pathlib import Path
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from PIL import Image, ImageOps
import requests
import numpy as np

ROOT = Path("ghibli-poc/output")
ROOT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ghibli.jp"
AGENT = "GhibliColourCollector-PoC/1.0 (personal, noncommercial, 2s between requests)"
session = requests.Session()
session.headers.update({"User-Agent": AGENT})
last = 0.0

def get(url, max_bytes):
    global last
    wait = max(0, 2.3 - (time.monotonic() - last))
    if wait: time.sleep(wait)
    last = time.monotonic()
    r = session.get(url, timeout=30, allow_redirects=False)
    r.raise_for_status()
    if r.is_redirect: raise RuntimeError("redirect blocked")
    if len(r.content) > max_bytes: raise RuntimeError("too large")
    return r.content

robots_url = BASE + "/robots.txt"
try:
    r = session.get(robots_url, timeout=25, allow_redirects=False)
    if r.status_code == 404:
        robots = RobotFileParser()
        robots.parse([])
    else:
        r.raise_for_status()
        robots = RobotFileParser()
        robots.parse(r.text.splitlines())
except Exception as e:
    raise SystemExit(f"POC blocked: could not verify robots.txt: {e}")

def classify(body):
    im = Image.open(io.BytesIO(body))
    im.verify()
    im = Image.open(io.BytesIO(body))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width < 640 or im.height < 360: return "too_small", None, 0, 0, 0
    im.thumbnail((320, 320), Image.Resampling.LANCZOS)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[:,:,0]*(360/255), hsv[:,:,1]/255, hsv[:,:,2]/255
    grey = (s < .22) | (v < .14)
    brown = (~grey) & (h >= 12) & (h < 55) & (v < .64)
    c = np.searchsorted([12,43,72,165,195,255,305,345], h, side="right")
    c[c==8] = 0
    freq = np.bincount(c[~(grey|brown)].astype(np.int32),minlength=8)/h.size
    i = int(np.argmax(freq))
    share=float(freq[i])
    names = ["red","orange","yellow","green","cyan","blue","purple","pink"]
    significant=int(np.sum(freq>=.12))
    entropy=-float(np.sum(freq[freq>0]*np.log2(freq[freq>0])))
    reason = ("grey" if grey.mean() > .30 else "brown" if brown.mean() > .20
              else "desaturated" if s.mean() < .30 else "no_majority" if share < .60
              else "multi_colour" if significant>2 or entropy>1.70 else "accepted")
    return reason, names[i], round(share,4),round(float(grey.mean()),4),round(float(brown.mean()),4)

# Candidate names match Ghibli's documented official gallery naming convention.
films = ["ponyo", "totoro", "chihiro", "marnie", "mononoke", "laputa", "kazetachinu", "kimitachi"]
progress_path = Path("ghibli-poc/progress.json")
progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
already_checked = set(progress.get("processed_urls", []))
all_candidates = [(film, BASE+f"/gallery/{film}{n:03d}.jpg") for n in range(1,51) for film in films]
candidates = [(film,url) for film,url in all_candidates if url not in already_checked][:24]
rows = []
saved_hashes = set(progress.get("saved_sha256", []))
accepted=0
downloaded=0
for film, url in candidates:
    if len(rows)>=24 or accepted>=6: break
    if not robots.can_fetch(AGENT, url):
        rows.append([url, "robots_disallow", "", 0, 0, 0, ""])
        continue
    try:
        content = get(url, 12_000_000)
        downloaded += 1
        digest = hashlib.sha256(content).hexdigest()
        if digest in saved_hashes:
            rows.append([url,"duplicate","",0,0,0,""])
            continue
        reason,colour,share,grey,brown = classify(content)
        saved = ""
        if reason=="accepted":
            folder = ROOT/"by_colour"/colour
            folder.mkdir(parents=True,exist_ok=True)
            destination = folder/(film+"-"+digest[:12]+".jpg")
            destination.write_bytes(content)
            with Image.open(destination) as verify:
                verify.verify()
            saved = str(destination)
            saved_hashes.add(digest)
            accepted += 1
        rows.append([url,reason,colour,share,grey,brown,saved])
        print(f"{len(rows):3d} {reason:16s} {colour or '':7s} share={share:.2f} {film}",flush=True)
    except Exception as e:
        rows.append([url, "error", "",0,0,0, str(e)[:150]])
        print(f"{len(rows):3d} ERROR {film}: {e}",flush=True)

# Checkpoint processed source URLs and hashes without placing copyrighted images in git.
processed_now = {r[0] for r in rows if r[1] != "error"}
progress["processed_urls"] = sorted(already_checked | processed_now)
progress["saved_sha256"] = sorted(saved_hashes)
progress["accepted_total"] = int(progress.get("accepted_total", 0)) + accepted
progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
with (ROOT/"manifest.csv").open("w",newline="") as f:
    writer=csv.writer(f); writer.writerow(["official_url","status","dominant_colour","fraction","grey_fraction","brown_fraction","saved_path"]); writer.writerows(rows)
summary = {"attempted":len(rows),"downloaded":downloaded,"accepted":accepted,"rejected":sum(1 for r in rows if r[1] not in ("accepted","error")),"errors":sum(r[1]=="error" for r in rows),"saved_files":[r[-1] for r in rows if r[1]=="accepted"]}
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2))
print("FINAL_SUMMARY "+json.dumps(summary),flush=True)
# An incremental hourly run with zero qualifying stills is normal.
