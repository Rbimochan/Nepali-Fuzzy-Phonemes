#!/usr/bin/env python3
"""
download_slr54.py -- Download the OpenSLR SLR54 Nepali ASR corpus.

Files:
  - utt_spk_text.tsv        (FileID / UserID / transcript index, ~10.9 MB)
  - asr_nepali_0.zip ... asr_nepali_f.zip   (16 shards, ~589 MB each, ~9.4 GB total)

Features:
  - Parallel downloads (default 4 workers)
  - Resume of partial downloads (HTTP Range)
  - Expected-size verification against server Content-Length
  - Automatic mirror fallback if the primary mirror fails
  - --max-shards to fetch only the first N shards (e.g. a small audio subset)

Usage examples:
  python download_slr54.py --out-dir ../data/slr54
  python download_slr54.py --out-dir ../data/slr54 --max-shards 2
  python download_slr54.py --out-dir ../data/slr54 --jobs 8 --extract
  python download_slr54.py --out-dir ../data/slr54 --mirror trmal --insecure
"""

import argparse
import concurrent.futures
import os
import ssl
import sys
import threading
import urllib.error
import urllib.request
import zipfile

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    tqdm = None

# Assign a unique console line to each parallel tqdm bar so they do not overlap.
_bar_lock = None
_bar_counter = None

# Mirror list (tried in order until one works for a given file).
# us.openslr.org is not a default because it occasionally has TLS issues.
MIRRORS = {
    "openslr": "https://www.openslr.org/resources/54",
    "trmal": "https://openslr.trmal.net/resources/54",
    "elda": "https://openslr.elda.org/resources/54",
    "magicdata": "https://openslr.magicdatatech.com/resources/54",
    "us": "https://us.openslr.org/resources/54",
}

INDEX_FILE = "utt_spk_text.tsv"

# 16 shards: 0..9 then a..f
SHARD_NAMES = [f"asr_nepali_{i}.zip" for i in range(10)] + [
    f"asr_nepali_{c}.zip" for c in "abcdef"
]


def make_ssl_context(insecure):
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context()


def get_expected_size(url, timeout=30, insecure=False):
    """Return Content-Length for url (HEAD request), or None if unavailable."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=make_ssl_context(insecure)) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def download_file(url, dest, expected_size, timeout=60, insecure=False):
    """
    Download url to dest with resume support.

    Returns True on success, False if the download failed (caller may try the
    next mirror), or "skip" if the file already exists and is complete.
    """
    partial = dest + ".part"
    start = 0
    if os.path.exists(partial):
        start = os.path.getsize(partial)
        if expected_size is not None and start >= expected_size:
            start = expected_size

    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if expected_size is not None:
            if size == expected_size:
                return "skip"
            # corrupt / incomplete final file -> start over
            os.remove(dest)
            size = 0
        else:
            # no size to verify against; assume complete
            return "skip"

    target = partial if start < (expected_size or 0) else dest

    bar = None
    try:
        req = urllib.request.Request(url)
        if start > 0:
            req.add_header("Range", f"bytes={start}-")
        with urllib.request.urlopen(req, timeout=timeout, context=make_ssl_context(insecure)) as resp:
            mode = "ab" if start > 0 else "wb"
            total = expected_size if expected_size is not None else None

            if tqdm is not None and total is not None:
                global _bar_lock, _bar_counter
                if _bar_lock is None:
                    _bar_lock = threading.Lock()
                    _bar_counter = 0
                with _bar_lock:
                    position = _bar_counter
                    _bar_counter += 1
                bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                           initial=start, desc=os.path.basename(url), miniters=1,
                           position=position, dynamic_ncols=True)

            with open(target, mode) as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            if bar is not None:
                bar.close()
                bar = None
    except Exception:
        if bar is not None:
            bar.close()
        return False

    size = os.path.getsize(target)
    if expected_size is not None and size < expected_size:
        return False
    if target != dest:
        os.replace(target, dest)
    return True


def resolve_url(name, mirror, insecure=False):
    base = MIRRORS[mirror]
    return f"{base}/{name}"


def download(name, mirrors, out_dir, expected_sizes, jobs, insecure):
    """Try each mirror for `name`; return (ok:bool, message:str)."""
    dest = os.path.join(out_dir, name)
    expected = expected_sizes.get(name)
    for mirror in mirrors:
        url = resolve_url(name, mirror, insecure)
        result = download_file(url, dest, expected, insecure=insecure)
        if result is True:
            return True, f"{name}: OK ({expected} bytes) via {mirror}"
        if result == "skip":
            return True, f"{name}: already present, skipped"
        # failed -> try next mirror
    return False, f"{name}: FAILED on all mirrors"


def main():
    ap = argparse.ArgumentParser(description="Download OpenSLR SLR54 (Nepali ASR corpus).")
    ap.add_argument("--out-dir", default="data/slr54", help="destination directory")
    ap.add_argument("--mirror", default="openslr", choices=sorted(MIRRORS),
                    help="primary mirror (auto-fallback to others on failure)")
    ap.add_argument("--mirrors", default=None,
                    help="comma-separated ordered mirror list, e.g. 'trmal,elda,openslr' "
                         "(overrides --mirror)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel download workers")
    ap.add_argument("--max-shards", type=int, default=len(SHARD_NAMES),
                    help="download only the first N shards (e.g. 2 for a small subset)")
    ap.add_argument("--no-index", action="store_true",
                    help="skip utt_spk_text.tsv (only useful if a shard zip already carries it)")
    ap.add_argument("--extract", action="store_true",
                    help="extract downloaded zips in place after downloading")
    ap.add_argument("--delete-zips", action="store_true",
                    help="delete zips after successful extraction (implies --extract)")
    ap.add_argument("--insecure", action="store_true",
                    help="disable TLS certificate verification (for broken mirrors)")
    args = ap.parse_args()

    if args.mirrors:
        ordered = [m.strip() for m in args.mirrors.split(",") if m.strip() in MIRRORS]
    else:
        ordered = [args.mirror] + [m for m in MIRRORS if m != args.mirror]

    os.makedirs(args.out_dir, exist_ok=True)

    shards = SHARD_NAMES[: args.max_shards]
    files = ([] if args.no_index else [INDEX_FILE]) + shards

    print(f"Mirrors: {ordered}")
    print(f"Downloading {len(files)} files to {os.path.abspath(args.out_dir)} "
          f"(shards limited to {args.max_shards})")

    # Probe expected sizes from the primary mirror.
    expected_sizes = {}
    for name in files:
        url = resolve_url(name, ordered[0], args.insecure)
        size = get_expected_size(url, insecure=args.insecure)
        if size is not None:
            expected_sizes[name] = size
        print(f"  {name:24s} expected ~{size} bytes" if size is not None
              else f"  {name:24s} expected unknown (resume disabled)")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(download, name, ordered, args.out_dir, expected_sizes, args.jobs, args.insecure): name
                for name in files}
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            ok, msg = fut.result()
            results[name] = ok
            print(msg)

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print("\nFAILED files:", ", ".join(failed))
        sys.exit(1)

    total = sum(os.path.getsize(os.path.join(args.out_dir, n)) for n in files)
    print(f"\nAll files downloaded. Total size: {total / 1e9:.2f} GB")

    if args.delete_zips or args.extract:
        for name in shards:
            path = os.path.join(args.out_dir, name)
            if not os.path.exists(path):
                continue
            print(f"Extracting {name} ...")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(args.out_dir)
            if args.delete_zips:
                os.remove(path)
                print(f"  deleted {name}")


if __name__ == "__main__":
    main()