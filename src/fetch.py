"""Fetch the raw tennis corpus into ``data/`` and record provenance.

This is the *only* sanctioned way data enters the project. Nothing under
``data/`` is committed (see ``.gitignore``); reproducibility and integrity come
from the pinned commit SHAs and the per-file SHA-256 hashes this script writes
into ``data/MANIFEST.json``.

Sources (all compiled by Jeff Sackmann, CC BY-NC-SA 4.0 — non-commercial,
attribution required, share-alike; see ``ATTRIBUTION.md``):

  * ``slam``  Grand Slam point-by-point, 2011-2024 (the PRIMARY dataset).
  * ``atp``   Men's match results + rankings + players (retirement labels for
              the Stage 0.5 positive control, ranking covariates for the
              weak-player test). Restricted to the 2011-2024 slam window.
  * ``wta``   Women's equivalent, for replication / generalization.
  * ``mcp``   Match Charting Project shot-by-shot (rally length, error type)
              for separating strategic from involuntary contamination.

Provenance is UNEVEN and the code encodes it (see ``SOURCES``):

  The three JeffSackmann repos named in the spec (``tennis_slam_pointbypoint``,
  ``tennis_atp``, ``tennis_wta``) return 404 as of Sept 2026. Do NOT re-point
  ``slam``/``atp``/``wta`` at a ``JeffSackmann/...`` URL — it will fail, and
  silently substituting a different source breaks provenance. Those three come
  from the archival mirror ``Aneeshers/tennis-sackmann-archive``, pinned by full
  40-char SHA. The Match Charting Project is still live upstream and is pulled
  directly from JeffSackmann, also pinned.

  The mirror is an ARCHIVE, not a fork: it shares no git history with upstream,
  so its contents cannot be diffed against the original repos. A commit pin to a
  third party is not self-verifying if that party also disappears, so we record
  a SHA-256 per file and verify it before use. The Hugging Face copy of the same
  archive is a second fallback, usable ONLY when its bytes reproduce our
  recorded hashes.

Design notes worth respecting (see CLAUDE.md):
  * Singles only. The slam directory also carries ``-doubles`` / ``-mixed``
    files; those are filtered out here so a later glob can't pull them in.
  * ``atp``/``wta`` match files are windowed to 2011-2024 and tour-level main
    draw only (no futures / qual_chall / doubles).
  * Idempotent: a file already present whose SHA-256 matches the manifest is not
    re-downloaded, so a partial run resumes cleanly.

Usage
-----
    python -m src.fetch                 # fetch every group
    python -m src.fetch slam atp        # fetch a subset
    python -m src.fetch --list          # show what each group would fetch
    python -m src.fetch --verify        # re-hash local files against MANIFEST;
                                        # a mismatch is a HARD STOP (exit 1)
    python -m src.fetch --source hf     # fetch via the Hugging Face fallback
                                        # (bytes still verified against hashes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --- provenance: pinned commits ------------------------------------------------
# Pin the FULL 40-char SHA, never an abbreviation (asserted below). Bump these
# deliberately and note it in the trial log; never let the corpus drift silently
# underneath a result.
#
# MIRROR_SHA verified 2026-09-20 against the authoritative repo
# (api.github.com/repos/Aneeshers/tennis-sackmann-archive/commits/main): the
# recorded value equals the live main HEAD byte-for-byte.
MIRROR_REPO = "Aneeshers/tennis-sackmann-archive"
MIRROR_SHA = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
MCP_REPO = "JeffSackmann/tennis_MatchChartingProject"
MCP_SHA = "1813a1309b7ed7ebf1c7e884b32bf675d00e4edf"

# Second fallback: the same archive on the Hugging Face Hub. Pinned to a
# revision, but treated as byte-equal ONLY when hashes reproduce.
HF_REPO = "Aneeshers/tennis-sackmann-archive"
HF_REVISION = "main"

# Per-source provenance quality, recorded verbatim into the manifest so the
# unevenness is never lost. Upstream repos are now dead (kept as the origin URL
# / "provenance origin", per the license attribution requirement).
SOURCES = {
    "slam": {
        "via": "mirror",
        "upstream_repo": "https://github.com/JeffSackmann/tennis_slam_pointbypoint",
        "upstream_status": "404 as of 2026-09 (repo removed)",
        "upstream_commit": "6febb77 (abbrev; full SHA unrecoverable — upstream gone)",
        "snapshot_date": "2024-10",
        "coverage": "2011-2024 (NOT uniformly complete; see README/audit)",
        "provenance_quality": "good — mirror names the upstream commit",
    },
    "atp": {
        "via": "mirror",
        "upstream_repo": "https://github.com/JeffSackmann/tennis_atp",
        "upstream_status": "404 as of 2026-09 (repo removed)",
        "upstream_commit": None,
        "snapshot_date": "2026-06",
        "coverage": "through 2026 (windowed here to 2011-2024)",
        "provenance_quality": "WEAK — upstream SHA not recorded by the mirror; trust risk",
    },
    "wta": {
        "via": "mirror",
        "upstream_repo": "https://github.com/JeffSackmann/tennis_wta",
        "upstream_status": "404 as of 2026-09 (repo removed)",
        "upstream_commit": None,
        "snapshot_date": "2026-06",
        "coverage": "through 2026 (windowed here to 2011-2024)",
        "provenance_quality": "WEAK — upstream SHA not recorded by the mirror; trust risk",
    },
    "mcp": {
        "via": "upstream",
        "upstream_repo": "https://github.com/JeffSackmann/tennis_MatchChartingProject",
        "upstream_status": "live",
        "upstream_commit": MCP_SHA,
        "snapshot_date": "2026-09",
        "coverage": "men's charting through 2020s",
        "provenance_quality": "good — pinned directly to a live upstream commit",
    },
}

WINDOW = range(2011, 2025)  # 2011-2024 inclusive, the slam point-by-point span

GH_RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
HF_RAW = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
API_CONTENTS = "https://api.github.com/repos/{repo}/contents/{path}?ref={sha}"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# --- per-group file selection --------------------------------------------------

def _is_slam_singles(name: str) -> bool:
    if "doubles" in name or "mixed" in name:
        return False
    if name in ("data_dictionary.txt", "UPSTREAM_README.md"):
        return True
    return bool(re.fullmatch(r"20\d\d-\w+-(points|matches)\.csv", name))


def _is_tour_window(name: str) -> bool:
    """Main-draw singles match/rankings/players files inside the slam window.

    Excludes doubles, futures and qualifier/challenger files: the slam
    point-by-point corpus is tour-level main draw, so the join targets are too.
    """
    if name in ("matches_data_dictionary.txt", "UPSTREAM_README.md",
                "atp_players.csv", "wta_players.csv"):
        return True
    if re.search(r"_(doubles|futures|qual_chall|amateur)_", name):
        return False
    m = re.fullmatch(r"(?:atp|wta)_matches_(20\d\d)\.csv", name)
    if m:
        return int(m.group(1)) in WINDOW
    # Rankings covering the 2010s / 2020s (2011-2024 fits within these two).
    return bool(re.fullmatch(r"(?:atp|wta)_rankings_(10s|20s)\.csv", name))


def _is_mcp_core(name: str) -> bool:
    """Men's charting: match index + decade point files covering the window."""
    if name in ("charting-m-matches.csv",
                "charting-m-points-2010s.csv",
                "charting-m-points-2020s.csv"):
        return True
    return name.endswith("data_dictionary.txt")


GROUPS = {
    "slam": dict(repo=MIRROR_REPO, sha=MIRROR_SHA, src="slam_pointbypoint",
                 dest="slam_pointbypoint", keep=_is_slam_singles),
    "atp": dict(repo=MIRROR_REPO, sha=MIRROR_SHA, src="atp",
                dest="atp", keep=_is_tour_window),
    "wta": dict(repo=MIRROR_REPO, sha=MIRROR_SHA, src="wta",
                dest="wta", keep=_is_tour_window),
    "mcp": dict(repo=MCP_REPO, sha=MCP_SHA, src="",
                dest="match_charting", keep=_is_mcp_core),
}

# Guardrails asserted at import so a bad edit fails loudly, not silently.
assert _SHA_RE.match(MIRROR_SHA), "MIRROR_SHA must be a full 40-char hex SHA"
assert _SHA_RE.match(MCP_SHA), "MCP_SHA must be a full 40-char hex SHA"
for _g in ("slam", "atp", "wta"):
    assert GROUPS[_g]["repo"] == MIRROR_REPO, (
        f"{_g} must come from the mirror, never a JeffSackmann/... URL "
        "(upstream is dead; substituting a source breaks provenance)")


# --- http helpers --------------------------------------------------------------

def _get(url: str, *, tries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tennis-contamination-fetch"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:  # transient
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to GET {url}: {last}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_dir(repo: str, sha: str, path: str) -> list[dict]:
    """List a directory at a pinned commit via the GitHub contents API.

    Returns entries with ``name`` and ``size`` (bytes). Used only for the file
    listing (a handful of API calls); the files themselves come from raw hosts
    so the 60/hour unauthenticated API budget is not a concern.
    """
    url = API_CONTENTS.format(repo=repo, path=path, sha=sha)
    entries = json.loads(_get(url))
    return [{"name": e["name"], "size": e["size"]}
            for e in entries if e["type"] == "file"]


def _url_for(group: str, rel_path: str, source: str) -> str:
    g = GROUPS[group]
    if source == "hf":
        # HF holds the mirror archive under the same directory layout; MCP is
        # not in that archive, so MCP has no HF fallback.
        if group == "mcp":
            raise ValueError("no Hugging Face fallback for the Match Charting Project")
        return HF_RAW.format(repo=HF_REPO, rev=HF_REVISION, path=rel_path)
    return GH_RAW.format(repo=g["repo"], sha=g["sha"], path=rel_path)


# --- fetch ---------------------------------------------------------------------

def plan(group: str) -> list[dict]:
    g = GROUPS[group]
    src = g["src"]
    listing = list_dir(g["repo"], g["sha"], src)
    chosen = [e for e in listing if g["keep"](e["name"])]
    for e in chosen:
        e["rel_path"] = f"{src}/{e['name']}" if src else e["name"]
    return sorted(chosen, key=lambda e: e["name"])


def _load_recorded_hashes() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    m = json.loads(MANIFEST.read_text())
    return {f["file"]: f.get("sha256") for f in m.get("files", []) if f.get("sha256")}


def fetch_group(group: str, records: list[dict], source: str,
                recorded: dict[str, str]) -> None:
    g = GROUPS[group]
    dest_dir = DATA_DIR / g["dest"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = plan(group)
    total = len(files)
    got = skipped = 0
    print(f"[{group}] {total} files -> {dest_dir.relative_to(DATA_DIR.parent)} (source={source})", flush=True)
    for i, e in enumerate(files, 1):
        out = dest_dir / e["name"]
        rel = str(out.relative_to(DATA_DIR))
        digest: str
        if out.exists() and out.stat().st_size == e["size"]:
            # Present at the right size: trust only if it also hashes correctly.
            digest = _sha256_file(out)
            if recorded.get(rel) in (None, digest):
                skipped += 1
            else:
                # On-disk bytes disagree with what the manifest recorded — do
                # not silently keep them.
                raise SystemExit(
                    f"HASH MISMATCH for {rel}: on disk {digest}, "
                    f"manifest {recorded[rel]}. Refusing to proceed; "
                    "delete the file and re-fetch, or investigate tampering.")
        else:
            url = _url_for(group, e["rel_path"], source)
            data = _get(url)
            digest = _sha256(data)
            expected = recorded.get(rel)
            if expected is not None and expected != digest:
                raise SystemExit(
                    f"HASH MISMATCH downloading {rel} from {source}: got "
                    f"{digest}, manifest expected {expected}. Hard stop.")
            out.write_bytes(data)
            got += 1
            print(f"[{group}] {i}/{total} {e['name']} ({e['size']/1e6:.1f} MB) sha256={digest[:12]}…", flush=True)
        records.append({
            "group": group,
            "file": rel,
            "size_bytes": e["size"],
            "sha256": digest,
            "source": source,
            "github_url": _url_for(group, e["rel_path"], "github"),
            "hf_url": None if group == "mcp" else _url_for(group, e["rel_path"], "hf"),
        })
    print(f"[{group}] done: {got} fetched, {skipped} present & hash-verified", flush=True)


def verify() -> int:
    """Re-hash every file in the manifest against its recorded SHA-256.

    A missing file or a mismatch is a HARD STOP (exit 1) — never a warning.
    """
    if not MANIFEST.exists():
        print("no data/MANIFEST.json; nothing to verify (run a fetch first)", file=sys.stderr)
        return 1
    m = json.loads(MANIFEST.read_text())
    bad: list[str] = []
    checked = 0
    for f in m.get("files", []):
        rel, expected = f["file"], f.get("sha256")
        path = DATA_DIR / rel
        if not path.exists():
            bad.append(f"MISSING  {rel}")
            continue
        if expected is None:
            bad.append(f"NO HASH RECORDED  {rel}")
            continue
        actual = _sha256_file(path)
        checked += 1
        if actual != expected:
            bad.append(f"MISMATCH {rel}\n    expected {expected}\n    actual   {actual}")
    if bad:
        print(f"VERIFY FAILED ({len(bad)} problem(s)):", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 1
    print(f"verify OK: {checked} files match their recorded SHA-256")
    return 0


def write_provenance(records: list[dict], groups: list[str], source: str) -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetched_via": source,
        "pins": {
            "mirror": {"repo": MIRROR_REPO, "sha": MIRROR_SHA,
                       "verified_against_live_head_utc": "2026-09-20"},
            "match_charting": {"repo": MCP_REPO, "sha": MCP_SHA},
            "hugging_face_fallback": {"repo": HF_REPO, "revision": HF_REVISION,
                                      "trust": "bytes only, verified against sha256"},
        },
        "sources": {g: SOURCES[g] for g in groups},
        "provenance_notes": [
            "Mirror is an archive, not a fork: no shared git history with "
            "upstream, so contents cannot be diffed against the original repos. "
            "Integrity rests on the per-file sha256 below, not on lineage.",
            "atp/wta provenance is weak: the mirror did not record an upstream "
            "commit SHA for the June 2026 snapshot. Treat as a known trust risk.",
            "Keep an independent cold copy; do not assume the mirror persists.",
            "Dataset is FROZEN: slam point-by-point ends Oct 2024; there will be "
            "no 2025-26 slam data. The Stage 0 train-early/test-late split is the "
            "only out-of-sample period this project will ever have.",
        ],
        "groups_fetched": groups,
        "file_count": len(records),
        "total_bytes": sum(r["size_bytes"] for r in records),
        "files": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    (DATA_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION_TXT)
    (DATA_DIR / "DATA_LICENSE.txt").write_text(DATA_LICENSE_TXT)
    print(f"wrote data/MANIFEST.json ({len(records)} files, "
          f"{manifest['total_bytes']/1e6:.0f} MB, sha256 per file), "
          f"ATTRIBUTION.txt, DATA_LICENSE.txt")


ATTRIBUTION_TXT = """\
Tennis data in this directory was compiled by Jeff Sackmann and is used here
under CC BY-NC-SA 4.0 (non-commercial; attribution required; share-alike). The
license grant is irrevocable: upstream withdrawal of the repositories does not
affect the grant on copies already distributed, so this use is fine.

Provenance origins (now-dead upstream repositories, linked per the attribution
requirement):
  Grand Slam point-by-point  https://github.com/JeffSackmann/tennis_slam_pointbypoint  (404)
  ATP                        https://github.com/JeffSackmann/tennis_atp                (404)
  WTA                        https://github.com/JeffSackmann/tennis_wta                (404)
  Match Charting Project     https://github.com/JeffSackmann/tennis_MatchChartingProject (live)

slam/atp/wta were retrieved from the archival mirror
Aneeshers/tennis-sackmann-archive (an archive, not a fork: no shared git history
with upstream); the Match Charting Project directly from JeffSackmann. Pinned
commit SHAs and a per-file SHA-256 are in MANIFEST.json. atp/wta carry no
recorded upstream commit SHA (June 2026 snapshot) — weaker provenance.

This data is NOT committed to git and must not be redistributed except under the
same CC BY-NC-SA 4.0 license. Use is non-commercial only; NonCommercial excludes
carrying this data into commercial research.
"""

DATA_LICENSE_TXT = """\
The data files in this directory are licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)
by Jeff Sackmann.

Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
Summary:           https://creativecommons.org/licenses/by-nc-sa/4.0/

You are free to share and adapt the material under these terms:
  Attribution   — credit Jeff Sackmann as compiler and link the original
                  (now-dead) source repositories as provenance origins.
  NonCommercial — not for commercial purposes; this excludes carrying the data
                  into commercial research.
  ShareAlike    — any redistributed data or derived dataset carries the same
                  CC BY-NC-SA 4.0 license.

The grant is irrevocable: the upstream repositories being removed does not
revoke the license on copies already distributed.

This license covers the DATA only. Project source code is licensed separately;
see the repository LICENSE / README.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="*", default=list(GROUPS),
                    help=f"groups to fetch (default: all -> {', '.join(GROUPS)})")
    ap.add_argument("--list", action="store_true",
                    help="show the file plan per group without downloading")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash local files against MANIFEST (hard stop on mismatch)")
    ap.add_argument("--source", choices=("github", "hf"), default="github",
                    help="download host: github mirror (default) or hugging face fallback")
    args = ap.parse_args(argv)

    if args.verify:
        return verify()

    groups = args.groups or list(GROUPS)
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        ap.error(f"unknown group(s): {', '.join(unknown)}; choose from {', '.join(GROUPS)}")
    if args.source == "hf" and "mcp" in groups:
        ap.error("mcp has no Hugging Face fallback; fetch it with --source github")

    if args.list:
        for g in groups:
            files = plan(g)
            mb = sum(f["size"] for f in files) / 1e6
            print(f"[{g}] {len(files)} files, {mb:.1f} MB")
            for f in files:
                print(f"    {f['name']} ({f['size']/1e6:.2f} MB)")
        return 0

    recorded = _load_recorded_hashes()
    records: list[dict] = []
    for g in groups:
        fetch_group(g, records, args.source, recorded)
    write_provenance(records, groups, args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
