"""
repo_collection.py — collect the RQ1 target repositories (paper 3.2.1-compliant)

[The paper's actual method — corrected]
This paper does *not* start from the Bui & Rocha [33] dataset. In Section 3.2.1, because GitHub tag
declarations are incomplete, the authors collect repositories independently:

  "Instead of using the 'jest' tag, we used GitHub Search [36] to obtain a list of non-fork
   JavaScript or TypeScript repositories with 1,000+ stars, yielding 9,516 repositories. We then
   parsed package.json to decide whether Jest was used."

  [36] = Ozren Dabic, Emad Aghajani, Gabriele Bavota.
         "Sampling projects in GitHub for MSR studies." MSR'21.
         -> SEART GitHub Search Engine (https://seart-ghs.si.usi.ch)

So the RQ1 population (9,516) can be obtained either way:
  (A) load the SEART GitHub Search CSV export  ... load_seart_csv()  [paper-compliant, recommended]
  (B) call the GitHub Search API directly      ... collect_via_github_search()
      (the official API caps at 1,000 results per query, so it is ill-suited to reproduce all 9,516)

[Role of Bui & Rocha [33] — optional]
  Emily Bui and Henrique Rocha. "Snapshot testing dataset." MSR'23, pp.558-562.
  A dataset of snapshot/JEST projects identified by GitHub's "jest" tag. This paper does not use it,
  but it can be loaded via load_seed_repos() (resolve_repo_list's seed_path) as input for *comparison*
  with a different method.
"""

from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path

FULLNAME_RE = re.compile(r"(?:github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def _normalize(identifier: str) -> str | None:
    """Normalize 'owner/name' / a URL / a '.git' suffix into 'owner/name'."""
    s = (identifier or "").strip().strip('"')
    if not s:
        return None
    m = FULLNAME_RE.search(s)
    if not m:
        return None
    full = m.group(1)
    return full if "/" in full else None


def _clone_url(full_name: str) -> str:
    return f"https://github.com/{full_name}.git"


# ===========================================================================
# (A) SEART GitHub Search CSV export [paper-compliant, recommended]
# ===========================================================================
# Load the SEART (https://seart-ghs.si.usi.ch) search-result CSV.
# Column names vary by version, so resolve known aliases case-insensitively.
_NAME_COLS = ("name", "full_name", "namewithowner", "repository", "repo")
_STAR_COLS = ("stargazers", "stargazers_count", "stars", "starscount", "starcount")
_FORK_COLS = ("isfork", "fork", "is_fork")
_LANG_COLS = ("mainlanguage", "language", "main_language")


def _pick(row: dict, aliases: tuple[str, ...]) -> str | None:
    lower = {k.lower(): k for k in row.keys()}
    for a in aliases:
        if a in lower:
            return row[lower[a]]
    return None


def load_seart_csv(path: str | Path, min_stars: int = 1000,
                   languages: tuple[str, ...] = ("JavaScript", "TypeScript"),
                   exclude_forks: bool = True) -> list[dict]:
    """Load a SEART GitHub Search export CSV and filter by the criteria.

    Returns: a list of {"full_name","clone_url","stargazers_count","language","fork"}.
    If metadata columns (star/fork/language) are present, filter within the CSV. If not, apply no
    filter and only list them (this can be supplemented later by enrich_and_filter).
    """
    repos: list[dict] = []
    seen: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full = _normalize(_pick(row, _NAME_COLS) or "")
            if not full or full in seen:
                continue

            stars_raw = _pick(row, _STAR_COLS)
            fork_raw = _pick(row, _FORK_COLS)
            lang_raw = _pick(row, _LANG_COLS)

            # filter only when the column exists (otherwise pass through without skipping)
            if stars_raw not in (None, "") and _to_int(stars_raw) < min_stars:
                continue
            if exclude_forks and _is_true(fork_raw):
                continue
            if lang_raw not in (None, "") and lang_raw not in languages:
                continue

            seen.add(full)
            repos.append({
                "full_name": full,
                "clone_url": _clone_url(full),
                "stargazers_count": _to_int(stars_raw) if stars_raw else None,
                "language": lang_raw,
                "fork": _is_true(fork_raw),
            })
    return repos


def _to_int(v) -> int:
    try:
        return int(float(str(v).replace(",", "")))
    except (ValueError, TypeError):
        return 0


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


# ===========================================================================
# (B) Call the GitHub Search API directly (official API; 1,000 results per query)
# ===========================================================================
def collect_via_github_search(min_stars: int = 1000, max_repos: int = 1000,
                              languages: tuple[str, ...] = ("JavaScript", "TypeScript")
                              ) -> list[dict]:
    """Collect non-fork JS/TS repositories via the GitHub Search API.

    Note: the official Search API returns at most 1,000 results per query, so use the SEART CSV
    (load_seart_csv) to reproduce all 9,516 of the paper.
    """
    import common

    repos: list[dict] = []
    for lang in languages:
        repos += common.search_candidate_repos(min_stars=min_stars,
                                                max_repos=max_repos, language=lang)
    return repos


# ===========================================================================
# Bui & Rocha [33] seed list (optional input for comparison)
# ===========================================================================
def load_seed_repos(path: str | Path) -> list[str]:
    """Load a seed-list file (txt/csv/json) and return a list of 'owner/name'.

    Input for using a different method's list (such as the "jest"-tag dataset of Bui & Rocha [33])
    for *comparison*. This is not the paper's main collection path.
    """
    import json

    path = Path(path)
    suffix = path.suffix.lower()
    raw: list[str] = []

    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("repositories") or data.get("items") or list(data.values())
        for item in data:
            if isinstance(item, str):
                raw.append(item)
            elif isinstance(item, dict):
                raw.append(item.get("full_name") or item.get("url")
                           or item.get("repository") or "")
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw.append(_pick(row, _NAME_COLS) or "")
    else:
        raw = path.read_text(errors="ignore").splitlines()

    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        full = _normalize(item)
        if full and full not in seen:
            seen.add(full)
            out.append(full)
    return out


def enrich_and_filter(full_names: list[str], min_stars: int = 1000,
                      languages: tuple[str, ...] = ("JavaScript", "TypeScript"),
                      exclude_forks: bool = True, sleep: float = 0.0) -> list[dict]:
    """Fetch metadata for a list of 'owner/name' via the GitHub API and filter by the criteria.

    Used to bring a seed list without metadata (Bui & Rocha etc.) into this study's criteria
    (non-fork, JS/TS, star>=min_stars). Requires GITHUB_TOKEN.
    """
    import requests

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("the GITHUB_TOKEN environment variable is required")
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}

    kept: list[dict] = []
    for full in full_names:
        try:
            resp = requests.get(f"https://api.github.com/repos/{full}",
                                headers=headers, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            r = resp.json()
        except Exception as e:
            print(f"skip {full}: {e}")
            continue
        if exclude_forks and r.get("fork"):
            continue
        if r.get("archived"):
            continue
        if r.get("language") not in languages:
            continue
        if (r.get("stargazers_count") or 0) < min_stars:
            continue
        kept.append({
            "full_name": r["full_name"],
            "clone_url": r["clone_url"],
            "stargazers_count": r.get("stargazers_count", 0),
            "language": r.get("language"),
            "fork": r.get("fork", False),
        })
        if sleep:
            time.sleep(sleep)
    return kept


# ===========================================================================
# JEST pre-filter (for RQ2/RQ3 batch collection; paper 4.2.1)
# ===========================================================================
def filter_jest_repos(repos: list[dict], scratch: str | None = None,
                      log=print) -> list[dict]:
    """Shallow-clone each repository and keep only those whose top-level package.json scripts
    contain a jest command (the pre-filter of paper 4.2.1).

    The RQ2/RQ3 runner.py would otherwise try npm ci + jest even on non-JEST repositories and count
    them as failures, so we narrow the set here first to save compute.
    """
    import tempfile
    import shutil
    import common

    kept: list[dict] = []
    for i, r in enumerate(repos, 1):
        tmp = tempfile.mkdtemp(dir=scratch)
        try:
            common.clone_repo(r["clone_url"], tmp)
            pkg = Path(tmp) / "package.json"
            if pkg.exists() and common.detect_jest_in_package_json(
                    pkg.read_text(errors="ignore")):
                kept.append(r)
        except Exception as e:
            log(f"skip {r['full_name']}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if i % 50 == 0:
            log(f"[jest-filter] scanned {i}/{len(repos)} / {len(kept)} are JEST")
    log(f"[jest-filter] {len(kept)}/{len(repos)} are JEST (jest in scripts)")
    return kept


# ===========================================================================
# Resolve the collection path
# ===========================================================================
def resolve_repo_list(seart_csv: str | None = None,
                      seed_path: str | None = None,
                      min_stars: int = 1000,
                      max_repos: int | None = None) -> list[dict]:
    """Decide the input repository list for RQ1 collection. Priority:

      1. seart_csv given -> load the SEART GitHub Search CSV (paper-compliant, recommended)
      2. seed_path given -> enrich a seed list from Bui & Rocha [33] etc. via the GitHub API (comparison)
      3. neither         -> the GitHub Search API directly (fallback, capped at 1,000)
    """
    if seart_csv:
        repos = load_seart_csv(seart_csv, min_stars=min_stars)
        print(f"{len(repos)} repos from SEART CSV [36] (star>={min_stars}/non-fork/JS-TS)")
    elif seed_path:
        full_names = load_seed_repos(seed_path)
        print(f"loaded {len(full_names)} repos from the seed list [33 etc.] (for comparison)")
        repos = enrich_and_filter(full_names, min_stars=min_stars)
        print(f"{len(repos)} after filtering")
    else:
        print("no collection path specified -> fallback to the GitHub Search API "
              "(capped at 1,000; use the SEART CSV to reproduce all 9,516)")
        repos = collect_via_github_search(min_stars=min_stars,
                                          max_repos=max_repos or 1000)
    if max_repos:
        repos = repos[:max_repos]
    return repos


if __name__ == "__main__":
    import tempfile

    d = tempfile.mkdtemp()
    # can a SEART-CSV-like input be parsed?
    csv_path = Path(d, "seart.csv")
    csv_path.write_text(
        "name,stargazers,isFork,mainLanguage\n"
        "facebook/jest,40000,false,TypeScript\n"
        "someone/fork-repo,5000,true,JavaScript\n"      # fork -> excluded
        "low/stars,10,false,JavaScript\n"               # too few stars -> excluded
        "vuejs/core,45000,false,TypeScript\n"
    )
    print("SEART CSV:", [r["full_name"] for r in load_seart_csv(csv_path)])
    # seed list (for comparison with Bui & Rocha)
    txt = Path(d, "seed.txt")
    txt.write_text("facebook/jest\nhttps://github.com/vuejs/core\nfacebook/jest\n")
    print("seed list:", load_seed_repos(txt))
