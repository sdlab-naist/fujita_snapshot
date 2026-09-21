"""
common.py — utilities shared across the RQs

Collects the shared foundational code for the reproduction of the snapshot-testing empirical study
(the effectiveness/adoption of snapshot testing in OSS).

Main responsibilities:
  1. Collect target projects from GitHub (RQ1: star>=1000, RQ2/3: star>=500)
  2. Detect Jest usage by parsing package.json
  3. Detect snapshot testing via static analysis of test code (counts of test/it, expect,
     toMatchSnapshot / toMatchInlineSnapshot)
  4. Classify projects as ST / UT / UT+ST
  5. Measure source lines of code (LOC)
  6. Statistical helpers (Mann-Whitney U + Bonferroni correction)

So that each RQ script runs even without a network or a Node.js environment, this provides both the
real-data collection pipeline (collect_*) and synthetic data generation that reproduces the paper's
reported values (synth_*). To use real data, set the GITHUB_TOKEN environment variable.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.stats import mannwhitneyu
except Exception:  # allow the import to succeed even without scipy
    mannwhitneyu = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ProjectMetrics:
    """Measurement result for one repository."""

    name: str
    stars: int = 0
    uses_jest: bool = False
    loc: int = 0                  # total source lines of code
    test_cases: int = 0          # total number of test() / it()
    assertions: int = 0          # total number of expect() (= number of assertions)
    snapshot_tests: int = 0      # number of tests that contain toMatchSnapshot / toMatchInlineSnapshot
    unit_tests: int = 0          # number of tests that do not contain a snapshot
    test_list_source: str = ""   # how tests were enumerated: "listTests" | "static" | ""

    @property
    def uses_snapshot(self) -> bool:
        return self.snapshot_tests > 0

    @property
    def category(self) -> str:
        """Classify as ST / UT / UT+ST.

        - UT    : has no snapshot test at all (unit tests only)
        - ST    : all tests are snapshot tests (has no unit test)
        - UT+ST : uses both
        """
        if self.snapshot_tests == 0:
            return "UT"
        if self.unit_tests == 0:
            return "ST"
        return "UT+ST"

    # --- per-1K-LOC normalized metrics (definition of paper 3.2.2) -------
    @property
    def test_cases_per_kloc(self) -> float:
        return self.test_cases / (self.loc / 1000.0) if self.loc else 0.0

    @property
    def snapshot_tests_per_kloc(self) -> float:
        return self.snapshot_tests / (self.loc / 1000.0) if self.loc else 0.0

    @property
    def unit_tests_per_kloc(self) -> float:
        return self.unit_tests / (self.loc / 1000.0) if self.loc else 0.0

    @property
    def assertions_per_test(self) -> float:
        return self.assertions / self.test_cases if self.test_cases else 0.0


# ---------------------------------------------------------------------------
# Static analysis: Jest detection / snapshot detection
# ---------------------------------------------------------------------------
# Jest's snapshot assertion methods (paper 2.3.4)
SNAPSHOT_MATCHERS = ("toMatchSnapshot", "toMatchInlineSnapshot")

# test()/it() calls. Also catch qualifiers like test.skip / it.only.
_TEST_DECL_RE = re.compile(r"\b(?:test|it)(?:\.\w+)?\s*\(")
_EXPECT_RE = re.compile(r"\bexpect\s*\(")
_SNAPSHOT_RE = re.compile(r"\b(?:%s)\s*\(" % "|".join(SNAPSHOT_MATCHERS))
# a jest command in the value of scripts (paper 3.2.1 / 4.2.1)
_JEST_CMD_RE = re.compile(r"\bjest\b")


def detect_jest_in_package_json(pkg_text: str) -> bool:
    """Given package.json text, decide whether it is a Jest project.

    Strictly following paper 3.2.1 / 4.2.1, decide **only by whether the "scripts" section contains
    at least one "jest" command**. The paper does not use declarations in dependencies /
    devDependencies as the criterion (to exclude repositories that have jest as a dependency but do
    not actually run with Jest), so here too we look only at scripts.
    """
    try:
        pkg = json.loads(pkg_text)
    except (json.JSONDecodeError, TypeError):
        return False
    scripts = pkg.get("scripts") or {}
    if not isinstance(scripts, dict):
        return False
    return any(_JEST_CMD_RE.search(str(v)) for v in scripts.values())


def analyze_test_source(source: str) -> dict:
    """Analyze the source string of a single test file (regex fallback).

    A substitute for when the Babel AST (the paper's implementation) is unavailable. Counts
    occurrences of test/it, expect, and snapshot matchers.

    Returns: {"test_cases", "assertions", "snapshot_matchers"}
    """
    # strip line/block comments to reduce false positives
    src = re.sub(r"//.*", "", source)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return {
        "test_cases": len(_TEST_DECL_RE.findall(src)),
        "assertions": len(_EXPECT_RE.findall(src)),
        "snapshot_matchers": len(_SNAPSHOT_RE.findall(src)),
    }


# --- Babel AST analysis (faithful to the paper's implementation) -----------
_TOOLS_DIR = Path(__file__).resolve().parent / "tools"
_AST_SCRIPT = _TOOLS_DIR / "analyze_tests.js"
_ast_available: bool | None = None


def ast_available() -> bool:
    """Decide once whether Node + analyze_tests.js + @babel are available."""
    global _ast_available
    if _ast_available is not None:
        return _ast_available
    import shutil
    import subprocess

    if shutil.which("node") is None or not _AST_SCRIPT.exists():
        _ast_available = False
        return False
    try:
        # smoke-test with a simple source (returns fatal if deps are missing)
        proc = subprocess.run(
            ["node", str(_AST_SCRIPT), "--stdin"],
            input="test('x', () => { expect(1).toMatchSnapshot(); });",
            capture_output=True, text=True, timeout=30,
        )
        out = json.loads(proc.stdout or "{}")
        _ast_available = "fatal" not in out and "snapshot_tests" in out
    except Exception:
        _ast_available = False
    return _ast_available


def analyze_repo_tests_ast(test_files: list[Path]) -> dict | None:
    """Analyze multiple test files at once with a Babel AST and return the totals.

    Returns: {"test_cases","assertions","snapshot_tests","unit_tests"} or None if the AST is unavailable.
    """
    if not test_files or not ast_available():
        return None
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(str(p) for p in test_files))
        list_path = tf.name
    try:
        proc = subprocess.run(
            ["node", str(_AST_SCRIPT), "--files-from", list_path],
            capture_output=True, text=True, timeout=600,
        )
        data = json.loads(proc.stdout or "{}")
        return data.get("totals")
    except Exception:
        return None
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


def _split_test_blocks(source: str) -> list[str]:
    """Roughly split the source into test()/it() blocks.

    Used to decide whether each test block contains a snapshot matcher and to count
    snapshot_tests / unit_tests. A simple parser that counts brace matching to carve out the body
    range of a single test() call.
    """
    src = re.sub(r"//.*", "", source)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    blocks: list[str] = []
    for m in _TEST_DECL_RE.finditer(src):
        i = m.end()  # right after '('
        depth_paren = 1
        depth_brace = 0
        start = i
        while i < len(src) and depth_paren > 0:
            c = src[i]
            if c == "(":
                depth_paren += 1
            elif c == ")":
                depth_paren -= 1
            elif c == "{":
                depth_brace += 1
            elif c == "}":
                depth_brace -= 1
            i += 1
        blocks.append(src[start:i])
    return blocks


def classify_tests(source: str) -> tuple[int, int]:
    """Return (snapshot_tests, unit_tests) from one file.

    Classify each test block by whether it contains toMatchSnapshot/toMatchInlineSnapshot.
    """
    snapshot = unit = 0
    for block in _split_test_blocks(source):
        if _SNAPSHOT_RE.search(block):
            snapshot += 1
        else:
            unit += 1
    return snapshot, unit


def count_loc(source: str) -> int:
    """Count executable lines (LOC) excluding blank and comment lines."""
    src = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    loc = 0
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        loc += 1
    return loc


# ---------------------------------------------------------------------------
# Analysis of a local repository (targeting a cloned directory)
# ---------------------------------------------------------------------------
SRC_GLOBS = ("*.js", "*.jsx", "*.ts", "*.tsx")
CODE_EXTS = {".js", ".jsx", ".ts", ".tsx"}
# Equivalent to Jest's default testMatch `**/?(*.)+(spec|test).[jt]s?(x)`.
# Since `?(*.)` is optional, note that bare test.js / spec.ts are also test files
# (e.g. vercel/arg puts all tests in a root test.js).
TEST_FILE_RE = re.compile(r"(?:^|\.)(?:test|spec)\.[jt]sx?$")


def is_test_file(path: str | Path) -> bool:
    """Decide whether a file is a test file, equivalent to Jest's default testMatch.

    - the file name is *.test.* / *.spec.* or bare test.* / spec.* (.js/.jsx/.ts/.tsx)
    - or the file is under a __tests__ directory
    For repositories with a custom testMatch / testRegex, use jest_test_matcher().
    """
    p = Path(path)
    if TEST_FILE_RE.search(p.name):
        return True
    return "__tests__" in p.parts


def is_production_code(path: str | Path, repo: str | Path) -> bool:
    """Follow the production-code definition of paper 3.2.2.

    Treats "a code file whose file name or directory name does not contain 'test'" as production
    code (= the denominator for the per-1K-LOC normalization). Exclude it if any component of the
    repo-relative path (directory name / file name) contains "test".
    """
    repo = Path(repo)
    try:
        rel = Path(path).relative_to(repo)
    except ValueError:
        rel = Path(Path(path).name)
    return not any("test" in part.lower() for part in rel.parts)


# ---------------------------------------------------------------------------
# Test-file decision based on the Jest config (testMatch / testRegex)
# ---------------------------------------------------------------------------
# To approximate `jest --listTests` of paper 4.2.1, read the repository's Jest config and resolve the
# set of test files. Relying only on the is_test_file naming convention would miss tests with a
# custom testRegex (e.g. greenkeeper's /test/.*\.js$) or testMatch (e.g. node-wpapi's
# **/tests/**/*.js), misclassifying such repositories as "UT with 0 tests" (distorting the RQ1 UT median).

def _glob_to_regex(pat: str) -> str:
    """A simple translator from a micromatch-style glob (Jest testMatch) to a regex.

    Supports: `**/` `**` `*` `?` `[...]`, extglob `?( )` `+( )` `*( )` `@( )`,
    micromatch alternation groups `(a|b|c)`, and brace expansion `{a,b,c}`.
    This is sufficient to handle Jest's default testMatch and common custom values.

    Note: without treating bare `(a|b)` (e.g. `*.test.(ts|tsx|js)`) and `{a,b}` as alternations, we
    would produce a regex requiring that literal string and miss test files (which was a cause of
    distortion in the RQ1 UT density).
    """
    def _split_top(s: str, sep: str) -> list[str]:
        parts, depth, start = [], 0, 0
        for k, ch in enumerate(s):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == sep and depth == 0:
                parts.append(s[start:k])
                start = k + 1
        parts.append(s[start:])
        return parts

    def _find_close(s: str, start: int, open_c: str, close_c: str) -> int:
        depth, j = 1, start
        while j < len(s) and depth:
            if s[j] == open_c:
                depth += 1
            elif s[j] == close_c:
                depth -= 1
            j += 1
        return j  # position after the closing bracket

    out: list[str] = []
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        if c in "?+*@" and i + 1 < n and pat[i + 1] == "(":
            j, depth = i + 2, 1
            while j < n and depth:
                if pat[j] == "(":
                    depth += 1
                elif pat[j] == ")":
                    depth -= 1
                j += 1
            # split the inside on depth-0 "|" and recursively convert each
            inner = pat[i + 2:j - 1]
            parts, depth2, start = [], 0, 0
            for k, ch in enumerate(inner):
                if ch == "(":
                    depth2 += 1
                elif ch == ")":
                    depth2 -= 1
                elif ch == "|" and depth2 == 0:
                    parts.append(inner[start:k])
                    start = k + 1
            parts.append(inner[start:])
            body = "|".join(_glob_to_regex(p) for p in parts)
            out.append("(?:%s)%s" % (body, {"?": "?", "+": "+", "*": "*", "@": ""}[c]))
            i = j
        elif c == "(":
            # micromatch alternation group `(a|b|c)`
            j = _find_close(pat, i + 1, "(", ")")
            inner = pat[i + 1:j - 1]
            body = "|".join(_glob_to_regex(p) for p in _split_top(inner, "|"))
            out.append("(?:%s)" % body)
            i = j
        elif c == "{":
            # brace expansion `{a,b,c}`
            j = _find_close(pat, i + 1, "{", "}")
            inner = pat[i + 1:j - 1]
            body = "|".join(_glob_to_regex(p) for p in _split_top(inner, ","))
            out.append("(?:%s)" % body)
            i = j
        elif pat.startswith("**/", i):
            out.append(r"(?:[^/]+/)*")
            i += 3
        elif pat.startswith("**", i):
            out.append(r".*")
            i += 2
        elif c == "*":
            out.append(r"[^/]*")
            i += 1
        elif c == "?":
            out.append(r"[^/]")
            i += 1
        elif c == "[":
            j = pat.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                out.append(pat[i:j + 1])
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


_JEST_CONFIG_KEYS = ("testMatch", "testRegex", "testPathIgnorePatterns")


def _jest_config_from_node(path: Path) -> dict | None:
    """require jest.config.{js,cjs} with node and return the config dict (None on failure)."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        return None
    script = (
        "const c = require(process.argv[1]);"
        "const r = (c && c.default) || c;"
        "if (typeof r !== 'object' || r === null) process.exit(1);"
        "const keys = %s;"
        "const o = {}; for (const k of keys) if (r[k] !== undefined) o[k] = r[k];"
        "console.log(JSON.stringify(o));" % json.dumps(list(_JEST_CONFIG_KEYS))
    )
    try:
        proc = subprocess.run(["node", "-e", script, str(path)],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def _jest_config_from_text(path: Path) -> dict | None:
    """Extract testMatch / testRegex from a config file's source text via regex.

    A fallback for when it cannot be evaluated with node (preset require failure, .ts/.mjs, etc.).
    Supports only string-literal arrays/singletons.
    """
    try:
        src = path.read_text(errors="ignore")
    except OSError:
        return None
    cfg: dict = {}
    m = re.search(r"testRegex\s*:\s*(['\"])((?:\\.|(?!\1).)*)\1", src)
    if m:
        cfg["testRegex"] = m.group(2)
    m = re.search(r"testMatch\s*:\s*\[(.*?)\]", src, flags=re.S)
    if m:
        vals = re.findall(r"(['\"])((?:\\.|(?!\1).)*)\1", m.group(1))
        if vals:
            cfg["testMatch"] = [v for _, v in vals]
    return cfg or None


def load_jest_config(repo_dir: str | Path) -> dict:
    """Resolve the repository's Jest config (testMatch/testRegex etc.).

    Priority: jest.config.{js,cjs,json,mjs,ts} -> the "jest" field of package.json.
    If none exists, an empty dict (= Jest default).
    """
    repo = Path(repo_dir)
    for fname in ("jest.config.js", "jest.config.cjs", "jest.config.json",
                  "jest.config.mjs", "jest.config.ts"):
        p = repo / fname
        if not p.exists():
            continue
        if fname.endswith(".json"):
            try:
                cfg = json.loads(p.read_text(errors="ignore"))
                if isinstance(cfg, dict):
                    return cfg
            except (json.JSONDecodeError, OSError):
                pass
            continue
        cfg = None
        if fname.endswith((".js", ".cjs")):
            cfg = _jest_config_from_node(p)
        if cfg is None:
            cfg = _jest_config_from_text(p)
        if cfg:
            return cfg
    pkg = repo / "package.json"
    if pkg.exists():
        try:
            jest = json.loads(pkg.read_text(errors="ignore")).get("jest")
            if isinstance(jest, dict):
                return jest
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def jest_test_matcher(repo_dir: str | Path):
    """Return a predicate `f(path) -> bool` that follows the repo's Jest config.

    - if there is a testRegex, re.search against "/repo/" + the repo-relative POSIX path
      (Jest applies it to absolute paths; because configs like `./tests/...` that write a leading `.`
       as any single character exist in the wild, we approximate by prepending a dummy parent dir)
    - if there is a testMatch, apply the glob to the repo-relative POSIX path
    - if neither, equivalent to the Jest default (is_test_file)
    - exclude paths matching testPathIgnorePatterns (node_modules is already excluded by the caller)
    """
    repo = Path(repo_dir)
    cfg = load_jest_config(repo)

    test_regexes: list[re.Pattern] = []
    raw = cfg.get("testRegex")
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for r in raw:
            if not isinstance(r, str):
                continue
            try:
                test_regexes.append(re.compile(r))
            except re.error:
                pass

    match_regexes: list[re.Pattern] = []
    tm = cfg.get("testMatch")
    if isinstance(tm, list):
        for g in tm:
            if not isinstance(g, str):
                continue
            g = g.replace("<rootDir>/", "").replace("<rootDir>", "")
            try:
                match_regexes.append(re.compile("^" + _glob_to_regex(g) + "$"))
            except re.error:
                pass

    ignore_regexes: list[re.Pattern] = []
    ipp = cfg.get("testPathIgnorePatterns")
    if isinstance(ipp, list):
        for pat in ipp:
            if not isinstance(pat, str):
                continue
            pat = pat.replace("<rootDir>/", "/").replace("<rootDir>", "")
            try:
                ignore_regexes.append(re.compile(pat))
            except re.error:
                pass

    def match(path: str | Path) -> bool:
        try:
            rel = Path(path).relative_to(repo).as_posix()
        except ValueError:
            rel = Path(path).name
        slashed = "/repo/" + rel
        if any(rx.search(slashed) for rx in ignore_regexes):
            return False
        if test_regexes:
            return any(rx.search(slashed) for rx in test_regexes)
        if match_regexes:
            return any(rx.match(rel) for rx in match_regexes)
        return is_test_file(path)

    return match


def analyze_repo_dir(repo_dir: str | Path, name: str | None = None,
                     use_ast: bool = True,
                     test_files: list[Path] | None = None) -> ProjectMetrics:
    """Walk a cloned repository directory and build a ProjectMetrics.

    If use_ast=True and Node+Babel are available, analyze the test code strictly with an AST
    (faithful to the paper's implementation). Otherwise, switch to the regex fallback.

    If test_files is passed, treat that set as the confirmed test files (e.g. the result of
    `npx jest --listTests`). If None, enumerate via static resolution of the Jest config.
    """
    repo = Path(repo_dir).resolve()
    name = name or repo.name
    pm = ProjectMetrics(name=name)

    pkg = repo / "package.json"
    if pkg.exists():
        pm.uses_jest = detect_jest_in_package_json(pkg.read_text(errors="ignore"))

    explicit: set[Path] | None = None
    if test_files is not None:
        explicit = {Path(p).resolve() for p in test_files}
        pm.test_list_source = "listTests"
        is_test = None
    else:
        # test-file decision (RQ1 static enumeration / fallback when RQ2/RQ3 listTests fails).
        # To avoid misses, use the union of the following as test candidates:
        #   (1) path contains "test"  ... the complement of the production-code definition of §3.2.2.
        #       Catches repos that put tests in a `test/` directory (e.g. swagger-js).
        #   (2) default naming *.test.* / *.spec.* / __tests__ ... (1) is the "test" substring, so it
        #       misses *.spec.* (which does not contain "test"); this supplements it.
        #   (3) custom testMatch/testRegex (jest_test_matcher) ... catches non-standard layouts.
        # Because the AST side counts only test/it methods, even if a non-test file named "test"
        # sneaks in, it does not affect the test-case count (it was excluded from LOC to begin with).
        pm.test_list_source = "static" if pm.uses_jest else ""
        if pm.uses_jest:
            _custom = jest_test_matcher(repo)
            is_test = lambda p: (not is_production_code(p, repo)
                                 or is_test_file(p) or _custom(p))
        else:
            is_test = lambda p: (not is_production_code(p, repo) or is_test_file(p))

    found_tests: list[Path] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        if "node_modules" in path.parts or ".git" in path.parts:
            continue
        if path.suffix.lower() not in CODE_EXTS:
            continue
        if (path in explicit) if explicit is not None else is_test(path):
            found_tests.append(path)
        elif is_production_code(path, repo):
            try:
                pm.loc += count_loc(path.read_text(errors="ignore"))
            except OSError:
                continue
        # anything else (e.g. src/testUtils.js, which is not a test but has "test" in the name) is
        # neither a test nor production code, so exclude it from LOC.
    test_files = found_tests

    # analyze the test code: prefer AST, fall back to regex on failure
    ast_totals = analyze_repo_tests_ast(test_files) if use_ast else None
    if ast_totals is not None:
        pm.test_cases = ast_totals["test_cases"]
        pm.assertions = ast_totals["assertions"]
        pm.snapshot_tests = ast_totals["snapshot_tests"]
        pm.unit_tests = ast_totals["unit_tests"]
    else:
        for path in test_files:
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            a = analyze_test_source(text)
            pm.test_cases += a["test_cases"]
            pm.assertions += a["assertions"]
            s, u = classify_tests(text)
            pm.snapshot_tests += s
            pm.unit_tests += u

    return pm


# ---------------------------------------------------------------------------
# GitHub collection pipeline (for real data; requires GITHUB_TOKEN)
# ---------------------------------------------------------------------------
def search_candidate_repos(min_stars: int, max_repos: int = 1000,
                           language: str = "JavaScript") -> list[dict]:
    """Fetch a list of non-fork JS/TS repositories via the GitHub Search API.

    In the paper, RQ1 uses star>=1000 and RQ2/RQ3 use star>=500.
    Requires requests and GITHUB_TOKEN. Do not call it when offline.
    """
    import requests  # lazy import

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("the GITHUB_TOKEN environment variable is required")
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}

    repos: list[dict] = []
    page = 1
    query = f"language:{language} stars:>={min_stars} fork:false"
    while len(repos) < max_repos and page <= 10:  # the API returns at most 1000
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc",
                    "per_page": 100, "page": page},
            headers=headers, timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            break
        repos.extend(items)
        page += 1
        time.sleep(2)  # rate-limit safeguard
    return repos[:max_repos]


class RepoNotYetExisting(RuntimeError):
    """A repository with no commit before the CLONE_BEFORE date (= it did not exist yet then)."""


def clone_repo(clone_url: str, dest: str | Path) -> Path:
    """Clone a repository.

    By default a shallow clone (depth=1, current HEAD).
    If the environment variable ``CLONE_BEFORE`` (an ISO date, e.g. "2025-07-01") is set, check out
    the latest commit before that date (to reproduce the paper's collection time).
    A repository with no commit before the given date raises ``RepoNotYetExisting``
    (it did not exist yet -> should be excluded from the population).
    """
    import subprocess

    dest = Path(dest)
    # reuse only a clone that already has contents (a resume cache). Because the caller may pass an
    # "empty existing directory" from tempfile.mkdtemp(), a plain existence check would skip the
    # clone (git can clone into an empty existing dir).
    if dest.exists() and any(dest.iterdir()):
        return dest

    before = os.environ.get("CLONE_BEFORE", "").strip()
    if not before:
        subprocess.run(["git", "clone", "--depth", "1", clone_url, str(dest)],
                       check=True, capture_output=True)
        return dest

    # --- pin to a past point in time ---
    # a blobless partial clone fetches the full history (commits/trees) fast and lazily fetches blobs.
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    clone_url, str(dest)], check=True, capture_output=True)
    cutoff = before if "T" in before else before + "T23:59:59"
    sha = subprocess.run(
        ["git", "-C", str(dest), "rev-list", "-1", f"--before={cutoff}", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if not sha:
        raise RepoNotYetExisting(f"{clone_url}: no commit before {before}")
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", sha],
                   check=True, capture_output=True)
    cdate = subprocess.run(
        ["git", "-C", str(dest), "show", "-s", "--format=%cI", sha],
        check=True, capture_output=True, text=True).stdout.strip()
    print(f"[pin] {clone_url} -> {sha[:12]} ({cdate})", flush=True)
    return dest


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def mann_whitney(a: Iterable[float], b: Iterable[float]) -> tuple[float, float]:
    """Mann-Whitney U test (two-sided). Returns (U statistic, p-value)."""
    if mannwhitneyu is None:
        raise RuntimeError("scipy is required (pip install scipy)")
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def bonferroni(p_values: list[float], alpha: float = 0.05,
               num_comparisons: int | None = None) -> list[bool]:
    """Bonferroni correction. Compare each p to alpha/num_comparisons; True if significant.

    If num_comparisons is omitted, use the number of comparisons = len(p_values).

    Note: the paper reports alpha=0.005 as the **significance level after Bonferroni correction**.
    To use that value directly as the threshold, pass num_comparisons=1 (dividing alpha by the number
    of comparisons again would be a double correction and yield a stricter threshold than the paper).
    """
    m = num_comparisons if num_comparisons is not None else len(p_values)
    if m <= 0:
        return []
    threshold = alpha / m
    return [p < threshold for p in p_values]


def describe(values: Iterable[float]) -> dict:
    """Return summary statistics: median, quartiles, mean, etc."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------
def save_metrics(metrics: list[ProjectMetrics], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(m) for m in metrics],
                                     ensure_ascii=False, indent=2))


def load_metrics(path: str | Path) -> list[ProjectMetrics]:
    data = json.loads(Path(path).read_text())
    return [ProjectMetrics(**{k: v for k, v in d.items()
                              if k in ProjectMetrics.__dataclass_fields__})
            for d in data]


# ---------------------------------------------------------------------------
# Research integrity: separating synthetic fixtures from real-data collection
# ---------------------------------------------------------------------------
SYNTH_BANNER = (
    "\n" + "!" * 70 + "\n"
    "!! SYNTHETIC FIXTURE — this is NOT a reproduction 'result'\n"
    "!! Purpose: pseudo data to check that the pipeline runs without a network/Node\n"
    "!!          environment. It is generated to match the paper's medians, so it\n"
    "!!          must not be interpreted as a reproduction result.\n"
    "!! For an actual reproduction, run --collect and report the measured values from\n"
    "!! the GitHub population at that time (it is normal for them not to match the paper).\n"
    + "!" * 70 + "\n"
)

# Main reference values used to generate synthetic fixtures (--demo) --------
REPORTED = {
    "rq1": {
        "js_ts_projects": 9516,
        "jest_projects": 1487,
        "jest_ratio": 0.156,           # 1487 / 9516
        "snapshot_projects": 569,
        "snapshot_adoption": 0.383,    # 569 / 1487
        "test_cases_per_kloc_median": {"UT+ST": 25.2, "UT": 16.2, "ST": 7.2},
        "unit_tests_median": {"UT": 16.2, "UT+ST": 16.7},
        "snapshot_tests_median": {"ST": 7.2, "UT+ST": 2.1},
        "assertions_per_test_median": {"UT": 1.62, "UT+ST": 1.7, "ST": 1.0},
        "mwu_p_ut_vs_utst_testcases": 0.23,
        "ratio_testcases": 1.6,        # 25.2 / 16.2
        "alpha_bonferroni": 0.005,
    },
    "rq2": {
        "n_projects": 283,
        "categories": ["All", "Snapshot", "Non-Snapshot",
                       "Only by Snapshot", "Only by Non-Snapshot"],
        "median_coverage": {
            "All": 85.4, "Snapshot": 62.0, "Non-Snapshot": 68.0,
            "Only by Snapshot": 3.8, "Only by Non-Snapshot": 8.9,
        },
    },
    "rq3": {
        "n_projects": 48,
        "median_mutation": {
            "All": 45.9, "Snapshot": 18.4, "Non-Snapshot": 26.8,
        },
    },
}


if __name__ == "__main__":
    # a simple self-test
    sample = """
    // sample.test.js
    describe('add', () => {
      test('snapshot', () => { expect(render()).toMatchSnapshot(); });
      it('unit', () => { expect(add(1,2)).toBe(3); });
    });
    """
    print("analyze:", analyze_test_source(sample))
    print("classify (snap, unit):", classify_tests(sample))
    # paper-compliant: decide by whether scripts has a jest command (not by a dependency declaration alone)
    print("jest(scripts):", detect_jest_in_package_json('{"scripts": {"test": "jest --ci"}}'))
    print("jest(deps only):", detect_jest_in_package_json('{"devDependencies": {"jest": "^29"}}'))
