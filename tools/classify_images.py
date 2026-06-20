"""
Image Usefulness Classifier — GitHub-native post-processing step for extractor.py
=================================================================================

The extractor analyses repos in *report-only* mode and does not write a local
output tree. The organised dataset lives in a GitHub repository (e.g.
``https://github.com/<you>/Extracted-Files``). This script reads each project's
images straight from that repo via the GitHub API, decides which images are
*useful* for a hardware-design dataset (block diagrams, schematics, FSM/state
diagrams, waveforms, timing diagrams, floorplans, pinouts, datapaths, ...)
versus *not useful* (logos, badges/shields, icons, avatars, banners,
screenshots, photos, QR codes, ...), and **commits the result back into the
same repo** — moving every image into ``image_categories/useful/``,
``image_categories/non_useful/``, or ``image_categories/review/`` and writing
the verdicts into ``image_classification.json``.

Nothing is read from or written to a local folder. The commit is built with the
GitHub Git Data API as a single commit; moved images are re-pointed by their
existing blob SHA, so image bytes are never re-uploaded.

Classification is HYBRID:

  Stage 1 (offline, no model API):
    * Filename / path keyword rules (logo/badge/icon/... vs block/diagram/...).
    * Visual feature similarity (k-NN) to your *example images* — a reference
      set with ``useful/`` and ``not_useful/`` subfolders, supplied either as a
      local folder (``--reference-dir``) or another GitHub folder
      (``--reference-url``). Features: aspect ratio, colour count, colourfulness,
      grayscale-ness, edge density, white-background fraction, transparency.

  Stage 2 (optional, costs model-API tokens):
    * For images Stage 1 can't decide confidently, ask a Claude vision model
      (default: claude-haiku-4-5, the cheapest current model) for a verdict,
      optionally few-shot'd with a couple of your reference example images.

SVG images ARE classified: they are rasterised to PNG (via cairosvg, else
svglib+reportlab) before feature extraction / vision. Install one of those for
SVG support: ``pip install svglib`` (recommended on Windows) or
``pip install cairosvg``.

Auth: needs a GitHub token with write access to the dataset repo
(``--github-token`` or env ``GITHUB_TOKEN``). The vision fallback needs
``pip install anthropic`` and ``ANTHROPIC_API_KEY``.

Usage examples
--------------
    # Preview verdicts for one extracted project folder (no commit):
    python classify_images.py \
        --url https://github.com/you/Extracted-Files/tree/main/lowrisc_ibex \
        --reference-dir refs --dry-run

    # Classify and commit the segregation back into the dataset repo:
    python classify_images.py \
        --url https://github.com/you/Extracted-Files/tree/main/lowrisc_ibex \
        --reference-dir refs

    # Whole repo (every project folder), with the Claude vision fallback:
    python classify_images.py \
        --url https://github.com/you/Extracted-Files \
        --reference-url https://github.com/you/Extracted-Files/tree/main/_refs \
        --vision
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote, urlparse

import numpy as np
import requests

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RASTER_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_EXTS = RASTER_EXTS | {".svg"}

MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}

CATEGORY_ROOT = "image_categories"
# Every image is moved into a category. A separate review bucket prevents
# ambiguous screenshots/photos from being silently mislabelled or discarded.
LABEL_DIRS = {"useful": "useful", "not_useful": "non_useful", "uncertain": "review"}

NOT_USEFUL_KEYWORDS = {
    "logo", "logos", "badge", "shield", "icon", "favicon", "avatar", "banner",
    "screenshot", "screen_shot", "photo", "qr", "sponsor", "button", "watermark",
    "octocat", "social", "thumbnail", "thumb", "doxygen", "opencores",
    "compatible", "asicart", "menu", "floppy", "smiley", "shrooms", "tie_dye",
    "segmentation", "vga", "lotus", "wings_of_fury", "stop", "class",
    "gecko", "multi_function_conn", "board", "pcb",
}
USEFUL_KEYWORDS = {
    "block", "blockdiagram", "block_diagram", "diagram", "arch", "architecture",
    "schematic", "waveform", "timing", "fsm", "statemachine", "state_machine",
    "state", "pinout", "pin", "floorplan", "datapath", "data_path", "pipeline",
    "microarch", "topology", "register_map", "regmap", "bd", "wave", "rtl",
}

MIN_USEFUL_DIMENSION = 48

DEFAULT_CONFIDENT_MARGIN = 0.22
DEFAULT_VISION_MODEL = "claude-haiku-4-5"
VISION_MAX_LONG_EDGE = 768

GITHUB_API = "https://api.github.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("classify_images")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    path: str                       # repo-relative path of the image
    label: str                      # useful | not_useful | uncertain
    confidence: float
    method: str                     # keyword | features | vision | default
    reason: str = ""
    moved_to: Optional[str] = None  # repo-relative destination, if moved


@dataclass
class RepoLocation:
    owner: str
    repo: str
    branch: Optional[str]           # None -> resolve default branch
    subpath: str                    # "" for repo root


@dataclass
class ReferenceProfile:
    useful: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    not_useful: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    # few-shot samples for the vision model: list of (label, png_bytes)
    samples: list[tuple[str, bytes]] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.useful.size > 0 and self.not_useful.size > 0


# ---------------------------------------------------------------------------
# GitHub URL parsing
# ---------------------------------------------------------------------------

class URLParseError(ValueError):
    """Raised when a string is not a usable GitHub repo/tree URL."""


def parse_repo_url(url: str) -> RepoLocation:
    """
    Parse a GitHub URL into owner / repo / branch / subpath.

    Accepts:
        https://github.com/owner/repo
        https://github.com/owner/repo/tree/<branch>/<subpath...>
        github.com/owner/repo/tree/main/foo/bar
    """
    if not url or not url.strip():
        raise URLParseError("Empty URL.")
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if "github.com" not in parsed.netloc.lower():
        raise URLParseError(f"Not a github.com URL: {url!r}")

    segs = [s for s in parsed.path.split("/") if s]
    if len(segs) < 2:
        raise URLParseError(
            f"URL must include owner and repo: {url!r} "
            f"(e.g. https://github.com/you/Extracted-Files/tree/main/lowrisc_ibex)"
        )
    owner, repo = segs[0], segs[1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    branch: Optional[str] = None
    subpath = ""
    if len(segs) >= 3 and segs[2] in ("tree", "blob"):
        if len(segs) >= 4:
            branch = segs[3]
            subpath = "/".join(segs[4:])
    return RepoLocation(owner=owner, repo=repo, branch=branch, subpath=subpath)


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------

class GitHubError(RuntimeError):
    pass


class GitHubClient:
    """Minimal GitHub REST + Git Data API client backed by `requests`."""

    def __init__(self, token: Optional[str]):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github+json"})
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.token = token

    def _request(self, method: str, path: str, **kw) -> requests.Response:
        url = path if path.startswith("http") else f"{GITHUB_API}{path}"
        resp = self.session.request(method, url, timeout=30, **kw)
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            raise GitHubError("GitHub API rate limit exhausted — pass a --github-token.")
        return resp

    def default_branch(self, owner: str, repo: str) -> str:
        r = self._request("GET", f"/repos/{owner}/{repo}")
        if r.status_code != 200:
            raise GitHubError(f"Cannot read repo {owner}/{repo}: {r.status_code} {r.text[:200]}")
        return r.json().get("default_branch", "main")

    def head_sha(self, owner: str, repo: str, branch: str) -> str:
        r = self._request("GET", f"/repos/{owner}/{repo}/git/ref/heads/{quote(branch)}")
        if r.status_code != 200:
            raise GitHubError(f"Cannot resolve branch {branch}: {r.status_code} {r.text[:200]}")
        return r.json()["object"]["sha"]

    def commit_tree_sha(self, owner: str, repo: str, commit_sha: str) -> str:
        r = self._request("GET", f"/repos/{owner}/{repo}/git/commits/{commit_sha}")
        if r.status_code != 200:
            raise GitHubError(f"Cannot read commit {commit_sha}: {r.status_code}")
        return r.json()["tree"]["sha"]

    def list_tree(self, owner: str, repo: str, tree_sha: str) -> list[dict]:
        """Recursively list every blob: returns [{path, sha, mode, type}]."""
        r = self._request("GET",
                          f"/repos/{owner}/{repo}/git/trees/{tree_sha}",
                          params={"recursive": "1"})
        if r.status_code != 200:
            raise GitHubError(f"Cannot list tree: {r.status_code} {r.text[:200]}")
        data = r.json()
        if data.get("truncated"):
            log.warning("Repo tree is truncated by GitHub — very large repos may "
                        "miss some files. Point --url at a specific project folder.")
        return [e for e in data.get("tree", []) if e.get("type") == "blob"]

    def raw_bytes(self, owner: str, repo: str, path: str, ref: str) -> Optional[bytes]:
        """Download a file's raw bytes via the contents API (auth-friendly)."""
        r = self._request(
            "GET", f"/repos/{owner}/{repo}/contents/{quote(path)}",
            params={"ref": ref},
            headers={"Accept": "application/vnd.github.raw"},
        )
        if r.status_code != 200:
            log.debug("raw fetch %s -> %s", path, r.status_code)
            return None
        return r.content

    def raw_text(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        data = self.raw_bytes(owner, repo, path, ref)
        return data.decode("utf-8", errors="ignore") if data is not None else None

    # -- committing via the Git Data API -------------------------------------

    def commit_changes(self, owner: str, repo: str, branch: str,
                       entries: list[dict], message: str) -> str:
        """
        Create a single commit applying `entries` (Git tree objects) on top of
        `branch` and fast-forward the ref. Returns the new commit SHA.

        Each entry is a GitHub tree object:
          move/keep blob: {"path", "mode", "type":"blob", "sha": <blob sha or None to delete>}
          new text file:  {"path", "mode":"100644", "type":"blob", "content": <text>}
        """
        head = self.head_sha(owner, repo, branch)
        base_tree = self.commit_tree_sha(owner, repo, head)

        r = self._request("POST", f"/repos/{owner}/{repo}/git/trees",
                          json={"base_tree": base_tree, "tree": entries})
        if r.status_code not in (200, 201):
            raise GitHubError(f"Create tree failed: {r.status_code} {r.text[:300]}")
        new_tree = r.json()["sha"]

        r = self._request("POST", f"/repos/{owner}/{repo}/git/commits",
                          json={"message": message, "tree": new_tree, "parents": [head]})
        if r.status_code not in (200, 201):
            raise GitHubError(f"Create commit failed: {r.status_code} {r.text[:300]}")
        new_commit = r.json()["sha"]

        r = self._request("PATCH", f"/repos/{owner}/{repo}/git/refs/heads/{quote(branch)}",
                          json={"sha": new_commit})
        if r.status_code not in (200, 201):
            raise GitHubError(f"Update ref failed: {r.status_code} {r.text[:300]}")
        return new_commit


# ---------------------------------------------------------------------------
# Image decoding (raster + SVG)
# ---------------------------------------------------------------------------

def rasterize_svg(data: bytes, size: int = 512) -> Optional[bytes]:
    """Render SVG bytes to PNG bytes. Tries cairosvg, then svglib+reportlab."""
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=data, output_width=size, output_height=size)
    except Exception:
        pass
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(io.BytesIO(data))
        if drawing is None:
            return None
        buf = io.BytesIO()
        renderPM.drawToFile(drawing, buf, fmt="PNG")
        return buf.getvalue()
    except Exception as e:
        log.debug("SVG rasterization unavailable/failed: %s", e)
        return None


def load_rgb(data: bytes, ext: str) -> Optional["Image.Image"]:
    """Decode image bytes to a PIL image (rasterising SVG). None if unreadable."""
    if not HAS_PIL:
        return None
    if ext == ".svg":
        png = rasterize_svg(data)
        if png is None:
            return None
        data = png
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception as e:
        log.debug("Cannot decode %s image: %s", ext, e)
        return None


# ---------------------------------------------------------------------------
# Feature extraction (Stage 1)
# ---------------------------------------------------------------------------

def extract_features(img: "Image.Image") -> np.ndarray:
    """Small interpretable feature vector from a PIL image."""
    has_alpha = 0.0
    if img.mode in ("RGBA", "LA", "P"):
        alpha = np.asarray(img.convert("RGBA").split()[-1])
        if alpha.min() < 255:
            has_alpha = 1.0

    w, h = img.size
    aspect = max(w, h) / max(1, min(w, h))
    log_area = float(np.log10(max(1, w * h)))

    small = img.convert("RGB").resize((64, 64))
    arr = np.asarray(small).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    quant = (arr.astype(np.uint8) >> 4).reshape(-1, 3)
    color_ratio = len(np.unique(quant, axis=0)) / quant.shape[0]

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    ) / 255.0

    spread = arr.max(axis=2) - arr.min(axis=2)
    gray_frac = float((spread < 16).mean())
    white_frac = float((arr.min(axis=2) > 235).mean())

    gray = arr.mean(axis=2)
    edge_density = float((np.abs(np.diff(gray, axis=1)).mean()
                          + np.abs(np.diff(gray, axis=0)).mean()) / 2.0) / 255.0

    return np.array([
        min(aspect, 8.0) / 8.0, log_area / 8.0, color_ratio,
        min(colorfulness, 1.0), gray_frac, min(edge_density, 1.0),
        white_frac, has_alpha,
    ], dtype=np.float32)


def likely_photograph(vec: np.ndarray) -> bool:
    """Return True only for strongly photo-like images, not line diagrams."""
    color_ratio = float(vec[2])
    colorfulness = float(vec[3])
    gray_fraction = float(vec[4])
    edge_density = float(vec[5])
    white_fraction = float(vec[6])
    return (
        color_ratio > 0.18
        and colorfulness > 0.22
        and gray_fraction < 0.45
        and edge_density < 0.20
        and white_fraction < 0.70
    )


def classify_by_features(vec: np.ndarray, profile: ReferenceProfile) -> tuple[str, float]:
    z = (vec - profile.mean) / profile.std
    uz = (profile.useful - profile.mean) / profile.std
    nz = (profile.not_useful - profile.mean) / profile.std
    d_useful = _knn(z, uz)
    d_not = _knn(z, nz)
    total = d_useful + d_not
    if total <= 1e-9:
        return "uncertain", 0.0
    margin = abs(d_not - d_useful) / total
    return ("useful" if d_useful < d_not else "not_useful"), float(margin)


def _knn(point: np.ndarray, cluster: np.ndarray, k: int = 3) -> float:
    dists = np.linalg.norm(cluster - point, axis=1)
    return float(np.sort(dists)[:min(k, len(dists))].mean())


# ---------------------------------------------------------------------------
# Keyword prior
# ---------------------------------------------------------------------------

def classify_by_keywords(rel_path: str) -> Optional[tuple[str, float, str]]:
    tokens = {t for t in re.split(r"[^a-z0-9]+", rel_path.lower()) if t}
    hit_not = tokens & NOT_USEFUL_KEYWORDS
    hit_use = tokens & USEFUL_KEYWORDS
    if hit_not and not hit_use:
        return "not_useful", 0.9, f"filename keyword(s): {', '.join(sorted(hit_not))}"
    if hit_use and not hit_not:
        return "useful", 0.85, f"filename keyword(s): {', '.join(sorted(hit_use))}"
    return None


# ---------------------------------------------------------------------------
# Reference profile (example images)
# ---------------------------------------------------------------------------

def build_reference_profile(local_dir: Optional[Path], gh_url: Optional[str],
                            gh: GitHubClient, shots: int) -> ReferenceProfile:
    """Load example images from a local folder or a GitHub folder and featurise."""
    profile = ReferenceProfile()
    pairs: dict[str, list[tuple[str, bytes]]] = {"useful": [], "not_useful": []}

    if gh_url:
        _collect_reference_github(gh_url, gh, pairs)
    elif local_dir:
        _collect_reference_local(local_dir, pairs)
    else:
        return profile

    useful_vecs, not_vecs, samples = [], [], []
    for label, items in pairs.items():
        for rel, data in items:
            ext = Path(rel).suffix.lower()
            img = load_rgb(data, ext)
            if img is None:
                continue
            (useful_vecs if label == "useful" else not_vecs).append(extract_features(img))
            if shots > 0 and sum(1 for s in samples if s[0] == label) < shots:
                png = _to_png_bytes(img)
                if png:
                    samples.append((label, png))

    if not useful_vecs or not not_vecs:
        log.warning("Need readable example images in BOTH 'useful/' and "
                    "'not_useful/' — Stage 1 uses keywords only.")
        return profile

    profile.useful = np.vstack(useful_vecs)
    profile.not_useful = np.vstack(not_vecs)
    allv = np.vstack([profile.useful, profile.not_useful])
    profile.mean = allv.mean(axis=0)
    profile.std = allv.std(axis=0) + 1e-6
    profile.samples = samples
    log.info("Reference profile: %d useful + %d not-useful example images.",
             len(useful_vecs), len(not_vecs))
    return profile


def _class_dirname(name: str) -> Optional[str]:
    n = name.lower()
    if n in ("useful", "good"):
        return "useful"
    if n in ("not_useful", "not-useful", "rejected", "useless", "bad"):
        return "not_useful"
    return None


def _collect_reference_local(root: Path, pairs: dict) -> None:
    if not root.exists():
        log.warning("Reference dir %s does not exist.", root)
        return
    for sub in root.iterdir():
        cls = _class_dirname(sub.name) if sub.is_dir() else None
        if not cls:
            continue
        for p in sorted(sub.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                try:
                    pairs[cls].append((p.name, p.read_bytes()))
                except OSError:
                    pass


def _collect_reference_github(url: str, gh: GitHubClient, pairs: dict) -> None:
    loc = parse_repo_url(url)
    branch = loc.branch or gh.default_branch(loc.owner, loc.repo)
    head = gh.head_sha(loc.owner, loc.repo, branch)
    blobs = gh.list_tree(loc.owner, loc.repo, gh.commit_tree_sha(loc.owner, loc.repo, head))
    prefix = loc.subpath.rstrip("/") + "/" if loc.subpath else ""
    for e in blobs:
        path = e["path"]
        if prefix and not path.startswith(prefix):
            continue
        ext = Path(path).suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        rel = path[len(prefix):]
        cls = _class_dirname(rel.split("/")[0]) if "/" in rel else None
        if not cls:
            continue
        data = gh.raw_bytes(loc.owner, loc.repo, path, branch)
        if data:
            pairs[cls].append((Path(path).name, data))


# ---------------------------------------------------------------------------
# Stage 2: Claude vision fallback
# ---------------------------------------------------------------------------

VISION_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["useful", "not_useful"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["label", "confidence", "reason"],
        "additionalProperties": False,
    },
}

VISION_SYSTEM = (
    "You classify images extracted from open-source hardware (HDL) projects for "
    "use in a hardware-design dataset. Label an image 'useful' if it conveys "
    "technical design information — block diagrams, architecture/microarchitecture "
    "diagrams, schematics, FSM/state diagrams, waveforms, timing diagrams, "
    "datapaths, pipelines, floorplans, pinouts, register maps. Label it "
    "'not_useful' if it is decorative or non-technical — company/project logos, "
    "build/coverage badges or shields, icons, avatars, banners, website "
    "screenshots, photos, QR codes. When unsure, pick the closer of the two."
)


def _to_png_bytes(img: "Image.Image", max_edge: int = VISION_MAX_LONG_EDGE) -> Optional[bytes]:
    try:
        im = img.convert("RGB")
        im.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _png_block(png: bytes) -> dict:
    return {"type": "image", "source": {
        "type": "base64", "media_type": "image/png",
        "data": base64.standard_b64encode(png).decode("utf-8")}}


class VisionClassifier:
    def __init__(self, model: str, profile: ReferenceProfile):
        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.few_shot: list[dict] = []
        for label, png in profile.samples:
            self.few_shot.append({"type": "text", "text": f"Example ({label}):"})
            self.few_shot.append(_png_block(png))

    def classify(self, png: bytes) -> Optional[tuple[str, float, str]]:
        content = list(self.few_shot) + [
            _png_block(png),
            {"type": "text", "text": "Classify THIS image as useful or not_useful "
                                     "for a hardware-design dataset."},
        ]
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=300, system=VISION_SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_config={"format": VISION_SCHEMA},
            )
        except Exception as e:
            log.warning("Vision call failed: %s", e)
            return None
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            d = json.loads(text)
            if d.get("label") in ("useful", "not_useful"):
                return d["label"], float(d.get("confidence", 0.5)), str(d.get("reason", ""))[:300]
        except (json.JSONDecodeError, ValueError):
            pass
        return None


# ---------------------------------------------------------------------------
# Per-image decision (hybrid)
# ---------------------------------------------------------------------------

def decide(rel_path: str, data: bytes, profile: ReferenceProfile,
           confident_margin: float, vision: Optional[VisionClassifier]) -> Verdict:
    ext = Path(rel_path).suffix.lower()

    # Stage 1a: decisive keyword.
    kw = classify_by_keywords(rel_path)
    if kw:
        return Verdict(rel_path, kw[0], kw[1], "keyword", kw[2])

    img = load_rgb(data, ext)
    if img is None:
        # Unreadable (e.g. SVG with no rasteriser installed) and no keyword.
        return Verdict(rel_path, "uncertain", 0.0, "default",
                       "could not decode image (install svglib/cairosvg for SVG)")

    # UI fragments, bullets and button glyphs are not hardware-design material.
    # This catches files such as tab_a.png and Stop.png without filename rules.
    if min(img.size) < MIN_USEFUL_DIMENSION:
        return Verdict(rel_path, "not_useful", 0.95, "dimensions",
                       f"tiny image dimensions {img.size[0]}x{img.size[1]}")

    features = extract_features(img)
    if likely_photograph(features):
        return Verdict(rel_path, "not_useful", 0.7, "visual_heuristic",
                       "photo-like visual features; hardware photos are excluded from the diagram dataset")

    # Stage 1b: feature similarity to example images.
    if profile.usable:
        label, margin = classify_by_features(features, profile)
        if margin >= confident_margin:
            return Verdict(rel_path, label, round(margin, 3), "features",
                           f"feature similarity margin {margin:.2f}")

    # Stage 2: vision fallback.
    if vision is not None:
        png = _to_png_bytes(img)
        if png:
            v = vision.classify(png)
            if v:
                return Verdict(rel_path, v[0], round(v[1], 3), "vision", v[2])

    return Verdict(rel_path, "uncertain", 0.0, "default",
                   "no decisive keyword, feature, or vision signal")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _project_roots(blobs: list[dict], prefix: str) -> dict[str, list[str]]:
    """
    Map project root (dir containing meta.json) -> list of image paths under it.
    Falls back to `prefix` as a single root when no meta.json is present.
    """
    meta_dirs = sorted({_dirname(e["path"]) for e in blobs
                        if Path(e["path"]).name == "meta.json"
                        and e["path"].startswith(prefix)})
    images = [e["path"] for e in blobs
              if Path(e["path"]).suffix.lower() in IMAGE_EXTS
              and e["path"].startswith(prefix)
              and CATEGORY_ROOT not in e["path"].split("/")]

    roots: dict[str, list[str]] = {}
    if meta_dirs:
        for img in images:
            root = max((d for d in meta_dirs if img.startswith(d + "/") or _dirname(img) == d),
                       key=len, default=None)
            roots.setdefault(root if root is not None else prefix.rstrip("/"), []).append(img)
    else:
        roots[prefix.rstrip("/")] = images
    return {k: v for k, v in roots.items() if v}


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def run(
    url: str,
    reference_dir: Optional[Path],
    reference_url: Optional[str],
    confident_margin: float,
    use_vision: bool,
    vision_model: str,
    vision_shots: int,
    github_token: Optional[str],
    message: Optional[str],
    dry_run: bool,
) -> dict:
    if not HAS_PIL:
        log.error("Pillow (PIL) is required: pip install pillow")
        return {}
    if not github_token and not dry_run:
        log.error("A GitHub token with write access is required to commit. "
                  "Pass --github-token or set GITHUB_TOKEN (or use --dry-run).")
        return {}

    loc = parse_repo_url(url)
    gh = GitHubClient(github_token)
    branch = loc.branch or gh.default_branch(loc.owner, loc.repo)
    prefix = (loc.subpath.rstrip("/") + "/") if loc.subpath else ""
    log.info("Dataset repo %s/%s  branch=%s  subpath=%r",
             loc.owner, loc.repo, branch, loc.subpath)

    profile = build_reference_profile(reference_dir, reference_url, gh,
                                      vision_shots if use_vision else 0)
    if not profile.usable:
        log.info("Stage-1 feature similarity disabled (no usable reference set).")

    vision: Optional[VisionClassifier] = None
    if use_vision:
        if not HAS_ANTHROPIC:
            log.warning("--vision set but `anthropic` not installed (pip install anthropic).")
        elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            log.warning("--vision set but ANTHROPIC_API_KEY is not set.")
        else:
            vision = VisionClassifier(vision_model, profile)
            log.info("Vision fallback enabled (model=%s).", vision_model)

    head = gh.head_sha(loc.owner, loc.repo, branch)
    blobs = gh.list_tree(loc.owner, loc.repo, gh.commit_tree_sha(loc.owner, loc.repo, head))
    sha_by_path = {e["path"]: e["sha"] for e in blobs}
    mode_by_path = {e["path"]: e.get("mode", "100644") for e in blobs}

    roots = _project_roots(blobs, prefix)
    if not roots:
        log.error("No images found under %r in %s/%s.", loc.subpath, loc.owner, loc.repo)
        return {}
    log.info("Found %d project folder(s) with images.", len(roots))

    entries: list[dict] = []          # Git tree changes to commit
    grand = {"useful": 0, "not_useful": 0, "uncertain": 0, "total": 0}

    for root, images in sorted(roots.items()):
        log.info("--- %s : %d image(s) ---", root or "<repo root>", len(images))
        verdicts: list[Verdict] = []
        for path in sorted(images):
            data = gh.raw_bytes(loc.owner, loc.repo, path, branch)
            if data is None:
                log.warning("  could not download %s", path)
                continue
            rel_in_project = path[len(root) + 1:] if root else path
            v = decide(rel_in_project, data, profile, confident_margin, vision)
            v.path = path  # record the full repo path
            _plan_move(v, root, path, sha_by_path, mode_by_path, entries)
            verdicts.append(v)
            log.info("  [%-10s] %.2f %-8s %s%s", v.label, v.confidence, v.method,
                     rel_in_project, f"  ({v.reason})" if v.reason else "")

        _plan_reports(root, verdicts, entries)
        for k in grand:
            grand[k] += _counts(verdicts)[k]

    if dry_run:
        log.info("DRY RUN — %d image(s) classified, nothing committed.", grand["total"])
    elif not entries:
        log.info("Nothing to commit.")
    else:
        msg = message or (f"classify images: {grand['useful']} useful / "
                          f"{grand['not_useful']} not_useful / {grand['uncertain']} uncertain")
        try:
            sha = gh.commit_changes(loc.owner, loc.repo, branch, entries, msg)
            log.info("Committed %s to %s/%s@%s", sha[:10], loc.owner, loc.repo, branch)
        except GitHubError as e:
            log.error("Commit failed: %s", e)

    log.info("=" * 60)
    log.info("SUMMARY: %s", json.dumps(grand, indent=2))
    log.info("=" * 60)
    return grand


def _plan_move(v: Verdict, root: str, path: str,
               sha_by_path: dict, mode_by_path: dict, entries: list[dict]) -> None:
    """Add tree entries to move every image into its explicit category folder."""
    sub = LABEL_DIRS.get(v.label)
    if sub is None:
        return
    rel_in_project = path[len(root) + 1:] if root else path
    dest = f"{root}/{CATEGORY_ROOT}/{sub}/{rel_in_project}" if root \
        else f"{CATEGORY_ROOT}/{sub}/{rel_in_project}"
    blob_sha = sha_by_path.get(path)
    if not blob_sha:
        return
    mode = mode_by_path.get(path, "100644")
    entries.append({"path": dest, "mode": mode, "type": "blob", "sha": blob_sha})  # add at new path
    entries.append({"path": path, "mode": mode, "type": "blob", "sha": None})       # delete old path
    v.moved_to = dest


def _plan_reports(root: str, verdicts: list[Verdict], entries: list[dict]) -> None:
    """Add the manifest (and meta.json update if it can be assumed) as text blobs."""
    manifest = {"summary": _counts(verdicts), "images": [vars(v) for v in verdicts]}
    manifest_path = f"{root}/image_classification.json" if root else "image_classification.json"
    entries.append({"path": manifest_path, "mode": "100644", "type": "blob",
                    "content": json.dumps(manifest, indent=2)})


def _counts(verdicts: list[Verdict]) -> dict:
    c = {"useful": 0, "not_useful": 0, "uncertain": 0, "total": len(verdicts)}
    for v in verdicts:
        c[v.label] = c.get(v.label, 0) + 1
    return c


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify images in a GitHub dataset repo as useful vs "
                    "not-useful and commit the segregation back.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", required=True,
                        help="GitHub URL of the dataset repo or a project folder "
                             "(e.g. https://github.com/you/Extracted-Files/tree/main/lowrisc_ibex).")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="Local folder with 'useful/' and 'not_useful/' example images.")
    parser.add_argument("--reference-url", type=str, default=None,
                        help="GitHub folder (tree URL) with 'useful/' and 'not_useful/' examples.")
    parser.add_argument("--confident-margin", type=float, default=DEFAULT_CONFIDENT_MARGIN,
                        help=f"Min feature margin (0..1) to decide without vision "
                             f"(default: {DEFAULT_CONFIDENT_MARGIN}).")
    parser.add_argument("--vision", action="store_true",
                        help="Enable the Claude vision fallback for uncertain images.")
    parser.add_argument("--vision-model", type=str, default=DEFAULT_VISION_MODEL,
                        help=f"Vision model id (default: {DEFAULT_VISION_MODEL}).")
    parser.add_argument("--vision-shots", type=int, default=2,
                        help="Reference examples per class to few-shot the vision model "
                             "(default: 2; 0 = text criteria only).")
    parser.add_argument("--github-token", type=str, default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub token with write access (or env GITHUB_TOKEN).")
    parser.add_argument("--message", type=str, default=None,
                        help="Override the commit message.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify and report only; commit nothing.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")

    args = parser.parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = run(
            url=args.url,
            reference_dir=args.reference_dir,
            reference_url=args.reference_url,
            confident_margin=args.confident_margin,
            use_vision=args.vision,
            vision_model=args.vision_model,
            vision_shots=args.vision_shots,
            github_token=args.github_token,
            message=args.message,
            dry_run=args.dry_run,
        )
    except URLParseError as e:
        log.error("Invalid --url: %s", e)
        return 2
    except GitHubError as e:
        log.error("GitHub error: %s", e)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return 130
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
