#!/usr/bin/env python3
"""
Biomedical NLP Knowledge Extractor — Generalized Publication-Grade Pipeline
============================================================================
Version: 3.0.0
License: MIT

Purpose: AI-augmented extraction of biomedical target (protein/gene/drug)
         interaction data from literature. Fully query-driven — adapts to any target.

Architecture
-----------
Stage 1 – Target Configuration
    User provides a natural-language query describing what to extract.
    Ollama analyzes the query and generates:
      - Target proteins/genes (e.g., SIRT1, CYP3A4, BACE1)
      - Interaction types (inhibition, activation, substrate, modulator, etc.)
      - Related compounds/drugs
      - Additional context keywords

Stage 2 – Dynamic Regex Pre-filter
    Generates target-specific patterns from Stage 1 config.
    Broad, conservative pass retains any sentence that MIGHT be relevant.
    False positives tolerated; false negatives avoided.

Stage 3 – AI Extraction
    Per-sentence structured extraction via Ollama → returns JSON with:
    compound, target, interaction_type, mechanism, evidence, confidence,
    and quantitative values.

Stage 4 – Persistence
    Per-file JSON in /results/json/
    Merged master JSON + two CSVs (compound-level, interaction-level)

Stage 5 – Dashboard
    Self-contained HTML dashboard with tables, charts, and download buttons.
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
import argparse
import textwrap
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import requests
import pandas as pd

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("biomedical_nlp")

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
LLM_MODEL       = "llama3.2:3b"
REQUEST_TIMEOUT = 300

RESULTS_DIR     = Path("results")
JSON_DIR        = RESULTS_DIR / "json"
CSV_DIR         = RESULTS_DIR / "csv"
HTML_DIR        = RESULTS_DIR / "dashboard"

OUT_JSON_MASTER      = RESULTS_DIR / "master_extractions.json"
OUT_JSON_MEMORY      = RESULTS_DIR / "knowledge_memory.json"
OUT_CSV_COMPOUND     = CSV_DIR / "compound_interactions.csv"
OUT_CSV_INTERACTION  = CSV_DIR / "interaction_details.csv"
OUT_DASHBOARD        = HTML_DIR / "dashboard.html"

# ─────────────────────────────────────────────
#  Progress Tracker
# ─────────────────────────────────────────────
class ProgressTracker:
    """
    Thread-safe progress tracker with:
    - Overall pipeline progress (stages)
    - Per-file progress bar with ETA
    - Live statistics (sentences, records)
    - Animated spinner for AI calls
    """

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    STAGE_NAMES = ["Query Analysis", "Building Patterns", "Scanning Files",
                   "Saving Results", "Generating Dashboard"]

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Pipeline stage
        self.current_stage = 0
        self.stage_name = "Initializing"

        # File-level progress
        self.total_files = 0
        self.files_done = 0
        self.current_file = ""
        self.file_progress = 0.0  # 0.0 to 1.0

        # Sentence-level
        self.total_sentences = 0
        self.sentences_scanned = 0
        self.sentences_filtered = 0

        # Records
        self.records_found = 0

        # Timing
        self.start_time = time.time()
        self.file_start_time = time.time()
        self.ai_call_active = False
        self.ai_call_count = 0
        self._spinner_index = 0
        self._spinner_lock = threading.Lock()

        # Display throttle (update every 0.1s max)
        self._last_display = 0.0
        self._display_interval = 0.1

        # Header already printed flag
        self._header_printed = False

    def start_pipeline(self, total_files: int):
        with self._lock:
            self.total_files = total_files
            self.start_time = time.time()
        self._print_header()
        # Start background display thread
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def _display_loop(self):
        """Background thread that refreshes display every 0.5s."""
        while not self._stop_event.is_set():
            self.display()
            # Wait for stop or interval
            self._stop_event.wait(0.5)

    def set_stage(self, stage_idx: int, name: str = ""):
        with self._lock:
            self.current_stage = stage_idx
            self.stage_name = name or self.STAGE_NAMES[stage_idx] if stage_idx < len(self.STAGE_NAMES) else f"Stage {stage_idx}"

    def start_file(self, filename: str, num_sentences: int):
        with self._lock:
            self.current_file = filename
            self.total_sentences = num_sentences
            self.sentences_scanned = 0
            self.sentences_filtered = 0
            self.file_progress = 0.0
            self.file_start_time = time.time()

    def update_file_progress(self, sentences_scanned: int, sentences_filtered: int, records_found: int):
        with self._lock:
            self.sentences_scanned = sentences_scanned
            self.sentences_filtered = sentences_filtered
            self.records_found = records_found
            if self.total_sentences > 0:
                self.file_progress = sentences_scanned / self.total_sentences

    def finish_file(self, records_count: int):
        with self._lock:
            self.files_done += 1
            self.records_found += records_count
            self.file_progress = 1.0

    def start_ai_call(self):
        with self._lock:
            self.ai_call_active = True
            self.ai_call_count += 1

    def stop_ai_call(self):
        with self._lock:
            self.ai_call_active = False

    def _print_header(self):
        if self._header_printed:
            return
        self._header_printed = True
        print()

    def _eta_string(self) -> str:
        """Calculate and return ETA string."""
        elapsed = time.time() - self.start_time
        if self.files_done == 0 or self.total_files == 0:
            return "calculating..."
        rate = elapsed / self.files_done
        remaining = self.total_files - self.files_done
        eta_seconds = rate * remaining
        if eta_seconds < 60:
            return f"~{int(eta_seconds)}s remaining"
        elif eta_seconds < 3600:
            mins = int(eta_seconds / 60)
            return f"~{mins}m {int(eta_seconds % 60)}s remaining"
        else:
            hours = int(eta_seconds / 3600)
            mins = int((eta_seconds % 3600) / 60)
            return f"~{hours}h {mins}m remaining"

    def _elapsed_string(self) -> str:
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"elapsed {int(elapsed)}s"
        elif elapsed < 3600:
            return f"elapsed {int(elapsed/60)}m {int(elapsed%60)}s"
        else:
            return f"elapsed {int(elapsed/3600)}h {int((elapsed%3600)/60)}m"

    def _spinner_char(self) -> str:
        with self._spinner_lock:
            self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_CHARS)
            return self.SPINNER_CHARS[self._spinner_index]

    def _build_bar(self, progress: float, width: int = 32) -> str:
        """Build an ASCII progress bar."""
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        pct = int(progress * 100)
        return f"[{bar}] {pct:3d}%"

    def display(self, force: bool = False):
        """Print the current progress state. Throttled unless force=True."""
        now = time.time()
        if not force and (now - self._last_display) < self._display_interval:
            return
        self._last_display = now

        with self._lock:
            # Stage indicator
            stage_dots = "░" * 5
            if self.current_stage < 5:
                stage_dots = "█" * self.current_stage + "░" * (5 - self.current_stage)

            # Overall progress
            if self.total_files > 0:
                overall_pct = self.files_done / self.total_files
                overall_bar = self._build_bar(overall_pct, 24)
                file_text = f"{self.files_done}/{self.total_files} files"
            else:
                overall_bar = "[░░░░░░░░░░░░░░░░░░░░░░]   0%"
                file_text = "0 files"

            # Per-file progress (only when active)
            file_bar = ""
            if self.current_file and self.total_files > 0:
                file_bar = f"\n  ↳ {self.current_file[:45]:<45} {self._build_bar(self.file_progress, 24)}"

            # AI spinner
            spinner = self._spinner_char() if self.ai_call_active else " "

            # Live stats
            stats = (
                f"  scan:{self.sentences_scanned}"
                f"  filter:{self.sentences_filtered}"
                f"  batch:{self.ai_call_count}"
                f"  records:{self.records_found}"
            )

            # Stage name and ETA
            eta = self._eta_string()
            elapsed = self._elapsed_string()

        # Clear line and print
        line = (
            f"\r{spinner} "
            f"[Stage █{self.current_stage} {self.stage_name[:18]:<18}] "
            f"{overall_bar} {file_text} "
            f"{eta:<22} "
            f"{elapsed}"
            f"{file_bar}"
            f"  {stats}   "
        )
        # Truncate to terminal width
        try:
            width = os.get_terminal_size().columns
            line = line[: width - 2] if len(line) >= width else line
        except OSError:
            pass
        print(line, end="", flush=True)

    def finish(self):
        """Print final completion line."""
        self._stop_event.set()
        if hasattr(self, '_display_thread'):
            self._display_thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time
        with self._lock:
            if elapsed < 60:
                elapsed_str = f"{int(elapsed)}s"
            elif elapsed < 3600:
                elapsed_str = f"{int(elapsed/60)}m {int(elapsed%60)}s"
            else:
                elapsed_str = f"{int(elapsed/3600)}h {int((elapsed%3600)/60)}m"
        print(f"\r✓ Pipeline complete in {elapsed_str} — {self.records_found} records from {self.files_done} files".ljust(
            max(60, os.get_terminal_size().columns - 2 if hasattr(os, 'get_terminal_size') else 80)))


# Global tracker instance
tracker = ProgressTracker()


# ─────────────────────────────────────────────
#  Default Interaction Types & Mechanisms
# ─────────────────────────────────────────────
DEFAULT_INTERACTION_TYPES = [
    "inhibition", "inhibitor", "inhibitory",
    "activation", "activator", "activating", "activator",
    "substrate", "metabolized by", "induction", "inducer",
    "modulation", "modulator", "antagonist", "agonist",
    "binding", "affinity", "Ki", "IC50", " potency",
]

DEFAULT_MECHANISMS = [
    "competitive", "non-competitive", "uncompetitive", "mixed",
    "mechanism-based", "irreversible", "covalent", "reversible",
    "time-dependent", "allosteric", "orthosteric",
]

# Generic quantitative markers for pre-filtering
QUANTITATIVE_TOKENS = [
    "ic50", "ki", "kd", "ec50", "potency", "affinity",
    "nm", "μm", "um", "mg/ml", "ng/ml", "nmol", "μmol",
    "nmol/l", "μmol/l", "selectivity", "fold",
]

# ─────────────────────────────────────────────
#  Knowledge Memory
# ─────────────────────────────────────────────
def load_memory() -> dict:
    if OUT_JSON_MEMORY.exists():
        with open(OUT_JSON_MEMORY) as f:
            mem = json.load(f)
        log.info("Memory loaded: %d known compounds, %d targets",
                 len(mem.get("compounds", [])),
                 len(mem.get("targets", {})))
        return mem
    return {
        "compounds": [],
        "targets": defaultdict(list),
        "interaction_types": defaultdict(list),
        "stats": {}
    }

def save_memory(mem: dict):
    with open(OUT_JSON_MEMORY, "w") as f:
        json.dump(mem, f, indent=2)

# ─────────────────────────────────────────────
#  Ollama Interface
# ─────────────────────────────────────────────
def ollama_call(prompt: str, system: str = "") -> str:
    """Call Ollama and return raw response text."""
    full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}" if system else prompt
    tracker.start_ai_call()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9},
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama at %s — is it running?", OLLAMA_URL)
        return ""
    except Exception as e:
        log.warning("Ollama error: %s", e)
        return ""
    finally:
        tracker.stop_ai_call()

def parse_json_response(text: str) -> list | dict | None:
    """Robustly extract JSON from an LLM response that may contain prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None

# ─────────────────────────────────────────────
#  Stage 1 — Query Analysis & Target Configuration
# ─────────────────────────────────────────────
ANALYZE_SYSTEM = """You are a biomedical NLP expert specializing in literature-based knowledge extraction.
Your task is to analyze the user's research query and generate a structured search configuration.
Return ONLY valid JSON — no explanations, no markdown, no prose."""

ANALYZE_PROMPT_TMPL = """
User query: "{query}"

Analyze this query and return a JSON object with these fields:

{{
  "targets": [
    {{
      "name": "canonical protein/gene target name (e.g., SIRT1, CYP3A4, BACE1, TNF-alpha)",
      "synonyms": ["list of alternative names or abbreviations"],
      "category": "enzyme | receptor | transporter | channel | transcription factor | protein | other"
    }}
  ],
  "interaction_types": [
    "list of interaction types relevant to query: inhibition, activation, substrate, modulation, etc."
  ],
  "mechanisms": [
    "list of mechanism terms to look for (competitive, allosteric, irreversible, etc.)"
  ],
  "compound_hints": ["list of any specific compounds or drug classes mentioned in the query"],
  "extra_keywords": ["additional context-specific search terms beyond interaction words"],
  "query_summary": "one-sentence plain-English summary of what to extract",
  "species_context": "human | rat | mouse | all (if relevant, otherwise empty string)"
}}

Rules:
- If the query mentions 'all targets' or is very broad, set targets to [] (wildcard).
- Extract ALL meaningful targets mentioned in the query.
- Include common synonyms for each target (e.g., SIRT1 → ["SIRT1", "Sirt1", "NAD+-dependent deacetylase"]).
- Set interaction_types to [] for wildcard (all interaction types).
- Use specific mechanism terms when the query mentions them.
Return valid JSON only.
"""

def analyze_query(user_query: str) -> dict:
    """
    Send user query to LLM for analysis → returns structured config.
    This replaces hardcoded CYP logic with dynamic, query-driven configuration.
    """
    log.info("Stage 1 — Analyzing user query with AI...")
    raw = ollama_call(ANALYZE_PROMPT_TMPL.format(query=user_query), ANALYZE_SYSTEM)
    parsed = parse_json_response(raw)

    if parsed and isinstance(parsed, dict):
        targets = parsed.get("targets", [])
        interactions = parsed.get("interaction_types", [])
        mechanisms = parsed.get("mechanisms", [])
        keywords = parsed.get("extra_keywords", [])

        log.info("  Query: %s", parsed.get("query_summary", ""))
        log.info("  Targets: %s", [t.get("name") if isinstance(t, dict) else t for t in targets] or "All")
        log.info("  Interaction types: %s", interactions or "All")
        log.info("  Mechanisms: %s", mechanisms or "All")

        return parsed

    log.warning("  Query analysis failed — using permissive defaults")
    return {
        "targets": [],
        "interaction_types": [],
        "mechanisms": [],
        "compound_hints": [],
        "extra_keywords": [],
        "query_summary": user_query,
        "species_context": "",
    }

# ─────────────────────────────────────────────
#  Stage 2 — Dynamic Regex Pattern Builder
# ─────────────────────────────────────────────
def build_target_patterns(config: dict) -> re.Pattern:
    """
    Dynamically build a regex pattern based on the targets in config.
    Supports: SIRT1, CYP isoforms, BACE1, TNF-alpha, etc.
    """
    target_list = config.get("targets", [])
    all_names = []

    if not target_list:
        # Wildcard mode — very permissive
        log.info("  Target mode: WILDCARD (all targets)")
        return re.compile(r".*", re.IGNORECASE)

    for t in target_list:
        if isinstance(t, dict):
            name = t.get("name", "")
            synonyms = t.get("synonyms", [])
            all_names.extend([name] + synonyms)
        elif isinstance(t, str):
            all_names.append(t)

    # Escape special regex characters, handle spaces
    escaped = []
    for name in all_names:
        # Handle Greek letters, hyphens, Greek symbols
        clean = re.escape(name)
        escaped.append(clean)

    # Build combined pattern
    names_pattern = "|".join(escaped)
    pattern = re.compile(
        rf"""
        (?:
            (?:protein|gene|enzyme|receptor|target)\s+of\s+  # "protein of X"
            |(?:binding\s+to\s+|interaction\s+with\s+)     # "binding to X"
        )?
        ({names_pattern})
        """,
        re.VERBOSE | re.IGNORECASE
    )
    return pattern

def build_interaction_pattern(config: dict) -> list[str]:
    """Build list of interaction keywords from config."""
    config_interactions = config.get("interaction_types", [])
    if not config_interactions:
        return DEFAULT_INTERACTION_TYPES
    return list(set(DEFAULT_INTERACTION_TYPES + config_interactions))

def build_mechanism_pattern(config: dict) -> list[str]:
    """Build list of mechanism keywords from config."""
    config_mechanisms = config.get("mechanisms", [])
    if not config_mechanisms:
        return DEFAULT_MECHANISMS
    return list(set(DEFAULT_MECHANISMS + config_mechanisms))

def build_extra_keywords(config: dict) -> list[str]:
    """Get extra keywords from config."""
    return config.get("extra_keywords", [])

def sentence_is_candidate(
    sentence: str,
    target_pattern: re.Pattern,
    interaction_tokens: list[str],
    mechanism_tokens: list[str],
    extra_keywords: list[str],
    config: dict,
    memory: dict
) -> bool:
    """
    Conservative pre-filter for a dynamic biomedical target.
    Returns True if sentence should be sent to AI extraction.

    Logic:
    - Must match at least one target pattern OR be in known compounds
    - Must match at least one interaction token OR mechanism token
    - If target CYPs specified, filter accordingly
    """
    s_lower = sentence.lower()

    # ── Target match ──
    has_target = bool(target_pattern.search(sentence))
    has_known_compound = any(c.lower() in s_lower for c in memory.get("compounds", []))

    if not (has_target or has_known_compound):
        return False

    # ── Interaction/mechanism match ──
    has_interaction = any(tok.lower() in s_lower for tok in interaction_tokens)
    has_mechanism = any(tok.lower() in s_lower for tok in mechanism_tokens)
    has_quant = any(tok.lower() in s_lower for tok in QUANTITATIVE_TOKENS)
    has_extra = any(kw.lower() in s_lower for kw in extra_keywords)

    if not (has_interaction or has_mechanism or has_quant or has_extra or has_known_compound):
        return False

    # ── Species filter (if specified) ──
    species = config.get("species_context", "")
    if species and species.lower() != "all":
        species_terms = {
            "human": ["human", "homo sapiens", "h sapiens"],
            "rat": ["rat", "rattus norvegicus", "r norvegicus"],
            "mouse": ["mouse", "mus musculus", "m musculus"],
        }
        terms = species_terms.get(species.lower(), [])
        if terms and not any(t in s_lower for t in terms):
            # Soft filter — warn but don't reject
            log.debug("  Sentence may be wrong species: %s", sentence[:80])

    return True

def split_sentences(text: str) -> list[str]:
    """Simple rule-based sentence splitter for biomedical text."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# ─────────────────────────────────────────────
#  Stage 3 — AI Extraction
# ─────────────────────────────────────────────
EXTRACT_SYSTEM = """You are a biomedical NLP extraction expert specializing in pharmaceutical drug-target interaction extraction.
Your job is to carefully read each sentence and extract ALL relevant interactions without skipping any."""

EXTRACT_PROMPT_TMPL = """
You are analyzing {n_sentences} sentences to extract drug-target interaction data.
For EVERY sentence below that contains a relevant biomedical interaction, you MUST return a JSON object.

CRITICAL: Do not skip sentences. If a sentence mentions any protein/gene target and a compound or drug affecting it, extract it.

Sentences to analyze:
{bulk_sentences}

For EACH sentence that contains a relevant interaction, extract and return:
{{
  "sentence_index": <1-based number matching the sentence above>,
  "compound_name": "drug/chemical name from the sentence, or null",
  "target_name": "canonical protein/gene name (e.g., SIRT1, CYP3A4, BACE1, TNF-alpha)",
  "target_species": "human | rat | mouse | null if not specified",
  "interaction_type": "inhibition | activation | substrate | induction | modulation | antagonist | agonist | binding | unknown",
  "mechanism": "competitive | non-competitive | allosteric | irreversible | reversible | mechanism-based | time-dependent | unknown",
  "quantitative_values": {{
    "IC50": "value with units or null",
    "Ki": "value with units or null",
    "EC50": "value with units or null",
    "Kd": "value with units or null",
    "potency": "value with units or null",
    "selectivity": "value or null"
  }},
  "evidence_sentence": "<exact sentence text>",
  "confidence": 0.0_to_1.0,
  "notes": "any caveats"
}}

Extraction rules:
- Return a JSON ARRAY with ONE object per sentence that has an interaction.
- Do NOT return anything for sentences with no relevant interaction.
- sentence_index must exactly match the sentence number.
- Set confidence: 0.9+ if compound+target+mechanism+quantitative all explicit; 0.75 if compound or mechanism missing; 0.6 if only target identified.
- Normalize: CYP3A4, CYP2D6, CYP2C9, SIRT1, BACE1, TNF-alpha, etc.
- compound_name: extract the specific drug/compound name if present.
- Return ONLY valid JSON array. No text before or after.
"""

def ai_extract_batch(sentences: list[str], config: dict) -> list[dict]:
    """
    Send a batch of pre-filtered sentences to the AI for structured extraction.
    Returns a flat list of extraction records.
    """
    if not sentences:
        return []

    n = len(sentences)
    numbered = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
    prompt = EXTRACT_PROMPT_TMPL.format(n_sentences=n, bulk_sentences=numbered)

    raw = ollama_call(prompt, EXTRACT_SYSTEM)
    if not raw:
        return []
    parsed = parse_json_response(raw)
    if not parsed:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]

    results = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        target = item.get("target_name")
        if not target:
            continue
        try:
            item["confidence"] = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            item["confidence"] = 0.6
        results.append(item)
    return results


def ai_extract_batch_with_retry(sentences: list[str], config: dict,
                                 min_records: int = 2,
                                 max_retries: int = 2) -> tuple[list[dict], int]:
    """
    Extract from a batch with automatic retry on low yield.
    If the model returns fewer than min_records, split the batch in half and retry.
    Returns (extractions, retries_used).
    """
    retries_used = 0

    # Try at current batch size
    results = ai_extract_batch(sentences, config)
    retries_used = 0

    # Low yield detection: if we got very few results relative to sentences,
    # and the batch has enough sentences to benefit from splitting
    should_retry = (
        len(sentences) >= 6  # only retry for batches large enough to split
        and len(results) < min_records  # suspiciously few extractions
        and len(results) == 0  # nothing found at all
    )

    if should_retry:
        for attempt in range(max_retries):
            retries_used += 1
            half = len(sentences) // 2

            # Try first half
            results_first = ai_extract_batch(sentences[:half], config)
            # Try second half
            results_second = ai_extract_batch(sentences[half:], config)

            combined = results_first + results_second
            if len(combined) > len(results):
                results = combined
                break

    return results, retries_used


def ai_extract_parallel(sentence_batches: list[list[str]], config: dict,
                         min_records_per_batch: int = 2) -> list[dict]:
    """
    Process multiple batches in PARALLEL using concurrent futures.
    Returns combined list of all extractions from all batches.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_records = []

    # Use ThreadPoolExecutor to run batches in parallel
    # Max workers = min of (4 or len(batches)) to avoid overwhelming Ollama
    max_workers = min(4, len(sentence_batches))

    def process_single_batch(batch):
        """Process one batch with retry logic."""
        records, retries = ai_extract_batch_with_retry(
            batch, config,
            min_records=min_records_per_batch,
            max_retries=1  # 1 retry split on failure
        )
        return records, retries

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_batch, batch): i
                   for i, batch in enumerate(sentence_batches)}

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                records, retries = future.result()
                all_records.extend(records)
            except Exception as exc:
                log.warning("Batch %d generated an exception: %s", batch_idx, exc)

    return all_records

# ─────────────────────────────────────────────
#  Stage 4 — Persistence
# ─────────────────────────────────────────────
def ensure_dirs():
    for d in [JSON_DIR, CSV_DIR, HTML_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def save_file_json(pubmed_id: str, filename: str, records: list[dict]):
    out = {
        "pubmed_id": pubmed_id,
        "source_file": filename,
        "extracted_at": datetime.now().isoformat(),
        "records": records,
    }
    path = JSON_DIR / f"{pubmed_id}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

def build_dataframes(all_records: list[dict]):
    """Split records into compound-level and interaction-level dataframes."""
    compound_rows = []
    interaction_rows = []

    for r in all_records:
        base = {
            "pubmed_id": r.get("pubmed_id", ""),
            "source_file": r.get("source_file", ""),
            "target_name": r.get("target_name", ""),
            "interaction_type": r.get("interaction_type", ""),
            "mechanism": r.get("mechanism", ""),
            "target_species": r.get("target_species", ""),
            "confidence": r.get("confidence", ""),
            "IC50": (r.get("quantitative_values") or {}).get("IC50", ""),
            "Ki": (r.get("quantitative_values") or {}).get("Ki", ""),
            "EC50": (r.get("quantitative_values") or {}).get("EC50", ""),
            "Kd": (r.get("quantitative_values") or {}).get("Kd", ""),
            "potency": (r.get("quantitative_values") or {}).get("potency", ""),
            "selectivity": (r.get("quantitative_values") or {}).get("selectivity", ""),
            "evidence_sentence": r.get("evidence_sentence", ""),
            "notes": r.get("notes", ""),
            "extracted_at": r.get("extracted_at", ""),
        }

        compound = r.get("compound_name")
        if compound and str(compound).lower() not in ("null", "none", ""):
            compound_rows.append({"compound_name": compound, **base})
        else:
            interaction_rows.append(base)

    df_compound = pd.DataFrame(compound_rows)
    df_interaction = pd.DataFrame(interaction_rows)
    return df_compound, df_interaction

# ─────────────────────────────────────────────
#  Main Processing Loop
# ─────────────────────────────────────────────
def process_folder(folder: Path, config: dict, memory: dict, batch_size: int = 8) -> list[dict]:
    txt_files = sorted(folder.glob("**/*.txt"))
    log.info("Found %d .txt files in %s", len(txt_files), folder)
    log.info("Batch size: %d sentences per AI call", batch_size)

    # Build dynamic patterns from config
    tracker.set_stage(1, "Building Patterns")
    target_pattern = build_target_patterns(config)
    interaction_tokens = build_interaction_pattern(config)
    mechanism_tokens = build_mechanism_pattern(config)
    extra_keywords = build_extra_keywords(config)

    log.info("  Target pattern: %s", config.get("targets", "WILDCARD"))
    log.info("  Interaction tokens (%d): %s", len(interaction_tokens),
             interaction_tokens[:5] + ["..."])
    log.info("  Mechanism tokens (%d): %s", len(mechanism_tokens),
             mechanism_tokens[:5] + ["..."])

    # Start pipeline tracking
    tracker.start_pipeline(len(txt_files))
    tracker.set_stage(2, "Scanning Files")

    all_records = []
    stats = {
        "files_processed": 0,
        "sentences_scanned": 0,
        "sentences_filtered": 0,
        "sentences_extracted": 0,
        "records_compound": 0,
        "records_interaction": 0,
        "files_with_hits": 0,
        "total_batches": 0,
    }

    for txt_path in txt_files:
        pubmed_id = re.findall(r"\d+", txt_path.stem)
        pubmed_id = pubmed_id[0] if pubmed_id else "UNKNOWN"

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        sentences = split_sentences(text)
        stats["sentences_scanned"] += len(sentences)

        # Start tracking this file
        tracker.start_file(txt_path.name, len(sentences))
        tracker.update_file_progress(0, 0, len(all_records))

        candidates = [
            s for s in sentences
            if sentence_is_candidate(
                s, target_pattern, interaction_tokens,
                mechanism_tokens, extra_keywords, config, memory
            )
        ]
        stats["sentences_filtered"] += len(candidates)
        tracker.update_file_progress(len(sentences), len(candidates), len(all_records))

        file_records = []

        # Split candidates into batches upfront
        batches = [
            candidates[i:i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

        if not candidates:
            tracker.update_file_progress(len(sentences), 0, len(all_records))
        else:
            # Process all batches in PARALLEL
            # ai_extract_parallel handles retries internally
            file_records = ai_extract_parallel(batches, config, min_records_per_batch=1)

            # Deduplicate by evidence_sentence (in case overlap caused duplicates)
            seen = set()
            deduped = []
            for ex in file_records:
                key = ex.get("evidence_sentence", "")
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(ex)
            file_records = deduped

            # Enrich records with metadata and update memory
            for ex in file_records:
                ex["pubmed_id"] = pubmed_id
                ex["source_file"] = txt_path.name
                ex["extracted_at"] = datetime.now().isoformat()
                compound = ex.get("compound_name")
                if compound and str(compound).lower() not in ("null", "none", ""):
                    if compound not in memory["compounds"]:
                        memory["compounds"].append(compound)
                target = ex.get("target_name")
                if target:
                    itype = ex.get("interaction_type", "")
                    if itype not in memory["targets"].get(target, []):
                        memory["targets"].setdefault(target, []).append(itype)

        if file_records:
            save_file_json(pubmed_id, txt_path.name, file_records)
            stats["files_with_hits"] += 1
            all_records.extend(file_records)
            log.info("  ✓ %s → %d records (%d batches)", txt_path.name, len(file_records),
                     (len(candidates) + batch_size - 1) // batch_size)
        else:
            log.debug("  ✗ %s → no records", txt_path.name)

        tracker.finish_file(len(file_records))
        tracker.display(force=True)
        stats["files_processed"] += 1

    stats["sentences_extracted"] = len(all_records)
    stats["records_compound"] = sum(
        1 for r in all_records
        if r.get("compound_name") and
        str(r["compound_name"]).lower() not in ("null", "none", "")
    )
    stats["records_interaction"] = stats["sentences_extracted"] - stats["records_compound"]

    log.info("\n=== Extraction Statistics ===")
    for k, v in stats.items():
        log.info("  %-28s %s", k, v)

    memory["stats"] = stats
    memory["last_run"] = datetime.now().isoformat()
    memory["query_config"] = config
    return all_records

# ─────────────────────────────────────────────
#  Stage 5 — Dashboard Generator
# ─────────────────────────────────────────────
def generate_dashboard(df_compound: pd.DataFrame, df_interaction: pd.DataFrame,
                       config: dict, memory: dict):
    """Generate a self-contained HTML dashboard."""

    comp_json = df_compound.to_json(orient="records") if not df_compound.empty else "[]"
    intr_json = df_interaction.to_json(orient="records") if not df_interaction.empty else "[]"

    # Summary stats
    target_counts = defaultdict(int)
    interaction_counts = defaultdict(int)
    for r in df_compound.to_dict("records") + df_interaction.to_dict("records"):
        t = r.get("target_name", "Unknown") or "Unknown"
        i = r.get("interaction_type", "Unknown") or "Unknown"
        target_counts[t] += 1
        interaction_counts[i] += 1

    stats = memory.get("stats", {})
    query_summary = config.get("query_summary", "Biomedical target interactions")
    timestamp = memory.get("last_run", datetime.now().isoformat())

    targets = config.get("targets", [])
    target_names = [t.get("name") if isinstance(t, dict) else t for t in targets]

    top_compounds = {}
    if not df_compound.empty and "compound_name" in df_compound.columns:
        top_compounds = (df_compound["compound_name"]
                         .value_counts().head(15)
                         .to_dict())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Biomedical NLP Extractor — Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:        #0a0e1a;
    --panel:     #111827;
    --border:    #1e2d45;
    --accent1:   #00d4ff;
    --accent2:   #7c3aed;
    --accent3:   #10b981;
    --accent4:   #f59e0b;
    --danger:    #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
  }}

  .header {{
    background: linear-gradient(135deg, #0f172a 0%, #1a1040 50%, #0a1628 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -10%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0,212,255,.06) 0%, transparent 60%);
    pointer-events: none;
  }}
  .header-grid {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 24px;
    max-width: 1600px;
    margin: 0 auto;
  }}
  .header-title {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: var(--accent1);
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .header h1 {{
    font-size: 26px;
    font-weight: 600;
    color: #fff;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }}
  .header .subtitle {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 13px;
    max-width: 600px;
  }}
  .header-meta {{
    text-align: right;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    line-height: 1.8;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}
  .badge-cyan   {{ background: rgba(0,212,255,.12); color: var(--accent1); border: 1px solid rgba(0,212,255,.3); }}
  .badge-purple {{ background: rgba(124,58,237,.15); color: #a78bfa; border: 1px solid rgba(124,58,237,.3); }}
  .badge-green  {{ background: rgba(16,185,129,.12); color: var(--accent3); border: 1px solid rgba(16,185,129,.3); }}

  .main {{
    max-width: 1600px;
    margin: 0 auto;
    padding: 32px 40px;
  }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .stat-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s, transform .2s;
  }}
  .stat-card:hover {{ border-color: var(--accent1); transform: translateY(-2px); }}
  .stat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color, var(--accent1));
  }}
  .stat-label {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .stat-value {{
    font-family: var(--mono);
    font-size: 32px;
    font-weight: 600;
    color: #fff;
    line-height: 1;
  }}
  .stat-sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 20px;
    margin-bottom: 32px;
  }}
  .chart-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
  }}
  .chart-title {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent1);
    margin-bottom: 16px;
  }}
  .chart-wrap {{ position: relative; height: 260px; }}

  .table-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 24px;
    overflow: hidden;
  }}
  .table-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    gap: 16px;
    flex-wrap: wrap;
  }}
  .table-title {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent1);
  }}
  .table-controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .search-box {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 7px 14px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    width: 240px;
    outline: none;
    transition: border-color .2s;
  }}
  .search-box:focus {{ border-color: var(--accent1); }}
  .search-box::placeholder {{ color: var(--muted); }}
  select.filter-select {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 7px 12px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    outline: none;
    cursor: pointer;
  }}
  .btn {{
    padding: 7px 16px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    border: none;
    transition: all .2s;
  }}
  .btn-cyan    {{ background: var(--accent1); color: #000; }}
  .btn-cyan:hover  {{ background: #00b8e0; }}
  .btn-purple  {{ background: var(--accent2); color: #fff; }}
  .btn-purple:hover {{ background: #6d28d9; }}
  .btn-outline {{
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
  }}
  .btn-outline:hover {{ border-color: var(--accent1); color: var(--accent1); }}

  .table-wrap {{ overflow-x: auto; max-height: 500px; overflow-y: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead tr {{ background: #0d1626; position: sticky; top: 0; z-index: 2; }}
  th {{
    padding: 10px 14px;
    text-align: left;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
  }}
  th:hover {{ color: var(--accent1); }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid rgba(30,45,69,.6);
    vertical-align: top;
    max-width: 340px;
  }}
  tr:hover td {{ background: rgba(0,212,255,.03); }}
  .evidence-cell {{
    max-width: 400px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: pointer;
    color: var(--muted);
  }}
  .evidence-cell:hover {{ white-space: normal; color: var(--text); }}

  .tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
  }}
  .tag-inh  {{ background: rgba(239,68,68,.15); color: #fca5a5; }}
  .tag-act  {{ background: rgba(16,185,129,.15); color: #6ee7b7; }}
  .tag-sub  {{ background: rgba(100,116,139,.15); color: #94a3b8; }}
  .tag-mod  {{ background: rgba(124,58,237,.15); color: #c4b5fd; }}
  .tag-unk  {{ background: rgba(100,116,139,.1); color: #64748b; }}
  .tag-compet {{ background: rgba(0,212,255,.12); color: #67e8f9; }}
  .tag-allo  {{ background: rgba(245,158,11,.15); color: #fcd34d; }}
  .tag-irrev {{ background: rgba(239,68,68,.15); color: #fca5a5; }}
  .tag-rev   {{ background: rgba(59,130,246,.15); color: #93c5fd; }}

  .conf-bar-wrap {{ background: rgba(255,255,255,.05); border-radius: 2px; height: 4px; width: 60px; }}
  .conf-bar {{ height: 4px; border-radius: 2px; background: var(--accent3); }}

  .pagination {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 24px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }}
  .page-btns {{ display: flex; gap: 6px; }}
  .page-btn {{
    padding: 4px 10px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11px;
    transition: all .15s;
  }}
  .page-btn:hover, .page-btn.active {{ border-color: var(--accent1); color: var(--accent1); }}
  .page-btn:disabled {{ opacity: .3; cursor: not-allowed; }}

  .query-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 24px;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.8;
  }}
  .query-panel strong {{ color: var(--accent1); }}

  .footer {{
    text-align: center;
    padding: 24px 40px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }}

  @media (max-width: 768px) {{
    .main {{ padding: 20px 16px; }}
    .header {{ padding: 20px 16px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-grid">
    <div>
      <div class="header-title">Biomedical NLP Knowledge Extractor v3.0</div>
      <h1>Target Interaction<br>Extraction Dashboard</h1>
      <p class="subtitle">{query_summary}</p>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
        <span class="badge badge-cyan">AI-Augmented</span>
        <span class="badge badge-purple">Dynamic Target Matching</span>
        <span class="badge badge-green">Publication Grade</span>
      </div>
    </div>
    <div class="header-meta">
      <div>Generated: {timestamp[:19].replace("T"," ")}</div>
      <div>Model: {LLM_MODEL}</div>
      <div>Files: {stats.get("files_processed",0):,}</div>
      <div>Hits: {stats.get("files_with_hits",0):,}</div>
    </div>
  </div>
</div>

<div class="main">

  <!-- Query Config -->
  <div class="query-panel">
    <strong>EXTRACTION QUERY</strong> &nbsp;|&nbsp; {query_summary}<br>
    <strong>TARGETS</strong> &nbsp;|&nbsp; {", ".join(target_names) if target_names else "All targets (wildcard)"}<br>
    <strong>INTERACTION TYPES</strong> &nbsp;|&nbsp; {", ".join(config.get("interaction_types", ["All"])) or "All"}<br>
    <strong>MECHANISMS</strong> &nbsp;|&nbsp; {", ".join(config.get("mechanisms", ["All"])) or "All"}<br>
    <strong>EXTRA KEYWORDS</strong> &nbsp;|&nbsp; {", ".join(config.get("extra_keywords", ["-"])) or "-"}
  </div>

  <!-- Stat Cards -->
  <div class="stats-grid">
    <div class="stat-card" style="--accent-color:#00d4ff">
      <div class="stat-label">Files Processed</div>
      <div class="stat-value">{stats.get("files_processed",0):,}</div>
      <div class="stat-sub">.txt abstracts</div>
    </div>
    <div class="stat-card" style="--accent-color:#7c3aed">
      <div class="stat-label">Sentences Scanned</div>
      <div class="stat-value">{stats.get("sentences_scanned",0):,}</div>
      <div class="stat-sub">total sentences</div>
    </div>
    <div class="stat-card" style="--accent-color:#f59e0b">
      <div class="stat-label">Sentences Filtered</div>
      <div class="stat-value">{stats.get("sentences_filtered",0):,}</div>
      <div class="stat-sub">passed pre-filter</div>
    </div>
    <div class="stat-card" style="--accent-color:#10b981">
      <div class="stat-label">Compound Records</div>
      <div class="stat-value">{stats.get("records_compound",0):,}</div>
      <div class="stat-sub">compound + target + interaction</div>
    </div>
    <div class="stat-card" style="--accent-color:#ef4444">
      <div class="stat-label">Interaction Records</div>
      <div class="stat-value">{stats.get("records_interaction",0):,}</div>
      <div class="stat-sub">target + interaction only</div>
    </div>
    <div class="stat-card" style="--accent-color:#06b6d4">
      <div class="stat-label">Files w/ Hits</div>
      <div class="stat-value">{stats.get("files_with_hits",0):,}</div>
      <div class="stat-sub">yielded extractions</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-grid">
    <div class="chart-panel">
      <div class="chart-title">Extractions by Target</div>
      <div class="chart-wrap"><canvas id="targetChart"></canvas></div>
    </div>
    <div class="chart-panel">
      <div class="chart-title">Interaction Type Distribution</div>
      <div class="chart-wrap"><canvas id="interactionChart"></canvas></div>
    </div>
    <div class="chart-panel">
      <div class="chart-title">Top 15 Compounds</div>
      <div class="chart-wrap"><canvas id="compoundChart"></canvas></div>
    </div>
  </div>

  <!-- Compound Table -->
  <div class="table-panel">
    <div class="table-header">
      <div>
        <div class="table-title">Compound-Level Extractions (Table A)</div>
        <div style="font-size:11px;color:var(--muted);margin-top:3px;">Rows with compound + target + interaction</div>
      </div>
      <div class="table-controls">
        <input class="search-box" id="compSearch" type="text" placeholder="Search compound, target, interaction...">
        <select class="filter-select" id="compTargetFilter">
          <option value="">All Targets</option>
        </select>
        <select class="filter-select" id="compInteractionFilter">
          <option value="">All Interactions</option>
        </select>
        <button class="btn btn-cyan" onclick="downloadCSV('compound')">↓ CSV</button>
        <button class="btn btn-outline" onclick="downloadJSON('compound')">↓ JSON</button>
      </div>
    </div>
    <div class="table-wrap">
      <table id="compoundTable">
        <thead>
          <tr>
            <th onclick="sortTable('compound',0)">PubMed ID</th>
            <th onclick="sortTable('compound',1)">Compound</th>
            <th onclick="sortTable('compound',2)">Target</th>
            <th onclick="sortTable('compound',3)">Interaction</th>
            <th onclick="sortTable('compound',4)">Mechanism</th>
            <th onclick="sortTable('compound',5)">IC50</th>
            <th onclick="sortTable('compound',6)">Ki</th>
            <th onclick="sortTable('compound',7)">Conf.</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody id="compoundBody"></tbody>
      </table>
    </div>
    <div class="pagination">
      <span id="compInfo"></span>
      <div class="page-btns" id="compPages"></div>
    </div>
  </div>

  <!-- Interaction Table -->
  <div class="table-panel">
    <div class="table-header">
      <div>
        <div class="table-title">Interaction-Level Extractions (Table B)</div>
        <div style="font-size:11px;color:var(--muted);margin-top:3px;">Rows with target + interaction (no specific compound)</div>
      </div>
      <div class="table-controls">
        <input class="search-box" id="intrSearch" type="text" placeholder="Search target, interaction...">
        <select class="filter-select" id="intrTargetFilter">
          <option value="">All Targets</option>
        </select>
        <button class="btn btn-purple" onclick="downloadCSV('interaction')">↓ CSV</button>
        <button class="btn btn-outline" onclick="downloadJSON('interaction')">↓ JSON</button>
      </div>
    </div>
    <div class="table-wrap">
      <table id="interactionTable">
        <thead>
          <tr>
            <th onclick="sortTable('interaction',0)">PubMed ID</th>
            <th onclick="sortTable('interaction',1)">Target</th>
            <th onclick="sortTable('interaction',2)">Interaction</th>
            <th onclick="sortTable('interaction',3)">Mechanism</th>
            <th onclick="sortTable('interaction',4)">Conf.</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody id="interactionBody"></tbody>
      </table>
    </div>
    <div class="pagination">
      <span id="intrInfo"></span>
      <div class="page-btns" id="intrPages"></div>
    </div>
  </div>

</div><!-- /main -->

<div class="footer">
  Biomedical NLP Extractor v3.0 &nbsp;·&nbsp; AI-Augmented Biomedical Literature Mining &nbsp;·&nbsp;
  Generated {timestamp[:10]}
</div>

<script>
const COMPOUND_DATA  = {comp_json};
const INTERACTION_DATA = {intr_json};
const TARGET_COUNTS   = {json.dumps(dict(target_counts))};
const INTERACTION_COUNTS = {json.dumps(dict(interaction_counts))};
const TOP_COMPOUNDS   = {json.dumps(top_compounds)};

function interactionTag(i) {{
  const map = {{
    'inhibition': 'tag-inh', 'inhibitor': 'tag-inh',
    'activation': 'tag-act', 'activator': 'tag-act',
    'substrate': 'tag-sub',
    'modulation': 'tag-mod', 'modulator': 'tag-mod',
    'antagonist': 'tag-mod', 'agonist': 'tag-act',
    'binding': 'tag-mod',
  }};
  const cls = map[i] || 'tag-unk';
  return `<span class="tag ${{cls}}">${{i || '—'}}</span>`;
}}

function mechanismTag(m) {{
  if (!m) return '<span class="tag tag-unk">—</span>';
  const lower = m.toLowerCase();
  if (lower.includes('compet')) return '<span class="tag tag-compet">${{m}}</span>';
  if (lower.includes('alloster')) return '<span class="tag tag-allo">${{m}}</span>';
  if (lower.includes('irrevers') || lower.includes('covalent')) return '<span class="tag tag-irrev">${{m}}</span>';
  if (lower.includes('revers')) return '<span class="tag tag-rev">${{m}}</span>';
  return `<span class="tag tag-unk">${{m}}</span>`;
}}

function confBar(v) {{
  const pct = Math.round((v || 0) * 100);
  const clr = pct >= 85 ? '#10b981' : pct >= 65 ? '#f59e0b' : '#ef4444';
  return `<div style="display:flex;align-items:center;gap:6px;">
    <div class="conf-bar-wrap"><div class="conf-bar" style="width:${{pct}}%;background:${{clr}}"></div></div>
    <span style="font-family:var(--mono);font-size:10px;color:var(--muted)">${{pct}}%</span>
  </div>`;
}}

const tableState = {{}};

function initTable(name, data, renderRow, bodyId, infoId, pagesId) {{
  tableState[name] = {{
    all: data,
    filtered: [...data],
    page: 1,
    pageSize: 25,
    renderRow,
    bodyId, infoId, pagesId
  }};
  renderTablePage(name);
}}

function renderTablePage(name) {{
  const s = tableState[name];
  const start = (s.page - 1) * s.pageSize;
  const rows  = s.filtered.slice(start, start + s.pageSize);
  document.getElementById(s.bodyId).innerHTML = rows.map(s.renderRow).join('');

  const total = s.filtered.length;
  const pages = Math.ceil(total / s.pageSize);
  document.getElementById(s.infoId).textContent =
    `Showing ${{start+1}}–${{Math.min(start+s.pageSize, total)}} of ${{total}} records`;

  const pagesEl = document.getElementById(s.pagesId);
  let btns = `<button class="page-btn" onclick="changePage('${{name}}',-1)" ${{s.page===1?'disabled':''}}>‹</button>`;
  const lo = Math.max(1, s.page-2), hi = Math.min(pages, s.page+2);
  if (lo > 1) btns += `<button class="page-btn" onclick="setPage('${{name}}',1)">1</button>`;
  if (lo > 2) btns += `<span style="color:var(--muted);padding:0 4px">…</span>`;
  for (let i=lo;i<=hi;i++)
    btns += `<button class="page-btn ${{i===s.page?'active':''}}" onclick="setPage('${{name}}',${{i}})">${{i}}</button>`;
  if (hi < pages-1) btns += `<span style="color:var(--muted);padding:0 4px">…</span>`;
  if (hi < pages) btns += `<button class="page-btn" onclick="setPage('${{name}}',${{pages}})">${{pages}}</button>`;
  btns += `<button class="page-btn" onclick="changePage('${{name}}',1)" ${{s.page===pages||pages===0?'disabled':''}}>›</button>`;
  pagesEl.innerHTML = btns;
}}

function changePage(name, delta) {{
  const s = tableState[name];
  const pages = Math.ceil(s.filtered.length / s.pageSize);
  s.page = Math.max(1, Math.min(pages, s.page + delta));
  renderTablePage(name);
}}
function setPage(name, n) {{
  tableState[name].page = n;
  renderTablePage(name);
}}

function filterTable(name, text, target, interaction) {{
  const s = tableState[name];
  const q = (text || '').toLowerCase();
  s.filtered = s.all.filter(r => {{
    const str = JSON.stringify(r).toLowerCase();
    if (q && !str.includes(q)) return false;
    if (target && r.target_name !== target) return false;
    if (interaction && r.interaction_type !== interaction) return false;
    return true;
  }});
  s.page = 1;
  renderTablePage(name);
}}

function renderCompoundRow(r) {{
  const ev = (r.evidence_sentence || '').replace(/"/g,'&quot;');
  return `<tr>
    <td style="font-family:var(--mono);color:var(--accent1)">${{r.pubmed_id||'—'}}</td>
    <td style="font-weight:500">${{r.compound_name||'—'}}</td>
    <td style="font-family:var(--mono);color:var(--accent3)">${{r.target_name||'—'}}</td>
    <td>${{interactionTag(r.interaction_type)}}</td>
    <td>${{mechanismTag(r.mechanism)}}</td>
    <td style="font-family:var(--mono);font-size:11px">${{r.IC50||'—'}}</td>
    <td style="font-family:var(--mono);font-size:11px">${{r.Ki||'—'}}</td>
    <td>${{confBar(r.confidence)}}</td>
    <td class="evidence-cell" title="${{ev}}">${{(r.evidence_sentence||'').substring(0,90)}}…</td>
  </tr>`;
}}

function renderInteractionRow(r) {{
  const ev = (r.evidence_sentence || '').replace(/"/g,'&quot;');
  return `<tr>
    <td style="font-family:var(--mono);color:var(--accent1)">${{r.pubmed_id||'—'}}</td>
    <td style="font-family:var(--mono);color:var(--accent3)">${{r.target_name||'—'}}</td>
    <td>${{interactionTag(r.interaction_type)}}</td>
    <td>${{mechanismTag(r.mechanism)}}</td>
    <td>${{confBar(r.confidence)}}</td>
    <td class="evidence-cell" title="${{ev}}">${{(r.evidence_sentence||'').substring(0,120)}}…</td>
  </tr>`;
}}

function populateFilters(name, data, targetId, interactionId) {{
  const targets = [...new Set(data.map(r => r.target_name).filter(Boolean))].sort();
  const interactions = [...new Set(data.map(r => r.interaction_type).filter(Boolean))].sort();
  const targetEl = document.getElementById(targetId);
  const intrEl = interactionId ? document.getElementById(interactionId) : null;
  if (targetEl) targets.forEach(t => targetEl.innerHTML += `<option value="${{t}}">${{t}}</option>`);
  if (intrEl) interactions.forEach(i => intrEl.innerHTML += `<option value="${{i}}">${{i}}</option>`);
}}

function downloadCSV(which) {{
  const data = which === 'compound' ? COMPOUND_DATA : INTERACTION_DATA;
  if (!data.length) {{ alert('No data to download'); return; }}
  const keys = Object.keys(data[0]);
  const lines = [keys.join(',')].concat(
    data.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','))
  );
  const blob = new Blob([lines.join('\\n')], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${{which}}_extractions.csv`;
  a.click();
}}

function downloadJSON(which) {{
  const data = which === 'compound' ? COMPOUND_DATA : INTERACTION_DATA;
  const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${{which}}_extractions.json`;
  a.click();
}}

const PALETTE = [
  '#00d4ff','#7c3aed','#10b981','#f59e0b','#ef4444','#06b6d4',
  '#8b5cf6','#34d399','#fbbf24','#f87171','#22d3ee','#a78bfa',
];

function makeBarChart(id, labels, values, label) {{
  const ctx = document.getElementById(id).getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label,
        data: values,
        backgroundColor: PALETTE.slice(0, labels.length),
        borderRadius: 4,
        borderSkipped: false,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.parsed.y}} records` }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#64748b', font: {{ family: 'IBM Plex Mono', size: 10 }} }},
              grid: {{ color: 'rgba(30,45,69,.4)' }} }},
        y: {{ ticks: {{ color: '#64748b', font: {{ family: 'IBM Plex Mono', size: 10 }} }},
              grid: {{ color: 'rgba(30,45,69,.4)' }}, beginAtZero: true }}
      }}
    }}
  }});
}}

function makeDoughnutChart(id, labels, values) {{
  const ctx = document.getElementById(id).getContext('2d');
  new Chart(ctx, {{
    type: 'doughnut',
    data: {{ labels, datasets: [{{ data: values, backgroundColor: PALETTE.slice(0, labels.length), borderColor: '#0a0e1a', borderWidth: 2, hoverOffset: 8 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, cutout: '62%', plugins: {{ legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ family: 'IBM Plex Mono', size: 10 }}, boxWidth: 12, padding: 10 }} }} }} }}
  }});
}}

document.addEventListener('DOMContentLoaded', () => {{
  initTable('compound',  COMPOUND_DATA,  renderCompoundRow,  'compoundBody',  'compInfo',  'compPages');
  initTable('interaction', INTERACTION_DATA, renderInteractionRow, 'interactionBody', 'intrInfo', 'intrPages');

  populateFilters('compound',  COMPOUND_DATA,  'compTargetFilter', 'compInteractionFilter');
  populateFilters('interaction', INTERACTION_DATA, 'intrTargetFilter', null);

  document.getElementById('compSearch').addEventListener('input', e =>
    filterTable('compound', e.target.value,
      document.getElementById('compTargetFilter').value,
      document.getElementById('compInteractionFilter').value));
  document.getElementById('compTargetFilter').addEventListener('change', e =>
    filterTable('compound', document.getElementById('compSearch').value,
      e.target.value, document.getElementById('compInteractionFilter').value));
  document.getElementById('compInteractionFilter').addEventListener('change', e =>
    filterTable('compound', document.getElementById('compSearch').value,
      document.getElementById('compTargetFilter').value, e.target.value));

  document.getElementById('intrSearch').addEventListener('input', e =>
    filterTable('interaction', e.target.value,
      document.getElementById('intrTargetFilter').value, null));
  document.getElementById('intrTargetFilter').addEventListener('change', e =>
    filterTable('interaction', document.getElementById('intrSearch').value,
      e.target.value, null));

  const targetLabels = Object.keys(TARGET_COUNTS).sort((a,b) => TARGET_COUNTS[b]-TARGET_COUNTS[a]).slice(0,12);
  const targetVals = targetLabels.map(k => TARGET_COUNTS[k]);
  makeBarChart('targetChart', targetLabels, targetVals, 'Records');

  const intrLabels = Object.keys(INTERACTION_COUNTS);
  const intrVals = intrLabels.map(k => INTERACTION_COUNTS[k]);
  makeDoughnutChart('interactionChart', intrLabels, intrVals);

  const compLabels = Object.keys(TOP_COMPOUNDS);
  const compVals = compLabels.map(k => TOP_COMPOUNDS[k]);
  makeBarChart('compoundChart', compLabels, compVals, 'Occurrences');
}});
</script>
</body>
</html>"""

    with open(OUT_DASHBOARD, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Dashboard written → %s", OUT_DASHBOARD)

# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(
        description="Biomedical NLP Knowledge Extractor — Publication-Grade Pipeline v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Examples:
            python biomedical_nlp_pipeline.py --folder ./abstracts
            python biomedical_nlp_pipeline.py --folder ./abstracts --query "SIRT1 activators" --batch-size 10
            python biomedical_nlp_pipeline.py --folder ./abstracts --query "BACE1 inhibitors" --batch-size 12
            python biomedical_nlp_pipeline.py --folder ./abstracts --model llama3.1:8b --batch-size 8 --no-prompt

          Batch size guide:
            3-5:  Higher accuracy per extraction, slower
            8-10: Good balance (default)
            15-20: Faster, still accurate with good LLMs
        """),
    )
    parser.add_argument("--folder",   required=True,  help="Folder containing .txt abstract files")
    parser.add_argument("--query",    default="",      help="Natural-language extraction query (e.g., 'SIRT1 activators', 'CYP3A4 inhibitors')")
    parser.add_argument("--model",    default=LLM_MODEL, help=f"Ollama model name (default: {LLM_MODEL})")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of sentences per AI batch call (default: 8, increase for faster processing)")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip AI query analysis and use permissive defaults (all targets)")
    args = parser.parse_args()

    LLM_MODEL = args.model

    ensure_dirs()
    memory = load_memory()

    print("\n" + "═"*60)
    print("  Biomedical NLP Knowledge Extractor — v3.0")
    print("  AI-Augmented Biomedical Literature Mining")
    print("═"*60 + "\n")

    # ── Stage 1: Query Analysis ──
    tracker.set_stage(0, "Query Analysis")
    user_query = args.query
    if not user_query and not args.no_prompt:
        print("Enter your extraction query (or press Enter for all targets):")
        print("  Examples:")
        print("    'SIRT1 activators and their mechanisms'")
        print("    'CYP3A4 time-dependent inhibitors with IC50 values'")
        print("    'BACE1 inhibitors in Alzheimer disease'")
        print("    'TNF-alpha antagonists in rheumatoid arthritis'")
        user_query = input("  › ").strip()
        if not user_query:
            user_query = "Extract all biomedical target interaction data including compounds, mechanisms, and quantitative values"

    if args.no_prompt or not user_query:
        config = {
            "targets": [],
            "interaction_types": [],
            "mechanisms": [],
            "compound_hints": [],
            "extra_keywords": [],
            "query_summary": "All biomedical target interactions (wildcard mode)",
            "species_context": "",
        }
    else:
        config = analyze_query(user_query)

    # ── Stage 2 + 3: Process ──
    folder = Path(args.folder)
    if not folder.exists():
        log.error("Folder not found: %s", folder)
        sys.exit(1)

    all_records = process_folder(folder, config, memory, batch_size=args.batch_size)

    if not all_records:
        log.warning("No records extracted. Try broadening your query or check your folder contents.")

    # ── Stage 4: Save ──
    tracker.set_stage(3, "Saving Results")
    log.info("\nSaving master JSON...")
    with open(OUT_JSON_MASTER, "w") as f:
        json.dump({
            "config": config,
            "records": all_records,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2)

    df_compound, df_interaction = build_dataframes(all_records)

    df_compound.to_csv(OUT_CSV_COMPOUND, index=False)
    df_interaction.to_csv(OUT_CSV_INTERACTION, index=False)
    log.info("CSV A (compound):    %s  [%d rows]", OUT_CSV_COMPOUND, len(df_compound))
    log.info("CSV B (interaction):  %s  [%d rows]", OUT_CSV_INTERACTION, len(df_interaction))

    save_memory(memory)

    # ── Stage 5: Dashboard ──
    tracker.set_stage(4, "Generating Dashboard")
    log.info("\nGenerating HTML dashboard...")
    generate_dashboard(df_compound, df_interaction, config, memory)

    tracker.finish()

    print(f"""
  Records (compound):   {len(df_compound)}
  Records (interaction):{len(df_interaction)}
  CSV A:  {OUT_CSV_COMPOUND}
  CSV B:  {OUT_CSV_INTERACTION}
  Master JSON: {OUT_JSON_MASTER}
  Dashboard:   {OUT_DASHBOARD}
  Open dashboard.html in your browser to explore & download results.
""")


if __name__ == "__main__":
    main()
