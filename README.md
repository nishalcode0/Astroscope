# Astroscope

An open-source computational astronomical observatory designed to monitor, process, analyze, and catalog space telescope observations.

---

## 🔭 Current Scope

The initial telescope scope focuses strictly on space-based astronomical observatories:
* **Hubble Space Telescope (HST)**
* **James Webb Space Telescope (JWST)**

*(Note: Roman Space Telescope is out of scope).*

---

## 📌 Current Status: Session 1 — Foundation

**Astroscope is currently in foundational architecture mode.**

* Session 1 establishes the scientific Python packaging, abstract archive adapters (`HubbleAdapter`, `JWSTAdapter`), typed `Observation` model, sub-module package structure, CLI skeleton, and offline test suite.
* **No live astronomical discovery or automated anomaly detection is active yet.**
* **No network data downloads or MAST API queries occur during Session 1.**

---

## 🏗️ Architecture

Astroscope uses a decoupled adapter design pattern for astronomical archive interactions. Telescope-specific queries are encapsulated behind a unified interface:

```text
                     ArchiveAdapter
                           │
             ┌─────────────┴─────────────┐
             │                           │
       HubbleAdapter               JWSTAdapter
             │                           │
         Hubble/MAST                 JWST/MAST
             └─────────────┬─────────────┘
                           │
                      Observation
```

Both adapters translate mission metadata into a common typed `Observation` model representing astronomical coordinates, instrument parameters, and references to scientific data products without carrying pixel payloads into memory.

---

## 🗺️ Scientific Roadmap

```text
Archive discovery (Session 2)
        ↓
Data ingestion
        ↓
FITS processing
        ↓
Source detection
        ↓
Photometry
        ↓
Time-series analysis
        ↓
Statistical anomaly detection
        ↓
Machine learning
        ↓
Catalog cross-matching
        ↓
Candidate ranking
        ↓
Autonomous monitoring
```

*All stages after Session 1 Foundation are labeled as future work.*

---

## 💻 Development & Setup

### 1. Virtual Environment & Installation

Create a virtual environment and perform an editable package installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Running Tests

Run the offline pytest test suite:

```bash
pytest
```

### 3. Command-Line Interface

Invoke the Astroscope CLI:

```bash
astroscope --help
```

Available subcommands (placeholders for future scientific modules):

```bash
astroscope discover
astroscope ingest
astroscope process
astroscope analyze
astroscope candidates
```
