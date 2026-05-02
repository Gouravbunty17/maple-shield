# Maple Shield — powered by CAIRN

**Canadian Sovereign Edge AI Intelligence System**

CAIRN is the internal detection engine behind Maple Shield. ARGUS is treated as the legacy project name; going forward, use **CAIRN detection engine** for the engine and **Maple Shield** for the product.

## Quick Links

- **CAIRN Engine v2:** [docs/CAIRN_ENGINE_V2.md](docs/CAIRN_ENGINE_V2.md)
- **Documentation:** [docs/README.md](docs/README.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Current Status

**Phase-1 Complete:** CPU validation baseline  
**CAIRN Engine v2:** Initial modular engine layer added  
**Active branch:** `feat/rebrand-to-maple-shield`

## What CAIRN adds

- Stable detection, risk, and frame schemas
- Configurable risk thresholds and weights
- Explainable risk factors, reasons, and operator actions
- JSON audit-ready frame records
- Adapters for existing Maple Shield MVP detections
- No-camera smoke demo for fast validation

## Smoke Test

```bash
python cairn_demo.py
```

With explicit config:

```bash
python cairn_demo.py --config configs/cairn.default.json
```

See [docs/CAIRN_ENGINE_V2.md](docs/CAIRN_ENGINE_V2.md) for the next integration steps.
