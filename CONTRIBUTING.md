# Contributing to Samanvaya

First off, thank you for considering contributing to Samanvaya! It's people like you that make this framework robust and production-ready for ISRO's lunar missions.

## Development Setup

1. Fork and clone the repository.
2. We highly recommend using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run tests to ensure everything works locally:
   ```bash
   pytest tests/
   ```

## Pull Request Process

1. Ensure any new dependencies are added to `requirements.txt` or `package.json`.
2. Ensure you have added necessary tests for your features/fixes. We enforce a high coverage standard (especially for the `src/security` and `src/features` layers).
3. If you modify any core algorithms (e.g., Phase Congruency, SubPixel Warper), you must include visual proof or metrics output verifying the change does not degrade RMSE (< 0.40 px).
4. Update the `README.md` with details of changes to the interface, if applicable.
5. Submit your PR and await maintainer review!

## Code Style

- **Python**: We follow PEP-8 and use type hints (`-> list[str]`, etc.) aggressively for all function signatures.
- **TypeScript/React**: We use ESLint and Prettier for the frontend.

## Architecture

Before contributing large features, please review our core architectural split:
- `src/features/` - Core mathematical algorithms (Phase Congruency, Log-Gabor filters)
- `src/matching/` - Feature matching (QuadTree ANMS, ASIFT)
- `src/registration/` - Non-linear warping and sub-pixel refinement
- `src/security/` - Zero-trust boundaries, RBAC, Merkle audit chains
- `src/api/` - FastAPI routes and job orchestration
- `ml_service/` - Isolated Machine Learning anomaly detection

Thank you for helping us map the Moon! 🚀🌕
