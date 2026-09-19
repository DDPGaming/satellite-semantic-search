# Agent Instructions

## General

This is a hackathon project.

Keep the implementation understandable to a first-year
computer science student.

Prefer simple, reliable solutions over unnecessary abstraction.

Do not introduce new frameworks or databases without approval.

Do not replace an existing technology choice without approval.

Do not rewrite working components unnecessarily.

Do not implement future milestones unless explicitly requested.

## Technology constraints

Use the technology stack specified in ARCHITECTURE.md.

Current defaults:

- Python
- Rasterio
- NumPy
- PyTorch
- FAISS
- SQLite
- FastAPI
- Streamlit
- pytest

## Data

Satellite imagery can be very large.

Never commit raw satellite imagery to Git.

Never commit model weights to Git unless explicitly approved.

Never commit generated vector indexes or local databases if
they are covered by .gitignore.

Preserve geospatial metadata.

## Offline requirement

The final system must be capable of operating without network
access after models, libraries, and datasets have been staged.

Do not introduce runtime dependencies on external APIs or
internet services without explicit approval.

## Development process

Before making a significant architectural change:

1. Explain the proposed change.
2. Explain why it is necessary.
3. Identify affected files.
4. Wait for approval.

For normal implementation tasks:

1. Inspect existing code.
2. Make the smallest change necessary.
3. Run tests.
4. Report what changed.
5. Do not modify unrelated components.
