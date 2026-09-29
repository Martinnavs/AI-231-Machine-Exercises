"""UI site (feature `ui-site`): a FastAPI dashboard for the voice assistant.

A **separate service** from the voice pipeline (its own process / terminal
window). The model side talks to it **one-way over its HTTP API**
(`POST /api/command`, `POST /api/listening`); the UI never calls the model.
See `feature-engineering/ui-site/SPEC.md` for the design and
`services/requirements.md` for the original requirement.
"""
