"""Start trained host inference using its saved, explicit local configuration."""

import os
from pathlib import Path


def configure(path: Path):
    from dotenv import dotenv_values

    if not path.is_file():
        raise RuntimeError(f"missing trained-service configuration: {path}")
    values = dotenv_values(path)
    if any(values.get(key) != "model" for key in ("LPD_BACKEND", "LPR_BACKEND")):
        raise RuntimeError("trained-local requires both trained model backends")
    # Nx loads the root Compose .env first. Its container paths must not override
    # this host's explicit weight and media paths (uvicorn --env-file does not override).
    os.environ.update({key: value for key, value in values.items() if value is not None})


def main():
    configure(Path("artifacts/local-serving.env"))
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8100, workers=1)


if __name__ == "__main__":
    main()
