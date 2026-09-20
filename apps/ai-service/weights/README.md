# Model weights

Put detection weights in `lpd/` and recognition weights in `lpr/`, then point
the service at them:

```
LPD_BACKEND=model
LPD_WEIGHTS=/app/weights/lpd/best.pt
LPR_BACKEND=model
LPR_WEIGHTS=/app/weights/lpr/best.pt
```

`apps/ai-service` is bind-mounted at `/app` in compose, so a file dropped here
shows up at `/app/weights/...` without rebuilding the image.

The weight files themselves are git-ignored — they are too big for the repo.
Keep them wherever the team keeps artifacts and record the source in the model
README (`app/lpd/README.md`, `app/lpr/README.md`).
