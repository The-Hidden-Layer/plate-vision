# Model weights

The AI service loads two trained models from here:

```
lpd/best.pt   plate detector (YOLO)
lpr/best.pt   plate reader (LPRNet)
```

The weight files are git-ignored (too big for the repo) and shared separately —
ask the team for them and drop them in place. `apps/ai-service` is bind-mounted
at `/app` in compose, so they show up at `/app/weights/...` without rebuilding
the image; restart the service to load new ones:

```
docker compose restart ai-service
```

Without them the service fails to start with the models enabled. For a
model-free demo set `LPD_BACKEND=stub` and `LPR_BACKEND=stub` in `.env`.

Other checkpoints can sit alongside and be selected with `LPD_WEIGHTS` /
`LPR_WEIGHTS` in `.env`. Record where each file came from in the model READMEs
(`app/lpd/README.md`, `app/lpr/README.md`).
