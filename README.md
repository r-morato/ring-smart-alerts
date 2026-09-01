# ring-smart-alerts

Turn Ring's generic *"motion detected"* into *"a person and a dog at the front
door"*.

When your Ring camera fires a **motion** or **ding** event, this app:

1. fetches a snapshot from Ring — a fresh frame if the camera can produce one,
   otherwise Ring's last stored snapshot, otherwise nothing,
2. runs it through a **local** YOLOv8n object-detection model (CPU, no GPU, no
   cloud AI), then a **local CLIP zero-shot** pass that refines each person into
   `adult` / `child` / `courier` with a hedged `(looks like a man/woman)` guess,
   and — when YOLO sees nothing — guesses `package` / `animal` / `person` /
   `vehicle` for the whole frame, and
3. posts a descriptive notification to **Home Assistant** via its REST
   `notify` service, with the snapshot attached (or text-only if no image was
   available).

Everything runs on a Raspberry Pi / home server. The only outbound traffic is to
Ring's own API (for the snapshot and the event stream) and to your Home Assistant
instance.

---

## How it fits together

```
Ring event (FCM push)  ──►  ring_client.py  ──►  snapshot bytes
                                                     │
                                                     ▼
                                       detector.py: YOLOv8n boxes
                                                     │  + CLIP refine per person
                                                     ▼  + CLIP fallback if blank
                                            notifier.py  ──►  Home Assistant
                                                              notify.<target>
```

| File | Responsibility |
|------|----------------|
| `ring_smart_alerts/config.py` | Load & validate settings from env / `.env` |
| `ring_smart_alerts/ring_client.py` | Ring auth (+ first-run 2FA), token/FCM cache, snapshots, event listener |
| `ring_smart_alerts/detector.py` | YOLOv8n boxes + `ClipClassifier` refinement; `summarize()` → phrase |
| `ring_smart_alerts/notifier.py` | POST to the Home Assistant `notify` service |
| `ring_smart_alerts/main.py` | Event loop tying it together, snapshot cleanup, graceful shutdown |

---

## Requirements

- **Python 3.11+** (developed on 3.12 — on Windows use `py -3.12`, the bare
  `python` alias is the Store stub).
- A Ring account. **Battery / low-power cameras cannot take a fresh snapshot
  while they are recording a motion clip**, and without a Ring Protect
  subscription there is no snapshot-on-motion. `get_snapshot` handles this in
  three steps: poll ~16 s for a fresh frame, fall back to Ring's last stored
  snapshot (may be a few minutes stale), and if there is still nothing, send a
  text-only alert. Hardwired doorbells (Wired / Pro / Pro 2 / Elite) stay
  powered and normally return a current frame on the first step.
- A Home Assistant instance with the mobile app companion (or any other `notify`
  integration) and a long-lived access token.
- For the CLIP refinement stage: `open-clip-torch` (in `requirements.txt`) plus a
  one-time ~340 MB checkpoint download. Set `ENABLE_CLIP=false` to run plain YOLO
  and skip both.

---

## Setup

```bash
git clone https://github.com/<you>/ring-smart-alerts.git
cd ring-smart-alerts

py -3.12 -m venv .venv           # Windows
# python3 -m venv .venv          # Linux / Pi
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux / Pi

pip install -r requirements.txt
```

The first detection downloads `yolov8n.pt` (~6 MB) automatically, and the first
CLIP call downloads its checkpoint (~340 MB for the default
`ViT-B-32-quickgelu/openai`).
`torch` comes in as a dependency of `ultralytics`; on a Pi this is the CPU build
and is large (~100 MB) — be patient on the first install.

### Configure

```bash
cp .env.example .env
$EDITOR .env
```

| Variable | Required | Notes |
|----------|:---:|-------|
| `RING_EMAIL`, `RING_PASSWORD` | ✅ | Ring account login |
| `HA_URL` | ✅ | e.g. `http://homeassistant.local:8123` |
| `HA_TOKEN` | ✅ | Long-lived access token (see below) |
| `HA_NOTIFY_TARGET` | ✅ | The bit after `notify.` — e.g. `mobile_app_<device>`. Must match an existing service exactly; a wrong name gives an opaque HTTP 400 |
| `MIN_CONFIDENCE` | | Default `0.35` |
| `EVENT_KINDS` | | Default `motion,ding` |
| `NOTIFY_ON_EMPTY` | | Default `true` — still alert when nothing is recognised |
| `ENABLE_CLIP` | | Default `true` — CLIP person/scene refinement; `false` = plain YOLO |
| `CLIP_MODEL` / `CLIP_PRETRAINED` | | Default `ViT-B-32-quickgelu` / `openai`. On a Pi try `MobileCLIP-S1` / `datacompdr` |
| `TOKEN_CACHE_PATH` | | Default `~/.config/ring-smart-alerts/token.json` |
| `SNAPSHOT_DIR` | | Temp dir for in-flight snapshots; auto-cleaned |

#### Home Assistant long-lived access token

Home Assistant → click your **profile** (bottom-left) → **Security** tab →
**Long-lived access tokens** → **Create token**. Copy it into `HA_TOKEN`.

#### Finding your notify target

Developer Tools → **Actions** (formerly Services) → search `notify.` — the
companion app registers as `notify.mobile_app_<device_name>`, where
`<device_name>` is the phone's name in the HA app (slugified). Put just
`mobile_app_<device_name>` in `HA_NOTIFY_TARGET`. If the name doesn't match a
real service, Home Assistant rejects every call with a bare `400: Bad Request`.

### First run (Ring 2FA)

```bash
py -3.12 -m ring_smart_alerts.main
```

Ring will require a 2FA code (sent by email/SMS). Enter it when prompted **once** —
the auth token and the FCM listener credentials are cached to
`TOKEN_CACHE_PATH`, so subsequent starts are non-interactive. Keep that file
private (it is created mode `600` on Unix).

Once running you'll see `Listening for Ring events (ding, motion)`. Trigger the
doorbell and watch for a log line plus a notification on your phone.

---

## Run as a service (Raspberry Pi / systemd)

`/etc/systemd/system/ring-smart-alerts.service`:

```ini
[Unit]
Description=ring-smart-alerts
After=network-online.target
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/ring-smart-alerts
ExecStart=/home/pi/ring-smart-alerts/.venv/bin/python -m ring_smart_alerts.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now ring-smart-alerts
journalctl -u ring-smart-alerts -f
```

Do the interactive first run **once** by hand (for the 2FA prompt) before
enabling the service.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest -q          # test_detector downloads yolov8n.pt once; the rest are offline
ruff check .
```

---

## Notes & limitations

- **`person` refinement is a best-effort guess.** Doorbell frames are low-res,
  wide-angle, often backlit or night IR, with the subject side-on. `adult` vs
  `child` (body proportion) and `courier` (uniform + parcel) hold up reasonably;
  `man` vs `woman` is noisy and is only ever surfaced hedged as "looks like a
  …". Nothing sticks below its confidence threshold — you just get "a person".
- **`package` / non-COCO `animal`** are only guessed by the CLIP whole-frame
  fallback, which runs *when YOLO finds nothing*. A parcel next to a detected
  person won't be called out; a fox alone on the step should be.
- **Snapshots are transient.** Each snapshot is written to a temp dir only for
  the duration of one event and deleted immediately after the notification is
  sent; a sweep at startup clears anything left by a crash.
- **`ring-doorbell` is an unofficial library** and its API has changed across
  versions. This targets `>= 0.9.14`; check its
  [docs](https://python-ring-doorbell.readthedocs.io/) if you upgrade.

## License

MIT — see [LICENSE](LICENSE).
