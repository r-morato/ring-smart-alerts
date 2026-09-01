# ring-smart-alerts

On a Ring motion or doorbell event, Ring only tells you "motion detected". This
service fetches the camera snapshot, runs object detection on it locally, and
sends a Home Assistant notification describing what is in frame, for example "a
person and a dog at the front door".

All image processing happens on the machine you run this on. It uses two models,
YOLOv8n for object detection and CLIP for a follow-up classification pass, both
on CPU. There is no LLM and no third-party vision API. The snapshot comes from
Ring, is analysed in memory, and is sent on to your own Home Assistant so the
notification can show it. It is not written to local disk and is not sent
anywhere else.

## What it does

On a Ring `motion` or `ding` event:

1. Fetch a snapshot from Ring. If the camera can produce a fresh frame it uses
   that; otherwise it falls back to Ring's last stored snapshot; if there is
   nothing, it sends a text-only alert.
2. Run YOLOv8n object detection on CPU. Each detected person is passed through a
   CLIP zero-shot pass that labels them `adult`, `child` or `courier`, with a
   hedged `(looks like a man/woman)` guess. If YOLO finds nothing, CLIP makes a
   whole-frame guess of `package`, `animal`, `person` or `vehicle`.
3. Post a notification to Home Assistant through its REST `notify` service, with
   the snapshot attached. The image is uploaded to Home Assistant's own image
   store so the companion app can display it, and older uploads are pruned.

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
| `ring_smart_alerts/notifier.py` | Upload the snapshot to HA's image store, POST the `notify` service, prune old images |
| `ring_smart_alerts/main.py` | Event loop tying it together, graceful shutdown |

### What crosses the network

| Direction | Endpoint | Carries |
|-----------|----------|---------|
| in ← | Ring API + Firebase Cloud Messaging | the event push stream and the snapshot bytes |
| out → | your Home Assistant (`HA_URL`) | the alert text, and the snapshot (uploaded to HA's own image store, `POST /api/image/upload`) |
| out → | your Home Assistant (websocket API) | delete calls for pruned snapshots |
| out → (first run only) | Ultralytics CDN, Hugging Face Hub | model weights, cached forever after |

The snapshot travels from Ring to your machine, is held in memory for analysis
and not written to disk, and is then sent to your Home Assistant so the
notification can display it. Home Assistant keeps the last `KEEP_IMAGES` (5)
snapshots and this app deletes the older ones over HA's websocket API.

---

## Requirements

- Python 3.11+ (developed on 3.12; on Windows use `py -3.12`, the bare
  `python` alias is the Store stub).
- A Ring account. Battery and low-power cameras cannot take a fresh snapshot
  while they are recording a motion clip, and without a Ring Protect
  subscription there is no snapshot-on-motion. `get_snapshot` handles this in
  three steps: poll ~16 s for a fresh frame, fall back to Ring's last stored
  snapshot (may be a few minutes stale), and if there is still nothing, send a
  text-only alert. Hardwired doorbells (Wired, Pro, Pro 2, Elite) stay
  powered and normally return a current frame on the first step.
- A Home Assistant instance with the mobile app companion (or any other `notify`
  integration) and a long-lived access token.
- For the CLIP refinement stage: `open-clip-torch` (in `requirements.txt`) plus a
  one-time ~340 MB checkpoint download. Set `ENABLE_CLIP=false` to run plain YOLO
  and skip both.

---

## Setup

```bash
git clone https://github.com/r-morato/ring-smart-alerts.git
cd ring-smart-alerts

py -3.12 -m venv .venv           # Windows
# python3 -m venv .venv          # Linux / Pi
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux / Pi

pip install -r requirements.txt
```

(If you forked first, clone your own fork's URL instead.)

The first detection downloads `yolov8n.pt` (~6 MB) automatically, and the first
CLIP call downloads its checkpoint (~340 MB for the default
`ViT-B-32-quickgelu/openai`).
`torch` comes in as a dependency of `ultralytics`; on a Pi this is the CPU build
and is large (~100 MB), so the first install takes a while.

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
| `HA_NOTIFY_TARGET` | ✅ | The part after `notify.`, e.g. `mobile_app_<device>`. Must match an existing service exactly; a wrong name gives an opaque HTTP 400 |
| `MIN_CONFIDENCE` | | Default `0.35` |
| `EVENT_KINDS` | | Default `motion,ding` |
| `NOTIFY_ON_EMPTY` | | Default `true`; still alert when nothing is recognised |
| `ENABLE_CLIP` | | Default `true`; CLIP person/scene refinement. `false` runs plain YOLO |
| `CLIP_MODEL` / `CLIP_PRETRAINED` | | Default `ViT-B-32-quickgelu` / `openai`. On a Pi try `MobileCLIP-S1` / `datacompdr` |
| `TOKEN_CACHE_PATH` | | Default `~/.config/ring-smart-alerts/token.json` |

#### Home Assistant long-lived access token

Home Assistant → click your **profile** (bottom-left) → **Security** tab →
**Long-lived access tokens** → **Create token**. Copy it into `HA_TOKEN`.

#### Finding your notify target

Developer Tools → **Actions** (formerly Services) → search `notify.`. The
companion app registers as `notify.mobile_app_<device_name>`, where
`<device_name>` is the phone's name in the HA app (slugified). Put just
`mobile_app_<device_name>` in `HA_NOTIFY_TARGET`. If the name doesn't match a
real service, Home Assistant rejects every call with a bare `400: Bad Request`.

### First run (Ring 2FA)

```bash
py -3.12 -m ring_smart_alerts.main
```

Ring will require a 2FA code (sent by email/SMS). Enter it when prompted once.
The auth token and the FCM listener credentials are then cached to
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
  ...". Nothing sticks below its confidence threshold, in which case you just
  get "a person".
- **`package` / non-COCO `animal`** are only guessed by the CLIP whole-frame
  fallback, which runs *when YOLO finds nothing*. A parcel next to a detected
  person won't be called out; a fox alone on the step should be.
- **Snapshots are not written to local disk.** A frame is held in memory for the
  length of one event. The copy that persists is the one in Home Assistant's
  image store, and only the last `KEEP_IMAGES` (5) are kept; older ones are
  deleted over HA's websocket API. If a delete fails it is logged and that one
  image is left in HA. This is harmless, since a snapshot is tens of KB; clear
  it from the image store by hand if you like.
- **The image-store upload needs the `image_upload` integration**, which is part
  of Home Assistant's `default_config` and on almost every install. The notify
  target must be a companion-app device (`mobile_app_*`) for the picture to
  render; other `notify` integrations get the text.
- **`ring-doorbell` is an unofficial library** and its API has changed across
  versions. This targets `>= 0.9.14`; check its
  [docs](https://python-ring-doorbell.readthedocs.io/) if you upgrade.

## License

MIT. See [LICENSE](LICENSE).
