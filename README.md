# ring-smart-alerts

Turn Ring's generic *"motion detected"* into *"a person and a dog at the front
door"*.

When your Ring camera fires a **motion** or **ding** event, this app:

1. fetches the current snapshot from Ring,
2. runs it through a **local** YOLOv8n object-detection model (CPU, no GPU, no
   cloud AI), and
3. posts a descriptive notification to **Home Assistant** via its REST
   `notify` service, with the snapshot attached.

Everything runs on a Raspberry Pi / home server. The only outbound traffic is to
Ring's own API (for the snapshot and the event stream) and to your Home Assistant
instance.

---

## How it fits together

```
Ring event (FCM push)  ──►  ring_client.py  ──►  snapshot bytes
                                                     │
                                                     ▼
                                            detector.py (YOLOv8n)
                                                     │  labels + confidence
                                                     ▼
                                            notifier.py  ──►  Home Assistant
                                                              notify.<target>
```

| File | Responsibility |
|------|----------------|
| `ring_smart_alerts/config.py` | Load & validate settings from env / `.env` |
| `ring_smart_alerts/ring_client.py` | Ring auth (+ first-run 2FA), token/FCM cache, snapshots, event listener |
| `ring_smart_alerts/detector.py` | Run YOLOv8n on an image, return de-duped labels; `summarize()` → phrase |
| `ring_smart_alerts/notifier.py` | POST to the Home Assistant `notify` service |
| `ring_smart_alerts/main.py` | Event loop tying it together, snapshot cleanup, graceful shutdown |

---

## Requirements

- **Python 3.11+** (developed on 3.12 — on Windows use `py -3.12`, the bare
  `python` alias is the Store stub).
- A Ring account. **Battery / low-power cameras cannot take a snapshot while
  they are recording a motion clip.** Without a Ring Protect subscription the
  snapshot returned during an event may be a few minutes stale, or the fetch may
  fail (handled gracefully — you still get a text-only alert).
- A Home Assistant instance with the mobile app companion (or any other `notify`
  integration) and a long-lived access token.

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

The first detection downloads `yolov8n.pt` (~6 MB) automatically. `torch` comes
in as a dependency of `ultralytics`; on a Pi this is the CPU build and is large
(~100 MB) — be patient on the first install.

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
| `HA_NOTIFY_TARGET` | ✅ | The bit after `notify.` — e.g. `mobile_app_pixel` |
| `MIN_CONFIDENCE` | | Default `0.35` |
| `EVENT_KINDS` | | Default `motion,ding` |
| `NOTIFY_ON_EMPTY` | | Default `true` — still alert when nothing is recognised |
| `TOKEN_CACHE_PATH` | | Default `~/.config/ring-smart-alerts/token.json` |
| `SNAPSHOT_DIR` | | Temp dir for in-flight snapshots; auto-cleaned |

#### Home Assistant long-lived access token

Home Assistant → click your **profile** (bottom-left) → **Security** tab →
**Long-lived access tokens** → **Create token**. Copy it into `HA_TOKEN`.

#### Finding your notify target

Developer Tools → **Actions** (formerly Services) → search `notify.` — the
companion app registers as `notify.mobile_app_<device_name>`. Put just
`mobile_app_<device_name>` in `HA_NOTIFY_TARGET`.

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
pytest -q          # test_detector downloads yolov8n.pt once; other tests are offline
ruff check .
```

---

## Notes & limitations

- **No "package" class.** YOLOv8n uses the 80 COCO classes, which don't include
  parcels/boxes. `detector.py._package_fallback()` is a documented stub for a
  future CLIP zero-shot check ("a cardboard box on a doorstep").
- **Snapshots are transient.** Each snapshot is written to a temp dir only for
  the duration of one event and deleted immediately after the notification is
  sent; a sweep at startup clears anything left by a crash.
- **`ring-doorbell` is an unofficial library** and its API has changed across
  versions. This targets `>= 0.9.14`; check its
  [docs](https://python-ring-doorbell.readthedocs.io/) if you upgrade.

## License

MIT — see [LICENSE](LICENSE).
