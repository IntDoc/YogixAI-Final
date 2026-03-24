# 🧘 Yogix AI API

Real-time yoga pose detection and correction powered by **Claude AI** (vision + coaching)
and **MediaPipe** (skeleton tracking). Built with Django Channels + WebSockets.

---

## Project Structure

```
yoga_final/
├── manage.py
├── requirements.txt
├── Procfile                  ← tells Railway how to start
├── nixpacks.toml             ← Railway build config
├── railway.toml              ← Railway deploy config
├── .env.example
├── yoga_api/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py               ← ASGI entry point (WebSocket support)
│   └── wsgi.py
├── api/
│   ├── pose_analysis.py      ← MediaPipe skeleton + angles
│   ├── claude_service.py     ← All Claude API calls
│   ├── consumers.py          ← WebSocket consumer
│   ├── views.py              ← REST endpoints
│   ├── urls.py
│   └── routing.py
└── templates/
    └── demo.html             ← Browser demo page
```

---

## API Endpoints

### WebSocket — Real-time pose analysis
```
ws://your-app.railway.app/ws/yoga/
```
Send binary JPEG frames. Receive JSON analysis.

**Send binary:** Raw JPEG bytes (one frame at a time)

**Send JSON (control messages):**
```json
{ "action": "reset" }
{ "action": "get_report" }
{ "action": "get_next_pose" }
{ "action": "ping" }
```

**Receive every frame:**
```json
{
  "type": "frame_result",
  "frame": 42,
  "detected": true,
  "full_body": true,
  "angles": { "left_knee": 172.1, "right_knee": 168.3, ... },
  "engagement": { "shoulder": 48, "core": 72, "legs": 30 },
  "similarity_score": 8.2,
  "avg_similarity_score": 7.9,
  "repetitions": 3,
  "hold_seconds": 12.4,
  "current_pose": "Warrior I",
  "landmarks": [{ "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.98 }, ...]
}
```

**Receive every ~15 frames (Claude classification):**
```json
{
  "type": "claude_result",
  "pose_name": "Warrior I",
  "confidence": 0.94,
  "category": "yoga",
  "corrections": ["Square your hips forward", "Extend your back arm fully"],
  "positive_feedback": "Great front knee alignment over your ankle!",
  "coaching_message": "Extend both arms to shoulder height."
}
```

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Health check + API info |
| GET  | `/demo/` | Interactive browser demo |
| POST | `/api/analyze-image/` | Deep analysis of uploaded image |
| POST | `/api/analyze-video-frame/` | Single frame analysis (REST alternative to WS) |
| POST | `/api/generate-report/` | Generate session report from JSON data |
| GET  | `/api/suggest-next-pose/?pose=Warrior+I` | Get next pose suggestion |

---

## Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 4. Run migrations
python manage.py migrate

# 5. Start server (must use Daphne for WebSocket support)
daphne -p 8000 yoga_api.asgi:application
```

Open http://localhost:8000/demo/ in your browser.

> **Note:** `python manage.py runserver` works for REST endpoints only.
> Use Daphne for WebSocket support.

---

## Deploy to Railway

### Step 1 — Push to GitHub
```bash
cd yoga_final
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Create Railway project
1. Go to [railway.app](https://railway.app) → **New Project**
2. Click **Deploy from GitHub repo**
3. Select your repository
4. Railway detects `nixpacks.toml` and starts building

### Step 3 — Add PostgreSQL (optional but recommended)
1. In your Railway project → **New** → **Database** → **PostgreSQL**
2. Railway auto-sets `DATABASE_URL` — no manual copy needed

### Step 4 — Set environment variables
In Railway dashboard → your service → **Variables** tab:

| Variable | Value |
|----------|-------|
| `ANTHROPIC_API_KEY` | `sk-ant-api03-...` |
| `SECRET_KEY` | Generate: `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"` |
| `DEBUG` | `False` |

> `PORT` and `DATABASE_URL` are set automatically by Railway — do not add them manually.

### Step 5 — Set Start Command (important!)
In Railway → your service → **Settings** → **Deploy** → **Start Command**:
```
daphne -b 0.0.0.0 -p $PORT yoga_api.asgi:application
```
This overrides any auto-detection and ensures Railway uses Daphne (required for WebSockets).

### Step 6 — Generate public URL
Railway → your service → **Settings** → **Networking** → **Generate Domain**

Your app will be live at: `https://your-app.up.railway.app`

Test:
- `https://your-app.up.railway.app/` → JSON health check
- `https://your-app.up.railway.app/demo/` → live demo
- `wss://your-app.up.railway.app/ws/yoga/` → WebSocket

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Yes | Your key from console.anthropic.com |
| `SECRET_KEY` | ✅ Yes | Django secret key — generate a fresh one |
| `DEBUG` | No | `False` for production (default) |
| `DATABASE_URL` | No | Auto-set by Railway PostgreSQL. Uses SQLite if not set. |
| `PORT` | No | Auto-set by Railway. Do not set manually. |

---

## Troubleshooting

**Build fails with "Failed to find WSGI_APPLICATION"**
→ Set the Start Command manually in Railway Settings (Step 5 above).

**MediaPipe / OpenCV build error**
→ The `nixpacks.toml` installs `libGL` and `libglib`. If it still fails,
add a Railway variable: `NIXPACKS_PKGS` = `libGL libglib gcc zlib`

**WebSocket connection refused**
→ Use `wss://` (not `ws://`) for HTTPS Railway URLs.

**Static files returning 404**
→ Make sure `whitenoise` is in requirements.txt (it is) and `collectstatic` runs in build (it does via nixpacks.toml).

**`SECRET_KEY` error on first deploy**
→ Add `SECRET_KEY` to Railway Variables before deploying.
