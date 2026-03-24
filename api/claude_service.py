"""
claude_service.py
All Claude API interactions — pose classification, coaching, next pose, reports.
"""

import base64
import json
import anthropic
from django.conf import settings

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


# ── Pose classification (Vision) ──────────────────────────────────────────────

CLASSIFY_SYSTEM = """You are an expert yoga and physiotherapy pose classifier with vision capability.
You will receive a JPEG image of a person and optionally their joint angles from MediaPipe.

Respond ONLY with this exact JSON structure, no markdown, no extra text:
{
  "pose_name": "Warrior I",
  "confidence": 0.92,
  "category": "yoga",
  "corrections": ["Extend left arm fully overhead", "Square hips forward"],
  "positive_feedback": "Great back leg extension and strong stance!"
}

Rules:
- pose_name: standard English name (e.g. "Warrior I", "Tree Pose", "Downward Dog", "Mountain Pose", "Chair Pose", "Plank", "Child's Pose", "Squat", "Lunge")
- confidence: 0.0 to 1.0
- category: "yoga" or "physiotherapy" or "strength" or "unknown"
- corrections: 0 to 3 short, actionable corrections. Empty list [] if pose looks good.
- positive_feedback: one encouraging sentence about what they are doing well
- If full body is not visible or pose is unrecognisable, set confidence below 0.5 and pose_name to "Unknown Pose"
"""


def classify_pose(image_bytes: bytes, angles: dict | None) -> dict:
    """Call Claude Vision to classify a pose from a JPEG image."""
    client = get_client()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    angle_text = ""
    if angles:
        lines = [f"  {k}: {v}°" for k, v in angles.items()]
        angle_text = "\n\nJoint angles from MediaPipe:\n" + "\n".join(lines)

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=400,
            system=CLASSIFY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Classify the yoga or exercise pose in this image.{angle_text}",
                        },
                    ],
                }
            ],
        )
        raw = response.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "pose_name": "Unknown Pose",
            "confidence": 0.0,
            "category": "unknown",
            "corrections": [],
            "positive_feedback": f"Pose detection error: {str(e)[:80]}",
        }


# ── Coaching message (Haiku — fast + cheap) ───────────────────────────────────

COACHING_SYSTEM = """You are a motivating yoga coach giving real-time verbal feedback.
Keep it to ONE short sentence, maximum 12 words. Be specific and positive.
Return ONLY the sentence, no quotes, no punctuation at the start."""


def get_coaching_message(pose_name: str, corrections: list, hold_seconds: float, score: float) -> str:
    client = get_client()
    corrections_text = ", ".join(corrections) if corrections else "none"
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            system=COACHING_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Pose: {pose_name}\n"
                    f"Hold time: {hold_seconds:.0f}s\n"
                    f"Accuracy: {score:.1f}/10\n"
                    f"Issues: {corrections_text}\n"
                    f"Give one short coaching cue."
                ),
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return "Keep holding — you're doing great!"


# ── Next pose suggestion (Haiku) ──────────────────────────────────────────────

NEXT_POSE_SYSTEM = """You are a yoga sequence planner.
Given the current pose and recent history, suggest the single best next pose.
Respond ONLY with this JSON, no markdown:
{
  "next_pose": "Warrior II",
  "reason": "Natural hip-opening progression from Warrior I",
  "difficulty": "beginner"
}
difficulty must be: "beginner", "intermediate", or "advanced"."""


def suggest_next_pose(current_pose: str, history: list, category: str = "yoga") -> dict:
    client = get_client()
    history_text = " → ".join(history[-5:]) if history else "none"
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=NEXT_POSE_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Current pose: {current_pose}\n"
                    f"Category: {category}\n"
                    f"Recent sequence: {history_text}\n"
                    f"Suggest the next pose."
                ),
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {
            "next_pose": "Child's Pose",
            "reason": "Rest and recovery",
            "difficulty": "beginner",
        }


# ── Session report (Opus) ─────────────────────────────────────────────────────

REPORT_SYSTEM = """You are a professional yoga and physiotherapy session analyst.
Write a detailed, encouraging session report in Markdown.
Include sections: Session Summary, Poses Performed, Performance Analysis,
Areas for Improvement, and Recommendations for Next Session.
Use headings, bullet points, and be specific but motivating."""


def generate_session_report(session_data: dict) -> str:
    client = get_client()

    poses_text = "\n".join(
        f"- {p.get('name', 'Unknown')}: {p.get('duration', 0):.0f}s, score {p.get('avg_score', 0):.1f}/10"
        for p in session_data.get("poses", [])
    ) or "No poses recorded"

    corrections_text = "\n".join(
        f"- {c}: {n}x"
        for c, n in sorted(
            session_data.get("corrections_summary", {}).items(),
            key=lambda x: -x[1]
        )[:5]
    ) or "No corrections recorded"

    duration = session_data.get("duration_seconds", 0)
    eng = session_data.get("avg_engagement", {})

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1500,
            system=REPORT_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Generate a yoga session report:\n\n"
                    f"User: {session_data.get('user_name', 'Practitioner')}\n"
                    f"Duration: {duration // 60}m {duration % 60}s\n"
                    f"Total repetitions: {session_data.get('repetitions', 0)}\n"
                    f"Average accuracy score: {session_data.get('avg_similarity_score', 0):.1f}/10\n"
                    f"Engagement — Shoulders: {eng.get('shoulder', 'N/A')}%, "
                    f"Core: {eng.get('core', 'N/A')}%, Legs: {eng.get('legs', 'N/A')}%\n\n"
                    f"Poses performed:\n{poses_text}\n\n"
                    f"Most common corrections:\n{corrections_text}"
                ),
            }],
        )
        return response.content[0].text
    except Exception as e:
        return f"# Session Report\n\nError generating report: {e}"


# ── Single image deep analysis ────────────────────────────────────────────────

ANALYZE_SYSTEM = """You are an expert yoga and physiotherapy coach analyzing a static image.
Respond ONLY with this JSON, no markdown:
{
  "pose_name": "Downward Dog",
  "confidence": 0.95,
  "category": "yoga",
  "form_score": 7.5,
  "alignment_notes": ["Good arm extension", "Hips could be higher"],
  "safety_flags": [],
  "corrections": ["Push heels toward floor", "Lengthen through the spine"],
  "benefits": ["Stretches hamstrings", "Strengthens shoulders"],
  "modifications": ["Bend knees slightly if hamstrings are tight"],
  "positive_feedback": "Excellent arm alignment and shoulder engagement!"
}
form_score: 1.0 to 10.0
safety_flags: list of safety concerns, empty if none."""


def analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg", context: str = "") -> dict:
    client = get_client()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    user_text = "Perform a detailed yoga/physiotherapy pose analysis."
    if context:
        user_text += f" Context: {context}"
    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=800,
            system=ANALYZE_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime_type, "data": b64},
                    },
                    {"type": "text", "text": user_text},
                ],
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {"error": str(e)}
