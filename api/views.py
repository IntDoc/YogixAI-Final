"""
views.py — REST API endpoints
"""

import json
import time
import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render

from .pose_analysis import process_frame
from .claude_service import analyze_image, classify_pose, generate_session_report, suggest_next_pose


def index(request):
    """Health check + API documentation."""
    return JsonResponse({
        "name": "Yogix AI API",
        "version": "2.0",
        "status": "healthy",
        "powered_by": "Claude AI + MediaPipe",
        "endpoints": {
            "websocket":      "ws://<host>/ws/yoga/",
            "demo":           "GET  /demo/",
            "analyze_image":  "POST /api/analyze-image/",
            "analyze_frame":  "POST /api/analyze-video-frame/",
            "generate_report":"POST /api/generate-report/",
            "suggest_pose":   "GET  /api/suggest-next-pose/?pose=Warrior+I",
        },
    })


def demo(request):
    return render(request, "demo.html")


@csrf_exempt
@require_http_methods(["POST"])
def analyze_image_view(request):
    """
    POST /api/analyze-image/
    Multipart form: field 'image' (JPEG/PNG), optional 'context' (string)
    Returns deep Claude pose analysis + MediaPipe angles.
    """
    if "image" not in request.FILES:
        return JsonResponse({"error": "No 'image' file provided."}, status=400)

    image_file = request.FILES["image"]
    image_bytes = image_file.read()
    mime_type = image_file.content_type or "image/jpeg"
    context = request.POST.get("context", "")

    # MediaPipe
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    mp_result = {}
    if frame is not None:
        mp_result = process_frame(frame)
        mp_result.pop("_raw", None)

    # Claude deep analysis
    claude_result = analyze_image(image_bytes, mime_type, context)

    return JsonResponse({
        "mediapipe": mp_result,
        "claude_analysis": claude_result,
        "timestamp": int(time.time()),
    })


@csrf_exempt
@require_http_methods(["POST"])
def analyze_frame_view(request):
    """
    POST /api/analyze-video-frame/
    Body: raw JPEG bytes (Content-Type: image/jpeg)
       OR multipart with field 'frame'
    Lightweight alternative to WebSocket for clients that can't use WS.
    """
    if request.content_type and "multipart" in request.content_type:
        if "frame" not in request.FILES:
            return JsonResponse({"error": "No 'frame' file provided."}, status=400)
        image_bytes = request.FILES["frame"].read()
    else:
        image_bytes = request.body
        if not image_bytes:
            return JsonResponse({"error": "Empty request body."}, status=400)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JsonResponse({"error": "Cannot decode image."}, status=400)

    frame = cv2.resize(frame, (640, 360))
    result = process_frame(frame)
    result.pop("_raw", None)

    if not result["detected"] or not result["full_body"]:
        return JsonResponse({
            "detected": result["detected"],
            "full_body": result.get("full_body", False),
            "message": "Full body not detected.",
            "angles": None,
            "claude_classification": None,
        })

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    claude_result = classify_pose(buf.tobytes(), result["angles"])

    return JsonResponse({
        "detected": True,
        "full_body": True,
        "angles": result["angles"],
        "engagement": result["engagement"],
        "landmarks": result["landmarks"],
        "claude_classification": claude_result,
        "timestamp": int(time.time()),
    })


@csrf_exempt
@require_http_methods(["POST"])
def generate_report_view(request):
    """
    POST /api/generate-report/
    JSON body with session data. Returns Markdown report from Claude.
    """
    try:
        session_data = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    if "duration_seconds" not in session_data:
        return JsonResponse({"error": "Missing required field: duration_seconds"}, status=400)

    report_md = generate_session_report(session_data)
    return JsonResponse({
        "report_markdown": report_md,
        "session_data": session_data,
        "generated_at": int(time.time()),
    })


@require_http_methods(["GET"])
def suggest_pose_view(request):
    """
    GET /api/suggest-next-pose/?pose=Warrior+I&history=Mountain+Pose,Warrior+I&category=yoga
    """
    current = request.GET.get("pose", "Mountain Pose")
    history_raw = request.GET.get("history", "")
    category = request.GET.get("category", "yoga")
    history = [p.strip() for p in history_raw.split(",") if p.strip()]
    result = suggest_next_pose(current, history, category)
    return JsonResponse(result)
