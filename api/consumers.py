"""
consumers.py
WebSocket consumer for real-time yoga pose analysis.

Protocol:
  Client → Server : binary JPEG bytes (one frame at a time)
  Client → Server : JSON text for control messages
  Server → Client : JSON text with analysis results

Claude is called every CLAUDE_EVERY_N_FRAMES to keep costs low.
MediaPipe runs on every frame for smooth landmark data.
"""

import asyncio
import json
import time
import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from .pose_analysis import process_frame, compute_similarity
from .claude_service import classify_pose, get_coaching_message, suggest_next_pose

CLAUDE_EVERY_N_FRAMES = 15   # Call Claude ~every 0.5s at 30fps client rate


class YogaConsumer(AsyncWebsocketConsumer):

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self):
        await self.accept()
        self._init_session()
        await self._send_json({
            "type": "connected",
            "message": "Yoga AI connected. Send JPEG frames as binary data.",
        })

    async def disconnect(self, close_code):
        if self._claude_task and not self._claude_task.done():
            self._claude_task.cancel()

    # ── Session state ──────────────────────────────────────────────────────────

    def _init_session(self):
        self.frame_count = 0
        self.session_start = time.time()
        self.current_pose = None
        self.pose_start_time = None
        self.reference_lm = None
        self.similarity_scores = []
        self.engagement_history = {"shoulder": [], "core": [], "legs": []}
        self.pose_data = []
        self.pose_history = []
        self.repetition_count = 0
        self._last_pose_for_reps = None
        self._hold_frames = 0
        self.corrections_summary = {}
        self._claude_task = None

    # ── Message dispatch ───────────────────────────────────────────────────────

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            await self._handle_control(text_data)
        elif bytes_data:
            await self._handle_frame(bytes_data)

    async def _handle_control(self, text_data):
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            return

        action = data.get("action", "")

        if action == "reset":
            self._init_session()
            await self._send_json({"type": "reset", "message": "Session reset."})

        elif action == "get_report":
            payload = self._build_report_payload()
            await self._send_json({"type": "report", **payload})

        elif action == "get_next_pose":
            pose = self.current_pose or "Mountain Pose"
            result = await sync_to_async(suggest_next_pose)(pose, self.pose_history)
            await self._send_json({"type": "next_pose", **result})

        elif action == "ping":
            await self._send_json({"type": "pong"})

    # ── Frame processing ───────────────────────────────────────────────────────

    async def _handle_frame(self, frame_bytes: bytes):
        self.frame_count += 1

        # Decode JPEG
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._send_json({"type": "error", "message": "Cannot decode image. Send JPEG bytes."})
            return

        frame = cv2.resize(frame, (640, 360))

        # MediaPipe (fast, runs every frame)
        result = await sync_to_async(process_frame)(frame)

        if not result["detected"]:
            await self._send_json({
                "type": "frame_result",
                "frame": self.frame_count,
                "detected": False,
                "message": "No person detected. Ensure full body is in frame.",
            })
            return

        if not result["full_body"]:
            await self._send_json({
                "type": "frame_result",
                "frame": self.frame_count,
                "detected": True,
                "full_body": False,
                "message": "Full body not visible. Step back from camera.",
            })
            return

        # Similarity score
        raw_lm = result["_raw"]
        if self.reference_lm is None:
            self.reference_lm = raw_lm
        similarity = compute_similarity(raw_lm, self.reference_lm)
        self.similarity_scores.append(similarity)
        if len(self.similarity_scores) > 30:
            self.similarity_scores.pop(0)
        avg_sim = round(sum(self.similarity_scores) / len(self.similarity_scores), 2)

        # Engagement history
        eng = result["engagement"]
        for key in ("shoulder", "core", "legs"):
            self.engagement_history[key].append(eng[key])
            if len(self.engagement_history[key]) > 60:
                self.engagement_history[key].pop(0)

        # Rep counting
        self._update_reps()

        # Fire Claude classification every N frames (non-blocking)
        if self.frame_count % CLAUDE_EVERY_N_FRAMES == 1:
            if self._claude_task is None or self._claude_task.done():
                self._claude_task = asyncio.create_task(
                    self._run_claude(frame, result["angles"])
                )

        hold_s = round(time.time() - self.pose_start_time, 1) if self.pose_start_time else 0.0

        await self._send_json({
            "type": "frame_result",
            "frame": self.frame_count,
            "detected": True,
            "full_body": True,
            "angles": result["angles"],
            "engagement": eng,
            "similarity_score": similarity,
            "avg_similarity_score": avg_sim,
            "repetitions": self.repetition_count,
            "hold_seconds": hold_s,
            "current_pose": self.current_pose,
            "landmarks": result["landmarks"],
        })

    async def _run_claude(self, frame, angles):
        """Background Claude classification task."""
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_bytes = buf.tobytes()

            result = await sync_to_async(classify_pose)(image_bytes, angles)

            pose_name = result.get("pose_name", "Unknown Pose")
            confidence = result.get("confidence", 0.0)
            corrections = result.get("corrections", [])

            if confidence >= 0.6:
                if self.current_pose != pose_name:
                    # Log completed pose before switching
                    if self.current_pose and self.pose_start_time:
                        self.pose_data.append({
                            "name": self.current_pose,
                            "duration": round(time.time() - self.pose_start_time, 1),
                            "avg_score": round(
                                sum(self.similarity_scores) / max(len(self.similarity_scores), 1), 2
                            ),
                        })
                        self.pose_history.append(self.current_pose)

                    self.current_pose = pose_name
                    self.pose_start_time = time.time()
                    self.reference_lm = None  # reset reference for new pose

                for c in corrections:
                    self.corrections_summary[c] = self.corrections_summary.get(c, 0) + 1

            # Quick coaching message
            hold_s = time.time() - self.pose_start_time if self.pose_start_time else 0
            avg_sim = sum(self.similarity_scores) / max(len(self.similarity_scores), 1)
            coaching = await sync_to_async(get_coaching_message)(
                pose_name, corrections, hold_s, avg_sim
            )

            await self._send_json({
                "type": "claude_result",
                "pose_name": pose_name,
                "confidence": confidence,
                "category": result.get("category", "unknown"),
                "corrections": corrections,
                "positive_feedback": result.get("positive_feedback", ""),
                "coaching_message": coaching,
            })

        except Exception as e:
            await self._send_json({"type": "claude_error", "message": str(e)})

    def _update_reps(self):
        if self.current_pose and self.current_pose == self._last_pose_for_reps:
            self._hold_frames += 1
            if self._hold_frames >= 30:
                self.repetition_count += 1
                self._hold_frames = 0
        else:
            self._last_pose_for_reps = self.current_pose
            self._hold_frames = 0

    def _build_report_payload(self) -> dict:
        duration = round(time.time() - self.session_start)
        avg_eng = {
            k: round(sum(v) / max(len(v), 1), 1)
            for k, v in self.engagement_history.items()
        }
        avg_sim = round(
            sum(self.similarity_scores) / max(len(self.similarity_scores), 1), 2
        )
        return {
            "duration_seconds": duration,
            "total_frames": self.frame_count,
            "poses": self.pose_data,
            "avg_similarity_score": avg_sim,
            "avg_engagement": avg_eng,
            "repetitions": self.repetition_count,
            "corrections_summary": self.corrections_summary,
            "pose_history": self.pose_history,
        }

    async def _send_json(self, data: dict):
        await self.send(text_data=json.dumps(data))
