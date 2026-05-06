from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ._log import get_logger
from ._math import (
    quat_from_basis,
    quat_from_two_vectors,
    quat_identity,
    vec_normalize,
)
from .errors import DependencyMissingError

log = get_logger("rtpfb.vmc")


# MediaPipe Holistic pose landmark indices.
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_RIGHT_EYE_INNER = 4
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

# MediaPipe Hands landmark indices.
HAND_WRIST = 0
HAND_THUMB_CMC, HAND_THUMB_MCP, HAND_THUMB_IP, HAND_THUMB_TIP = 1, 2, 3, 4
HAND_INDEX_MCP, HAND_INDEX_PIP, HAND_INDEX_DIP, HAND_INDEX_TIP = 5, 6, 7, 8
HAND_MIDDLE_MCP, HAND_MIDDLE_PIP, HAND_MIDDLE_DIP, HAND_MIDDLE_TIP = 9, 10, 11, 12
HAND_RING_MCP, HAND_RING_PIP, HAND_RING_DIP, HAND_RING_TIP = 13, 14, 15, 16
HAND_LITTLE_MCP, HAND_LITTLE_PIP, HAND_LITTLE_DIP, HAND_LITTLE_TIP = 17, 18, 19, 20

# Per-finger triple of (proximal, intermediate, distal, tip) MediaPipe indices.
_FINGERS = (
    ("Thumb", HAND_THUMB_CMC, HAND_THUMB_MCP, HAND_THUMB_IP, HAND_THUMB_TIP),
    ("Index", HAND_INDEX_MCP, HAND_INDEX_PIP, HAND_INDEX_DIP, HAND_INDEX_TIP),
    ("Middle", HAND_MIDDLE_MCP, HAND_MIDDLE_PIP, HAND_MIDDLE_DIP, HAND_MIDDLE_TIP),
    ("Ring", HAND_RING_MCP, HAND_RING_PIP, HAND_RING_DIP, HAND_RING_TIP),
    ("Little", HAND_LITTLE_MCP, HAND_LITTLE_PIP, HAND_LITTLE_DIP, HAND_LITTLE_TIP),
)

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)

# Rest-pose direction vectors in VMC world coords (Y-up, T-pose).
_DIR_SPINE_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_DIR_LEFT = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
_DIR_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_DIR_DOWN = np.array([0.0, -1.0, 0.0], dtype=np.float32)
_DIR_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _mp_to_vmc(p) -> np.ndarray:
    """MediaPipe normalised landmark → VMC world coords (Y-up, metres-ish)."""
    return np.array(
        [
            (float(p[0]) - 0.5) * 1.5,
            (1.0 - float(p[1])) * 1.8 - 0.9,
            float(p[2]),
        ],
        dtype=np.float32,
    )


def _compute_body_transforms(pose) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute (head_pos, world_rotation) for every major body bone.

    Rotations are world-space; the receiver (EVMC4U / VMC4U) is configured
    to interpret them as such. Lets us compute one bone at a time without
    walking a parent chain.
    """
    lms = pose.pose_landmarks
    if lms is None or lms.shape[0] < 33:
        return {}

    midhip = (lms[MP_LEFT_HIP] + lms[MP_RIGHT_HIP]) * 0.5
    midshoulder = (lms[MP_LEFT_SHOULDER] + lms[MP_RIGHT_SHOULDER]) * 0.5

    midhip_w = _mp_to_vmc(midhip)
    midshoulder_w = _mp_to_vmc(midshoulder)
    nose_w = _mp_to_vmc(lms[MP_NOSE])
    left_ear_w = _mp_to_vmc(lms[MP_LEFT_EAR])
    right_ear_w = _mp_to_vmc(lms[MP_RIGHT_EAR])

    # Spine direction = up the torso. Used for hips/spine/chest/neck rotation.
    spine_dir = vec_normalize(midshoulder_w - midhip_w)
    spine_rot = quat_from_two_vectors(_DIR_SPINE_UP, spine_dir)

    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Hips": (midhip_w, spine_rot),
        "Spine": (midhip_w, spine_rot),
        "Chest": ((midhip_w + midshoulder_w) * 0.5, spine_rot),
        "UpperChest": (midshoulder_w, spine_rot),
        "Neck": (midshoulder_w, spine_rot),
    }

    # Head: build orthonormal basis from ears (right axis) + nose (forward).
    head_pos = (left_ear_w + right_ear_w) * 0.5
    right_axis = vec_normalize(right_ear_w - left_ear_w)
    forward_axis = vec_normalize(nose_w - head_pos)
    if float(np.linalg.norm(right_axis)) > 1e-6 and float(np.linalg.norm(forward_axis)) > 1e-6:
        up_axis = vec_normalize(np.cross(forward_axis, right_axis))
        # Re-orthogonalise forward to be perpendicular to up
        forward_axis = vec_normalize(np.cross(right_axis, up_axis))
        head_rot = quat_from_basis(right_axis, up_axis, forward_axis)
    else:
        head_rot = spine_rot
    transforms["Head"] = (head_pos, head_rot)

    # Arms — each segment, world rotation from rest direction → live direction.
    transforms.update(
        _arm_chain(
            "Left",
            shoulder=lms[MP_LEFT_SHOULDER],
            elbow=lms[MP_LEFT_ELBOW],
            wrist=lms[MP_LEFT_WRIST],
            chest_rot=spine_rot,
            rest=_DIR_LEFT,
        )
    )
    transforms.update(
        _arm_chain(
            "Right",
            shoulder=lms[MP_RIGHT_SHOULDER],
            elbow=lms[MP_RIGHT_ELBOW],
            wrist=lms[MP_RIGHT_WRIST],
            chest_rot=spine_rot,
            rest=_DIR_RIGHT,
        )
    )

    # Legs.
    transforms.update(
        _leg_chain(
            "Left",
            hip=lms[MP_LEFT_HIP],
            knee=lms[MP_LEFT_KNEE],
            ankle=lms[MP_LEFT_ANKLE],
            toes=lms[MP_LEFT_FOOT_INDEX],
        )
    )
    transforms.update(
        _leg_chain(
            "Right",
            hip=lms[MP_RIGHT_HIP],
            knee=lms[MP_RIGHT_KNEE],
            ankle=lms[MP_RIGHT_ANKLE],
            toes=lms[MP_RIGHT_FOOT_INDEX],
        )
    )

    return transforms


def _arm_chain(
    side: str,
    shoulder,
    elbow,
    wrist,
    chest_rot: np.ndarray,
    rest: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    shoulder_w = _mp_to_vmc(shoulder)
    elbow_w = _mp_to_vmc(elbow)
    wrist_w = _mp_to_vmc(wrist)

    upper = quat_from_two_vectors(rest, vec_normalize(elbow_w - shoulder_w))
    lower = quat_from_two_vectors(rest, vec_normalize(wrist_w - elbow_w))
    return {
        f"{side}Shoulder": (shoulder_w, chest_rot),
        f"{side}UpperArm": (shoulder_w, upper),
        f"{side}LowerArm": (elbow_w, lower),
        f"{side}Hand": (wrist_w, lower),
    }


def _leg_chain(
    side: str,
    hip,
    knee,
    ankle,
    toes,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    hip_w = _mp_to_vmc(hip)
    knee_w = _mp_to_vmc(knee)
    ankle_w = _mp_to_vmc(ankle)
    toes_w = _mp_to_vmc(toes)

    upper = quat_from_two_vectors(_DIR_DOWN, vec_normalize(knee_w - hip_w))
    lower = quat_from_two_vectors(_DIR_DOWN, vec_normalize(ankle_w - knee_w))
    foot = quat_from_two_vectors(_DIR_FORWARD, vec_normalize(toes_w - ankle_w))
    return {
        f"{side}UpperLeg": (hip_w, upper),
        f"{side}LowerLeg": (knee_w, lower),
        f"{side}Foot": (ankle_w, foot),
        f"{side}Toes": (toes_w, foot),
    }


def _compute_finger_transforms(
    side: str, hand_landmarks
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each finger, three phalanx bones (Proximal/Intermediate/Distal).

    Hand landmarks come from MediaPipe Holistic in image-normalised coords.
    Positions are passed straight through; rotations are derived from
    parent→child finger-bone direction. Rest direction depends on the
    finger but a single +Y rest works well enough across thumb / fingers
    once the receiver applies its bind-pose offset.
    """
    if hand_landmarks is None or hand_landmarks.shape[0] < 21:
        return {}

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for finger_name, prox, inter, dist, tip in _FINGERS:
        prox_w = _mp_to_vmc(hand_landmarks[prox])
        inter_w = _mp_to_vmc(hand_landmarks[inter])
        dist_w = _mp_to_vmc(hand_landmarks[dist])
        tip_w = _mp_to_vmc(hand_landmarks[tip])

        prox_rot = quat_from_two_vectors(_DIR_SPINE_UP, vec_normalize(inter_w - prox_w))
        inter_rot = quat_from_two_vectors(_DIR_SPINE_UP, vec_normalize(dist_w - inter_w))
        dist_rot = quat_from_two_vectors(_DIR_SPINE_UP, vec_normalize(tip_w - dist_w))

        out[f"{side}{finger_name}Proximal"] = (prox_w, prox_rot)
        out[f"{side}{finger_name}Intermediate"] = (inter_w, inter_rot)
        out[f"{side}{finger_name}Distal"] = (dist_w, dist_rot)
    return out


class VMCSender:
    """VMC (Virtual Motion Capture) protocol output over OSC / UDP.

    Per frame:
      * /VMC/Ext/Root/Pos        — root translation (hip midpoint) + spine rotation
      * /VMC/Ext/Bone/Pos        — per-bone position + rotation, ~25 body bones
                                    + up to 30 finger bones (5 fingers × 3 joints
                                    × 2 hands) when MediaPipe hand tracking is on
      * /VMC/Ext/Blend/Val + Apply — ARKit-standard blendshapes (52 of them when
                                     a Face Landmarker v2 task is loaded)
      * /VMC/Ext/T + /VMC/Ext/OK — frame timestamp + end-of-frame
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 39539):
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except ImportError as exc:
            raise DependencyMissingError(
                "python-osc is required for VMC. pip install python-osc "
                "(or install with the [mocap] extra)"
            ) from exc

        self.host = host
        self.port = port
        self._client = SimpleUDPClient(host, port)
        self._t0 = time.time()
        log.info("VMC sender → %s:%d", host, port)

    def send(self, pose, blendshapes: Optional[dict[str, float]] = None) -> None:
        if pose is None or pose.pose_landmarks is None:
            return

        body = _compute_body_transforms(pose)
        if not body:
            return

        # Root translation comes from Hips, rotation from spine direction.
        hips_pos, hips_rot = body["Hips"]
        self._client.send_message(
            "/VMC/Ext/Root/Pos",
            ["root", *hips_pos.tolist(), *hips_rot.tolist()],
        )

        for name, (pos, rot) in body.items():
            self._client.send_message(
                "/VMC/Ext/Bone/Pos",
                [name, *pos.tolist(), *rot.tolist()],
            )

        left_hand = getattr(pose, "left_hand_landmarks", None)
        right_hand = getattr(pose, "right_hand_landmarks", None)
        for side, hand in (("Left", left_hand), ("Right", right_hand)):
            for name, (pos, rot) in _compute_finger_transforms(side, hand).items():
                self._client.send_message(
                    "/VMC/Ext/Bone/Pos",
                    [name, *pos.tolist(), *rot.tolist()],
                )

        if blendshapes:
            for name, value in blendshapes.items():
                self._client.send_message(
                    "/VMC/Ext/Blend/Val",
                    [name, float(value)],
                )
            self._client.send_message("/VMC/Ext/Blend/Apply", [])

        self._client.send_message("/VMC/Ext/T", [time.time() - self._t0])
        self._client.send_message("/VMC/Ext/OK", [1])


def blendshapes_from_face_landmarks(face_landmarks) -> dict[str, float]:
    """Fallback: synthesise 3 blendshapes (jawOpen + eyeBlinkLeft/Right) from
    raw MediaPipe Holistic face landmarks.

    Used only when the Face Landmarker v2 model isn't loaded — the v2 task
    outputs all 52 ARKit blendshapes natively, with much higher fidelity.
    Prefer ``rtpfb.pose.FaceBlendshapeExtractor``.
    """
    if face_landmarks is None:
        return {}
    if face_landmarks.shape[0] < 468:
        return {}

    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]
    chin = face_landmarks[152]
    forehead = face_landmarks[10]

    mouth_open = abs(lower_lip[1] - upper_lip[1])
    face_height = max(1e-6, abs(chin[1] - forehead[1]))
    jaw_open = float(min(1.0, mouth_open / (face_height * 0.25)))

    left_eye_upper = face_landmarks[159]
    left_eye_lower = face_landmarks[145]
    right_eye_upper = face_landmarks[386]
    right_eye_lower = face_landmarks[374]

    left_open = abs(left_eye_lower[1] - left_eye_upper[1]) / (face_height * 0.06 + 1e-6)
    right_open = abs(right_eye_lower[1] - right_eye_upper[1]) / (face_height * 0.06 + 1e-6)
    eye_blink_left = float(max(0.0, 1.0 - min(1.0, left_open)))
    eye_blink_right = float(max(0.0, 1.0 - min(1.0, right_open)))

    return {
        "jawOpen": jaw_open,
        "eyeBlinkLeft": eye_blink_left,
        "eyeBlinkRight": eye_blink_right,
    }
