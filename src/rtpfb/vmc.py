from __future__ import annotations

import time
from typing import Optional

from ._log import get_logger
from .errors import DependencyMissingError

log = get_logger("rtpfb.vmc")


# VMC humanoid bones (Unity / VRM standard, used by EVMC4U on the Unreal side
# to drive a MetaHuman skeleton).
VMC_BONES_FULL = (
    "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftEye", "RightEye", "Jaw",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
)

# MediaPipe Holistic pose landmark indices (33-pose).
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

# Identity quaternion (the renderer's IK / SpringBone solves the actual
# rotations from the bone positions — same trick VSeeFace uses).
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _mp_to_vmc_pos(mp_pos, scale_y: float = 1.8, hip_y: float = 0.9):
    """Convert a MediaPipe normalized landmark (x right, y down, z forward)
    to VMC world coordinates (x right, y up, z forward, metres)."""
    x = (float(mp_pos[0]) - 0.5) * 1.5
    y = (1.0 - float(mp_pos[1])) * scale_y - hip_y
    z = float(mp_pos[2])
    return x, y, z


class VMCSender:
    """VMC (Virtual Motion Capture) protocol output over OSC / UDP.

    What it sends per frame:
      * /VMC/Ext/Root/Pos        — root translation + identity rotation
      * /VMC/Ext/Bone/Pos        — per-bone position + identity rotation,
                                    one packet per bone we have data for
      * /VMC/Ext/Blend/Val       — optional ARKit-style face blendshape values
      * /VMC/Ext/Blend/Apply     — flush blendshapes
      * /VMC/Ext/T               — frame timestamp (seconds since start)
      * /VMC/Ext/OK              — end-of-frame marker

    What receives it (free, all zero-dollar stack):
      * Unreal Engine 5 + EVMC4U plugin → drives a MetaHuman skeletal mesh.
        Add Chaos Cloth on the chest cluster and the squish-on-lean falls
        out of the soft-body solver.
      * VSeeFace / VTube Studio / Unity VMC4U → for VRM avatars.

    First-cut policy: send positions only, identity rotations. The renderer's
    IK solver (Unreal Control Rig / VRM SpringBone IK) reconstructs joint
    angles from the bone positions. This is how every consumer-grade VMC
    pipeline works in practice.
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
        """Emit one VMC frame for ``pose`` (a ``rtpfb.pose.PoseFrame``)."""
        if pose is None or pose.pose_landmarks is None:
            return

        lms = pose.pose_landmarks
        if lms.shape[0] < 33:
            log.debug("pose has only %d landmarks, skipping VMC frame", lms.shape[0])
            return

        hip_mid = (lms[MP_LEFT_HIP] + lms[MP_RIGHT_HIP]) * 0.5
        root_pos = _mp_to_vmc_pos(hip_mid)

        self._client.send_message(
            "/VMC/Ext/Root/Pos",
            ["root", *root_pos, *_IDENTITY_QUAT],
        )

        for name, idx in self._bone_index_pairs():
            pos = _mp_to_vmc_pos(lms[idx])
            self._client.send_message(
                "/VMC/Ext/Bone/Pos",
                [name, *pos, *_IDENTITY_QUAT],
            )

        # Synthesise a Hips bone explicitly (between the two hip landmarks)
        # since MediaPipe doesn't have one.
        self._client.send_message(
            "/VMC/Ext/Bone/Pos",
            ["Hips", *root_pos, *_IDENTITY_QUAT],
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

    @staticmethod
    def _bone_index_pairs() -> list[tuple[str, int]]:
        return [
            ("Head", MP_NOSE),
            ("LeftShoulder", MP_LEFT_SHOULDER),
            ("RightShoulder", MP_RIGHT_SHOULDER),
            ("LeftUpperArm", MP_LEFT_SHOULDER),
            ("RightUpperArm", MP_RIGHT_SHOULDER),
            ("LeftLowerArm", MP_LEFT_ELBOW),
            ("RightLowerArm", MP_RIGHT_ELBOW),
            ("LeftHand", MP_LEFT_WRIST),
            ("RightHand", MP_RIGHT_WRIST),
            ("LeftUpperLeg", MP_LEFT_HIP),
            ("RightUpperLeg", MP_RIGHT_HIP),
            ("LeftLowerLeg", MP_LEFT_KNEE),
            ("RightLowerLeg", MP_RIGHT_KNEE),
            ("LeftFoot", MP_LEFT_ANKLE),
            ("RightFoot", MP_RIGHT_ANKLE),
            ("LeftToes", MP_LEFT_FOOT_INDEX),
            ("RightToes", MP_RIGHT_FOOT_INDEX),
        ]


def blendshapes_from_face_landmarks(face_landmarks) -> dict[str, float]:
    """Derive a small set of ARKit-style blendshapes from MediaPipe face
    landmarks.

    Lossy and approximate — a real implementation should use the MediaPipe
    Face Landmarker v2 task API, which outputs all 52 blendshapes natively.
    The handful we synthesise here is enough to get mouth + eye motion
    showing up on the avatar as a baseline.
    """
    if face_landmarks is None:
        return {}

    lm = face_landmarks  # (N, 3) normalised
    if lm.shape[0] < 468:
        return {}

    # Mouth open: vertical distance between upper-lip top (13) and lower-lip
    # bottom (14), normalised by face height.
    upper_lip = lm[13]
    lower_lip = lm[14]
    chin = lm[152]
    forehead = lm[10]

    mouth_open = abs(lower_lip[1] - upper_lip[1])
    face_height = max(1e-6, abs(chin[1] - forehead[1]))
    jaw_open = float(min(1.0, mouth_open / (face_height * 0.25)))

    # Eye blink: vertical distance between upper / lower eyelids on each
    # eye, normalised.
    left_eye_upper = lm[159]
    left_eye_lower = lm[145]
    right_eye_upper = lm[386]
    right_eye_lower = lm[374]

    left_open = abs(left_eye_lower[1] - left_eye_upper[1]) / (face_height * 0.06 + 1e-6)
    right_open = abs(right_eye_lower[1] - right_eye_upper[1]) / (face_height * 0.06 + 1e-6)
    eye_blink_left = float(max(0.0, 1.0 - min(1.0, left_open)))
    eye_blink_right = float(max(0.0, 1.0 - min(1.0, right_open)))

    return {
        "jawOpen": jaw_open,
        "eyeBlinkLeft": eye_blink_left,
        "eyeBlinkRight": eye_blink_right,
    }
