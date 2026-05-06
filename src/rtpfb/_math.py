"""Minimal quaternion / vector math for the VMC mocap path.

Quaternions are stored as ``(x, y, z, w)`` numpy arrays — same convention as
VMC's ``/VMC/Ext/Bone/Pos`` packet, scipy's ``Rotation.as_quat``, and Unity.
"""

from __future__ import annotations

import numpy as np

_X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_Y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_Z = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def quat_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def vec_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return quat_identity()
    return q / n


def quat_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion that rotates ``v1`` to ``v2``.

    Returns identity if either input is degenerate.
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return quat_identity()

    a = v1 / n1
    b = v2 / n2
    cos_theta = float(np.dot(a, b))
    cos_theta = max(-1.0, min(1.0, cos_theta))

    if cos_theta > 1.0 - 1e-7:
        return quat_identity()

    if cos_theta < -1.0 + 1e-7:
        # 180° rotation: pick any axis perpendicular to a
        axis = np.cross(a, _X)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, _Y)
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)

    axis = np.cross(a, b)
    s = float(np.sqrt((1.0 + cos_theta) * 2.0))
    inv_s = 1.0 / s
    return np.array(
        [axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, s * 0.5],
        dtype=np.float32,
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product. ``a`` then ``b`` in standard order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q``."""
    qx, qy, qz, qw = q
    u = np.array([qx, qy, qz], dtype=np.float32)
    s = qw
    return 2.0 * float(np.dot(u, v)) * u + (s * s - float(np.dot(u, u))) * v + 2.0 * s * np.cross(u, v)


def quat_from_basis(right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Build a quaternion from an orthonormal basis (column vectors).

    Right-handed: ``right × up = forward``.
    """
    m00, m01, m02 = right[0], up[0], forward[0]
    m10, m11, m12 = right[1], up[1], forward[1]
    m20, m21, m22 = right[2], up[2], forward[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0)) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = float(np.sqrt(1.0 + m00 - m11 - m22)) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = float(np.sqrt(1.0 + m11 - m00 - m22)) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = float(np.sqrt(1.0 + m22 - m00 - m11)) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return quat_normalize(np.array([qx, qy, qz, qw], dtype=np.float32))
