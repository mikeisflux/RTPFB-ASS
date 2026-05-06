from __future__ import annotations

import numpy as np
import pytest

from rtpfb._math import (
    quat_from_basis,
    quat_from_two_vectors,
    quat_identity,
    quat_inverse,
    quat_mul,
    quat_normalize,
    quat_rotate,
    vec_normalize,
)


def test_quat_identity_shape():
    q = quat_identity()
    assert q.shape == (4,)
    np.testing.assert_array_almost_equal(q, [0.0, 0.0, 0.0, 1.0])


def test_vec_normalize_unit():
    v = vec_normalize(np.array([3.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(v, [1.0, 0.0, 0.0])


def test_vec_normalize_zero():
    v = vec_normalize(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(v, [0.0, 0.0, 0.0])


def test_quat_normalize_identity_on_zero():
    q = quat_normalize(np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(q, quat_identity())


def test_from_two_vectors_same_returns_identity():
    q = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(q, quat_identity())


def test_from_two_vectors_opposite_returns_180():
    q = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    # 180-degree rotation: w should be 0, axis is some perpendicular unit vector.
    assert abs(q[3]) < 1e-5
    assert abs(np.linalg.norm(q[:3]) - 1.0) < 1e-5


def test_from_two_vectors_rotates_v1_to_v2():
    """The quaternion produced by quat_from_two_vectors(v1, v2) really does
    rotate v1 to v2 when applied via quat_rotate."""
    cases = [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.5, 0.5, 0.5]), np.array([-0.3, 0.7, 0.2])),
    ]
    for v1, v2 in cases:
        q = quat_from_two_vectors(v1, v2)
        rotated = quat_rotate(q, v1)
        # Compare as direction vectors.
        np.testing.assert_array_almost_equal(
            vec_normalize(rotated), vec_normalize(v2), decimal=4
        )


def test_quat_mul_identity_neutral():
    q = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_almost_equal(quat_mul(q, quat_identity()), q)
    np.testing.assert_array_almost_equal(quat_mul(quat_identity(), q), q)


def test_quat_inverse_round_trips():
    q = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.7]))
    q_inv = quat_inverse(q)
    np.testing.assert_array_almost_equal(quat_mul(q, q_inv), quat_identity(), decimal=4)


def test_quat_rotate_identity_no_op():
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(quat_rotate(quat_identity(), v), v)


def test_quat_from_basis_identity():
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    forward = np.array([0.0, 0.0, 1.0])
    q = quat_from_basis(right, up, forward)
    np.testing.assert_array_almost_equal(q, quat_identity(), decimal=5)


def test_quat_from_basis_rotates_consistently():
    # 90° rotation around Y: right (X) → forward direction (Z)
    right = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, 1.0, 0.0])
    forward = np.array([1.0, 0.0, 0.0])
    q = quat_from_basis(right, up, forward)
    rotated = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(rotated, [0.0, 0.0, -1.0], decimal=4)
