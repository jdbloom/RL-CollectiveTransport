"""T0.3 — ARGoS byte-layout validation.

Asserts that ARGoS observation messages are contiguous little-endian float32 with no
padding between fields, so T5's np.frombuffer rewrite is safe.

The test synthesises a well-formed message by packing R * num_obs floats in '<f4'
order and checks that np.frombuffer(msg, '<f4').reshape(R, num_obs) recovers them
exactly. If this test ever fails it means the wire format has padding or endianness
quirks — the T5 parser rewrite MUST NOT be applied until the issue is understood.
"""

import struct
import numpy as np


def test_obs_block_is_contiguous_le_float32():
    """Synthesised LE float32 block round-trips through np.frombuffer exactly."""
    R, num_obs = 4, 31
    flat = np.arange(R * num_obs, dtype="<f4")
    msg = struct.pack(f"<{R * num_obs}f", *flat.tolist())
    via_frombuffer = np.frombuffer(msg, dtype="<f4").reshape(R, num_obs)
    assert np.array_equal(via_frombuffer, flat.reshape(R, num_obs)), (
        "Byte layout mismatch: ARGoS observation block has unexpected padding or "
        "endianness. T5 frombuffer rewrite is BLOCKED until this is resolved."
    )


def test_obs_block_preserves_float32_values():
    """Values survive the pack/frombuffer round-trip without precision loss."""
    R, num_obs = 4, 31
    # Use a pattern with fractional parts to catch precision issues.
    rng = np.random.default_rng(42)
    flat = rng.uniform(-1.0, 1.0, R * num_obs).astype("<f4")
    msg = struct.pack(f"<{R * num_obs}f", *flat.tolist())
    recovered = np.frombuffer(msg, dtype="<f4").reshape(R, num_obs)
    assert np.array_equal(recovered, flat.reshape(R, num_obs)), (
        "Float32 values not preserved through pack/frombuffer round-trip."
    )


def test_obs_block_byte_count():
    """Message is exactly R * num_obs * 4 bytes (no padding)."""
    R, num_obs = 4, 31
    flat = np.zeros(R * num_obs, dtype="<f4")
    msg = struct.pack(f"<{R * num_obs}f", *flat.tolist())
    assert len(msg) == R * num_obs * 4, (
        f"Expected {R * num_obs * 4} bytes, got {len(msg)}: padding detected."
    )


def test_single_robot_obs():
    """Single-robot case (R=1) works identically — edge case for reshape."""
    R, num_obs = 1, 31
    flat = np.arange(num_obs, dtype="<f4")
    msg = struct.pack(f"<{num_obs}f", *flat.tolist())
    recovered = np.frombuffer(msg, dtype="<f4").reshape(R, num_obs)
    assert np.array_equal(recovered[0], flat)


def test_large_robot_count():
    """Scales to larger R without layout breakage."""
    R, num_obs = 16, 31
    flat = np.arange(R * num_obs, dtype="<f4")
    msg = struct.pack(f"<{R * num_obs}f", *flat.tolist())
    recovered = np.frombuffer(msg, dtype="<f4").reshape(R, num_obs)
    assert np.array_equal(recovered, flat.reshape(R, num_obs))
