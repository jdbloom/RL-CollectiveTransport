import numpy as np
import pytest
from struct import pack
from src.env import ZMQ_Utility, angle_normalize_unsigned_deg, angle_normalize_signed_deg


@pytest.fixture
def zmq_util():
    util = ZMQ_Utility()
    params_bytes = pack('9f', 4.0, 2.0, 31.0, 3.0, 4.0, 9.0, 0.0, 10.0, 0.0)
    util.get_params(params_bytes)
    util.set_obstacles_fields()
    return util


class TestMessageValidation:
    def test_parse_msg_wrong_size_raises(self, zmq_util):
        wrong_msg = b"\x00" * 5
        with pytest.raises(ValueError, match="Message size mismatch"):
            zmq_util.parse_msg(wrong_msg, 'status', zmq_util.EXPERIMENT_FIELDS, zmq_util.EXPERIMENT_FMT)

    def test_parse_msg_correct_size_works(self, zmq_util):
        msg = pack('3B', 0, 0, 0)
        result = zmq_util.parse_msg(msg, 'status', zmq_util.EXPERIMENT_FIELDS, zmq_util.EXPERIMENT_FMT)
        assert result['exp_done'] == 0

    def test_parse_msgs_wrong_part_count_raises(self, zmq_util):
        msgs = [b"\x00"] * 5
        with pytest.raises(ValueError, match="Expected 7 message parts"):
            zmq_util.parse_msgs(msgs)

    def test_parse_msgs_correct_parts_works(self, zmq_util):
        num_robots = 4
        status = pack('3B', 0, 0, 0)
        obs = pack(f'{31 * num_robots}f', *([0.0] * (31 * num_robots)))
        failures = pack(f'{num_robots}I', *([0] * num_robots))
        rewards = pack(f'{num_robots}f', *([0.0] * num_robots))
        stats = pack(f'{4 * num_robots}f', *([0.0] * (4 * num_robots)))
        robot_stats = pack(f'{6 * num_robots}f', *([0.0] * (6 * num_robots)))
        obj_stats = pack('9f', *([0.0] * 9))
        msgs = [status, obs, failures, rewards, stats, robot_stats, obj_stats]
        env_obs, fail, rew, st, rst, ost = zmq_util.parse_msgs(msgs)
        assert len(env_obs) == num_robots


class TestNaNCGuards:
    def test_angle_normalize_nan_returns_zero(self):
        assert angle_normalize_unsigned_deg(float('nan')) == 0.0

    def test_angle_normalize_inf_returns_zero(self):
        assert angle_normalize_unsigned_deg(float('inf')) == 0.0

    def test_angle_normalize_signed_nan_returns_zero(self):
        assert angle_normalize_signed_deg(float('nan')) == 0.0

    def test_angle_normalize_normal_values_unchanged(self):
        assert angle_normalize_unsigned_deg(370) == pytest.approx(10)
        assert angle_normalize_unsigned_deg(-10) == pytest.approx(350)
        assert angle_normalize_signed_deg(190) == pytest.approx(-170)


class TestNamedTupleCache:
    def test_parse_msg_caches_namedtuple(self, zmq_util):
        msg = pack('3B', 0, 0, 0)
        zmq_util.parse_msg(msg, 'status', zmq_util.EXPERIMENT_FIELDS, zmq_util.EXPERIMENT_FMT)
        zmq_util.parse_msg(msg, 'status', zmq_util.EXPERIMENT_FIELDS, zmq_util.EXPERIMENT_FMT)
        assert 'status' in zmq_util._namedtuple_cache
