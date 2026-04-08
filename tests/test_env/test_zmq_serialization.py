"""Tests for ZMQ message serialization/deserialization round-trips."""

import numpy as np
import pytest
from struct import pack, unpack

from src.env import ZMQ_Utility


class TestParseParams:
    def test_params_parsed_correctly(self, zmq_util):
        assert zmq_util.params['num_robots'] == 4
        assert zmq_util.params['num_obstacles'] == 2
        assert zmq_util.params['num_obs'] == 31
        assert zmq_util.params['num_actions'] == 3
        assert zmq_util.params['use_gate'] == 0

    def test_params_are_int(self, zmq_util):
        assert isinstance(zmq_util.params['num_robots'], int)
        assert isinstance(zmq_util.params['num_obs'], int)

    def test_num_prisms_parsed(self, zmq_util):
        assert zmq_util.params['num_prisms'] == 0

    def test_num_prisms_is_int(self, zmq_util):
        assert isinstance(zmq_util.params['num_prisms'], int)


class TestParseStatus:
    def test_not_done(self, zmq_util):
        msg = pack('3B', 0, 0, 0)
        exp_done, ep_done, reached = zmq_util.parse_status(msg)
        assert exp_done is False
        assert ep_done is False
        assert reached is False

    def test_episode_done_reached_goal(self, zmq_util):
        msg = pack('3B', 0, 1, 1)
        exp_done, ep_done, reached = zmq_util.parse_status(msg)
        assert exp_done is False
        assert ep_done is True
        assert reached is True

    def test_experiment_done(self, zmq_util):
        msg = pack('3B', 1, 0, 0)
        exp_done, _, _ = zmq_util.parse_status(msg)
        assert exp_done is True


class TestParseObs:
    def test_obs_shape_per_robot(self, zmq_util):
        values = [float(i) for i in range(4 * 31)]
        msg = pack(f'{4*31}f', *values)
        obs = zmq_util.parse_obs(msg)
        assert len(obs) == 4
        assert obs[0].shape == (31,)
        assert obs[0].dtype == np.float32

    def test_obs_values_correct(self, zmq_util):
        values = [float(i) for i in range(4 * 31)]
        msg = pack(f'{4*31}f', *values)
        obs = zmq_util.parse_obs(msg)
        assert obs[0][0] == pytest.approx(0.0)
        assert obs[0][30] == pytest.approx(30.0)
        assert obs[1][0] == pytest.approx(31.0)
        assert obs[3][30] == pytest.approx(123.0)


class TestSerializeActions:
    def test_round_trip(self, zmq_util):
        actions = [[0.5, -0.3, 0.0], [0.1, 0.2, 0.0], [-0.1, 0.4, 0.0], [0.0, 0.0, 0.0]]
        msg = zmq_util.serialize_actions(actions)
        for r in range(4):
            offset = r * 3 * 4
            l, rw, g = unpack('3f', msg[offset:offset+12])
            assert l == pytest.approx(actions[r][0])
            assert rw == pytest.approx(actions[r][1])
            assert g == pytest.approx(actions[r][2])

    def test_output_size(self, zmq_util):
        actions = [[0.0, 0.0, 0.0]] * 4
        msg = zmq_util.serialize_actions(actions)
        assert len(msg) == 4 * 3 * 4


class TestParseRewards:
    def test_rewards_per_robot(self, zmq_util):
        msg = pack('4f', -1.5, -2.0, -1.0, -3.0)
        rewards = zmq_util.parse_rewards(msg)
        assert len(rewards) == 4
        assert rewards[0][0] == pytest.approx(-1.5)
        assert rewards[3][0] == pytest.approx(-3.0)


class TestParseObjStats:
    def test_obj_stats_shape(self, zmq_util):
        msg = pack('9f', 1.0, 2.0, 0.0, 0.0, 0.0, 45.0, 90.0, 0.5, -0.3)
        stats = zmq_util.parse_obj_stats(msg)
        assert stats.shape == (9,)
        assert stats[5] == pytest.approx(45.0)
        assert stats[7] == pytest.approx(0.5)   # comX
        assert stats[8] == pytest.approx(-0.3)  # comY


class TestParsePrismData:
    def test_prism_sizes_parse(self):
        util = ZMQ_Utility()
        params_bytes = pack('9f', 4.0, 0.0, 31.0, 3.0, 4.0, 9.0, 0.0, 10.0, 2.0)
        util.get_params(params_bytes)
        util.set_prism_sizes()
        msg = pack('2I', 4, 5)
        sizes = util.parse_prism_sizes(msg)
        assert sizes == [4, 5]

    def test_prism_points_parse(self):
        util = ZMQ_Utility()
        params_bytes = pack('9f', 4.0, 0.0, 31.0, 3.0, 4.0, 9.0, 0.0, 10.0, 2.0)
        util.get_params(params_bytes)
        util.set_prism_sizes()
        sizes = [4, 5]
        util.set_prism_points(sizes)
        coords = [float(i) * 0.1 for i in range(18)]
        msg = pack(f'{18}f', *coords)
        points = util.parse_prism_points(msg)
        assert len(points) == 18
        assert points[0] == pytest.approx(0.0)
        assert points[1] == pytest.approx(0.1)

    def test_no_prisms_no_setup_needed(self, zmq_util):
        assert zmq_util.params['num_prisms'] == 0
        assert zmq_util.PRISM_SIZE_FIELDS == []
