"""Profile Main.py loop with manual timers.

Usage:
    # In terminal 1 (from project root):
    argos3 -c argos/collectiveRlTransport_profile.argos

    # In terminal 2 (from rl_code/):
    python profile_main.py Data/profile_run
"""
import time
import os
import sys
import yaml
import zmq
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.agent import Agent
from src.env import ZMQ_Utility, calculate_gsp_reward

recording_path = sys.argv[1] if len(sys.argv) > 1 else "Data/profile_run"
config = yaml.safe_load(open(os.path.join(recording_path, "agent_config.yml")))

# Timers
timers = {
    "zmq_recv": 0, "zmq_send": 0, "parse_msgs": 0, "filter_prox": 0,
    "make_gsp_states": 0, "choose_gsp": 0, "make_agent_state": 0,
    "choose_action": 0, "gsp_reward": 0, "store_transition": 0,
    "learn": 0, "learn_gsp": 0, "serialize": 0, "total_step": 0,
}

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f'tcp://*:{config["PORT"]}')

Utility = ZMQ_Utility()
msg = socket.recv()
Utility.get_params(msg)
Utility.set_obstacles_fields()

if Utility.params["num_prisms"] > 0:
    Utility.set_prism_sizes()
    prism_sizes = Utility.parse_prism_sizes(socket.recv())
    socket.send(b"ok")
    Utility.set_prism_points(prism_sizes)
    prism_points = Utility.parse_prism_points(socket.recv())
    socket.send(b"ok")
else:
    socket.send(b"ok")

n_obs = Utility.params["num_obs"]
n_agents = Utility.params["num_robots"]
n_actions = Utility.params["num_actions"] - 1

agent = Agent(
    config=config, network=config["LEARNING_SCHEME"],
    n_agents=n_agents, n_obs=n_obs + 1, n_actions=n_actions,
    options_per_action=config["OPTIONS_PER_ACTION"], id=0,
    min_max_action=config["MIN_MAX_ACTION"],
    meta_param_size=config["META_PARAM_SIZE"],
    gsp=config["GSP"], recurrent=config["RECURRENT"],
    attention=config["ATTENTION"], neighbors=config["NEIGHBORS"],
    gsp_input_size=config["GSP_INPUT_SIZE"],
    gsp_output_size=config["GSP_OUTPUT_SIZE"],
    gsp_min_max_action=config["GSP_MIN_MAX_ACTION"],
    gsp_look_back=config["GSP_LOOK_BACK"],
    gsp_sequence_length=config["GSP_SEQUENCE_LENGTH"],
    prox_filter_angle_deg=config["PROX_FILTER_ANGLE_DEG"],
)

print(f"Agent: {config['LEARNING_SCHEME']}+{'GSP-N' if config['NEIGHBORS'] else 'GSP' if config['GSP'] else 'IC'}")
print(f"Device: {agent.networks['actor'].device}")
print(f"Profiling {config['NUM_EPISODES']} episodes...", flush=True)

step_count = 0
episode = 0
old_cyl_ang = 0
prev_gsp = np.zeros(n_agents)
prev_states = [None] * n_agents
prev_actions = [None] * n_agents

while True:
    t_step = time.perf_counter()

    # ZMQ receive
    t0 = time.perf_counter()
    msgs = socket.recv_multipart()
    timers["zmq_recv"] += time.perf_counter() - t0

    # Parse
    t0 = time.perf_counter()
    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
    if exp_done:
        socket.send(b"ok")
        break
    env_obs, failures, rewards, stats, robot_stats, obj_stats = Utility.parse_msgs(msgs)
    timers["parse_msgs"] += time.perf_counter() - t0

    # Filter proximity
    t0 = time.perf_counter()
    agent_prox = []
    for i in range(n_agents):
        filtered, idx = agent.filter_prox_values(env_obs[i][7:], env_obs[i][5])
        agent_prox.append(np.mean(filtered) if filtered else 0)
        rewards[i] += -1 * sum(filtered)
    timers["filter_prox"] += time.perf_counter() - t0

    # GSP states
    t0 = time.perf_counter()
    gsp_states = agent.make_gsp_states(agent_prox, prev_gsp)
    timers["make_gsp_states"] += time.perf_counter() - t0

    # Choose GSP heading
    t0 = time.perf_counter()
    next_heading_gsp = agent.choose_agent_gsp(gsp_states)
    timers["choose_gsp"] += time.perf_counter() - t0

    # Make agent states + choose actions
    actions_to_take = []
    action_nums = []
    t_state = 0
    t_action = 0
    for i in range(n_agents):
        t0 = time.perf_counter()
        heading = next_heading_gsp[i] if isinstance(next_heading_gsp, list) else next_heading_gsp
        if hasattr(heading, '__len__') and len(heading) > 0:
            heading = heading[-1] if len(heading) > 1 else heading[0]
        state = agent.make_agent_state(env_obs[i], heading_gsp=heading)
        t_state += time.perf_counter() - t0

        t0 = time.perf_counter()
        action, action_num = agent.choose_agent_action(state, failures[i][0])
        t_action += time.perf_counter() - t0
        actions_to_take.append(action)
        action_nums.append(action_num)

        # Store transition from previous step
        if prev_states[i] is not None:
            t0 = time.perf_counter()
            agent.store_agent_transition(
                prev_states[i], (prev_actions[i], actions_to_take[i]),
                rewards[i], state, episode_done
            )
            timers["store_transition"] += time.perf_counter() - t0

        prev_states[i] = state
        prev_actions[i] = action_num

    timers["make_agent_state"] += t_state
    timers["choose_action"] += t_action

    # Serialize + send
    t0 = time.perf_counter()
    msg = Utility.serialize_actions(actions_to_take)
    timers["serialize"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    socket.send(msg)
    timers["zmq_send"] += time.perf_counter() - t0

    # GSP reward
    t0 = time.perf_counter()
    gsp_reward, label = calculate_gsp_reward(
        config["GSP"], old_cyl_ang, obj_stats[5], next_heading_gsp, n_agents
    )
    timers["gsp_reward"] += time.perf_counter() - t0
    old_cyl_ang = obj_stats[5]

    # Learn
    t0 = time.perf_counter()
    agent.learn()
    timers["learn"] += time.perf_counter() - t0

    # Update GSP predictions
    prev_gsp = np.array([
        h if isinstance(h, (int, float, np.floating))
        else h.item() if hasattr(h, "item")
        else float(h[0]) if hasattr(h, "__len__") else float(h)
        for h in next_heading_gsp
    ])

    timers["total_step"] += time.perf_counter() - t_step
    step_count += 1

    if episode_done:
        episode += 1
        prev_states = [None] * n_agents
        prev_actions = [None] * n_agents
        print(f"  Episode {episode-1} done ({step_count} total steps)", flush=True)

# Report
print(f"\n{'='*70}", flush=True)
print(f"RL-CT PROFILING: {step_count} steps across {episode} episodes", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Operation':<20s} {'Total (s)':<12s} {'Per-step (ms)':<14s} {'% of step':<10s}", flush=True)
print(f"{'-'*56}", flush=True)
total = timers["total_step"]
for key in [
    "zmq_recv", "parse_msgs", "filter_prox", "make_gsp_states",
    "choose_gsp", "make_agent_state", "choose_action", "serialize",
    "zmq_send", "gsp_reward", "store_transition", "learn", "total_step",
]:
    t = timers[key]
    per_step = t / step_count * 1000 if step_count > 0 else 0
    pct = t / total * 100 if total > 0 else 0
    print(f"{key:<20s} {t:<12.3f} {per_step:<14.3f} {pct:<10.1f}%", flush=True)

zmq_total = timers["zmq_recv"] + timers["zmq_send"]
compute_total = total - zmq_total
print(f"\n{'ZMQ (waiting on ARGoS):':<30s} {zmq_total:.1f}s ({zmq_total/total*100:.1f}%)", flush=True)
print(f"{'Python compute:':<30s} {compute_total:.1f}s ({compute_total/total*100:.1f}%)", flush=True)
print(f"{'Avg step time:':<30s} {total/step_count*1000:.2f} ms", flush=True)

socket.close()
context.term()
