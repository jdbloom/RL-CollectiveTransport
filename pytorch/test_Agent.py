import unittest
import torch as T
import numpy as np
from python_code.Agent import Agent


class TestAgent(unittest.TestCase):
    #------------------
    # Testing the Make Comms function
    #------------------
    def test_make_comms(self):
        agent = Agent(num_agents = 6, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DDQN',
                               comms_scheme = 'None', alphabet_size = 4)
        #------------------
        #Test None
        #------------------
        agent.comms_scheme = 'None'
        agent.make_comms_scheme()
        self.assertEqual(len(agent.left_contacts), agent.num_agents)
        self.assertEqual(len(agent.right_contacts), agent.num_agents)
        for i in range(agent.num_agents):
            self.assertEqual(agent.left_contacts[i], [])
            self.assertEqual(agent.right_contacts[i], [])
        #------------------
        # Test Left
        #------------------
        agent.comms_scheme = 'Left'
        agent.make_comms_scheme()
        self.assertEqual(len(agent.left_contacts), agent.num_agents)
        self.assertEqual(len(agent.right_contacts), agent.num_agents)
        for i in range(agent.num_agents):
            self.assertEqual(agent.left_contacts[i][0], (i+1)%agent.num_agents)
            self.assertEqual(agent.right_contacts[i], [])
        #------------------
        #Test Right
        #------------------
        agent.comms_scheme = 'Right'
        agent.make_comms_scheme()
        self.assertEqual(len(agent.left_contacts), agent.num_agents)
        self.assertEqual(len(agent.right_contacts), agent.num_agents)
        for i in range(agent.num_agents):
            self.assertEqual(agent.left_contacts[i], [])
            self.assertEqual(agent.right_contacts[i][0], (i+agent.num_agents - 1)%agent.num_agents)
        #------------------
        #Test Both
        #------------------
        agent.comms_scheme = 'Neighbors'
        agent.make_comms_scheme()
        self.assertEqual(len(agent.left_contacts), agent.num_agents)
        self.assertEqual(len(agent.right_contacts), agent.num_agents)
        for i in range(agent.num_agents):
            self.assertEqual(agent.left_contacts[i][0], (i+1)%agent.num_agents)
            self.assertEqual(agent.right_contacts[i][0], (i+agent.num_agents - 1)%agent.num_agents)
        #------------------
        #Test Unknown Communication Scheme
        #------------------
        agent.comms_scheme = 'SkipOver'
        with self.assertRaises(Exception):
            agent.make_comms_scheme()

    #------------------
    # Testing the Make Learning Scheme function
    #------------------
    def test_make_learning_scheme(self):
        agent = Agent(num_agents = 4, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DDQN',
                               comms_scheme = 'None', alphabet_size = 4)
        #------------------
        #Test None
        #------------------
        agent.learning_scheme = 'None'

        #------------------
        #Test DQN
        #------------------
        agent.learning_scheme = 'DQN'
        agent.make_learning_scheme()
        state = np.random.rand(31).astype(np.float32)
        #testing q_eval
        actions = agent.q_eval(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.num_ops_per_action**agent.num_actions)
        #testing q_next
        actions = agent.q_next(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.num_ops_per_action**agent.num_actions)

        #------------------
        #Test DDQN
        #------------------
        agent.learning_scheme = 'DDQN'
        agent.make_learning_scheme()
        observations = np.random.rand(31)
        comms = np.random.rand(2*agent.alphabet_size)
        state = np.concatenate((observations, comms)).astype(np.float32)
        #testing q_eval
        actions = agent.q_eval(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.num_ops_per_action**agent.num_actions)
        #testing q_next
        actions = agent.q_next(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.num_ops_per_action**agent.num_actions)
        #testing q_comms_eval
        actions = agent.q_comms_eval(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.alphabet_size)
        #testing q_comms_next
        actions = agent.q_comms_next(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.alphabet_size)

        #------------------
        #Test DDPG
        #------------------
        agent.learning_scheme = 'DDPG'
        agent.make_learning_scheme()
        observations = np.random.rand(31).astype(np.float32)
        comms = np.random.rand(2*agent.alphabet_size)
        state = np.concatenate((observations, comms)).astype(np.float32)
        #testing actor
        action = agent.actor(T.from_numpy(state).to(agent.actor.device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        for i in range(action.shape[0]):
            self.assertTrue(action[i] <= agent.min_max_action and action[i] >= -agent.min_max_action)
        #testing target actor
        action = agent.target_actor(T.from_numpy(state).to(agent.target_actor.device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        for i in range(action.shape[0]):
            self.assertTrue(action[i] <= agent.min_max_action and action[i] >= -agent.min_max_action)
        #testing critic
        actions = T.from_numpy(np.array((1, -1)).astype(np.float32)).to(agent.critic.device)
        action_value = agent.critic([T.from_numpy(state).to(agent.critic.device).unsqueeze(0), actions.unsqueeze(0)]).squeeze(0)
        self.assertEqual(action_value.shape[0], 1)
        #testing target critic
        action_value = agent.target_critic([T.from_numpy(state).to(agent.critic.device).unsqueeze(0), actions.unsqueeze(0)]).squeeze(0)
        self.assertEqual(action_value.shape[0], 1)
        #testing q_comms_eval
        actions = agent.q_comms_eval(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.alphabet_size)
        #testing q_comms_next
        actions = agent.q_comms_next(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.alphabet_size)
        #------------------
        #Test TD3
        #------------------
        agent.learning_scheme = 'TD3'
        agent.make_learning_scheme()
        observations = np.random.rand(31).astype(np.float32)
        comms = np.random.rand(2*agent.alphabet_size)
        state = np.concatenate((observations, comms)).astype(np.float32)
        #testing actor
        action = agent.actor(T.from_numpy(state).to(agent.actor.device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        for i in range(action.shape[0]):
            self.assertTrue(action[i] <= agent.min_max_action and action[i] >= -agent.min_max_action)
        #testing target actor
        action = agent.target_actor(T.from_numpy(state).to(agent.actor.device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        for i in range(action.shape[0]):
            self.assertTrue(action[i] <= agent.min_max_action and action[i] >= -agent.min_max_action)
        #testing critic 1
        actions = T.from_numpy(np.array((1, -1)).astype(np.float32)).to(agent.critic_1.device)
        action_value = agent.critic_1(T.from_numpy(state).to(agent.critic_1.device).unsqueeze(0), actions.unsqueeze(0)).squeeze(0)
        self.assertEqual(action_value.shape[0], 1)
        #testing target critic 1
        actions = T.from_numpy(np.array((1, -1)).astype(np.float32)).to(agent.target_critic_1.device)
        action_value = agent.target_critic_1(T.from_numpy(state).to(agent.target_critic_1.device).unsqueeze(0), actions.unsqueeze(0)).squeeze(0)
        self.assertEqual(action_value.shape[0], 1)
        #testing critic 2
        actions = T.from_numpy(np.array((1, -1)).astype(np.float32)).to(agent.critic_2.device)
        action_value = agent.critic_2(T.from_numpy(state).to(agent.critic_2.device).unsqueeze(0), actions.unsqueeze(0)).squeeze(0)
        self.assertEqual(action_value.shape[0], 1)
        #testing target critic 2
        actions = T.from_numpy(np.array((1, -1)).astype(np.float32)).to(agent.target_critic_2.device)
        action_value = agent.target_critic_2(T.from_numpy(state).to(agent.target_critic_2.device).unsqueeze(0), actions.unsqueeze(0)).squeeze(0)
        self.assertEqual(action_value.shape[0], 1)
        #testing q_comms_eval
        actions = agent.q_comms_eval(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.alphabet_size)
        #testing q_comms_next
        actions = agent.q_comms_next(T.from_numpy(state).to(agent.q_eval.device).unsqueeze(0)).squeeze(0)
        self.assertEqual(actions.shape[0], agent.alphabet_size)
        #------------------
        #Test Unknown Learning Scheme
        #------------------
        agent.learning_scheme = 'TheBest'
        with self.assertRaises(Exception):
            agent.make_learning_scheme()

    def test_choose_action(self):
        agent = Agent(num_agents = 4, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DDQN',
                               comms_scheme = 'None', alphabet_size = 4)
        observations = np.random.rand(31).astype(np.float32)
        comms = np.random.rand(2*agent.alphabet_size)
        state = np.concatenate((observations, comms))
        #------------------
        #Test Failure
        #------------------
        agent.learning_scheme = 'None'
        agent.make_learning_scheme()
        actions, action_code = agent.choose_action(observations, failure = True)
        self.assertEqual(actions, [0, 0, 1])
        self.assertEqual(action_code, 9)
        #------------------
        #Test None
        #------------------
        agent.learning_scheme = 'None'
        agent.make_learning_scheme()
        actions, action_code = agent.choose_action(observations, failure = False)
        self.assertEqual(actions, [0, 0, 1])
        self.assertEqual(action_code, 9)
        #------------------
        #Test DQN
        #------------------
        agent.learning_scheme = 'DQN'
        agent.make_learning_scheme()
        actions, action_code = agent.choose_action(observations, failure = False)
        self.assertEqual(actions.shape[0], 3)
        self.assertEqual(actions[2], 0)
        for i in range(agent.num_actions):
            self.assertTrue(actions[i] <= agent.min_max_action and actions[i] >= -agent.min_max_action)
        #------------------
        #Test DDQN
        #------------------
        agent.learning_scheme = 'DDQN'
        agent.make_learning_scheme()
        actions, action_code = agent.choose_action(observations, failure = False)
        self.assertEqual(actions.shape[0], 3)
        self.assertEqual(actions[2], 0)
        for i in range(agent.num_actions):
            self.assertTrue(actions[i] <= agent.min_max_action and actions[i] >= -agent.min_max_action)
        #------------------
        #Test DDPG
        #------------------
        agent.learning_scheme = 'DDPG'
        agent.make_learning_scheme()
        actions, action_code = agent.choose_action(state, failure = False)
        self.assertEqual(actions.shape[0], 3)
        self.assertEqual(actions[2], 0)
        for i in range(agent.num_actions):
            self.assertTrue(actions[i] <= agent.min_max_action and actions[i] >= -agent.min_max_action)
        #------------------
        #Test TD3
        #------------------
        agent.learning_scheme = 'TD3'
        agent.make_learning_scheme()
        actions, action_code = agent.choose_action(state, failure = False)
        self.assertEqual(actions.shape[0], 3)
        self.assertEqual(actions[2], 0)
        for i in range(agent.num_actions):
            self.assertTrue(actions[i] <= agent.min_max_action and actions[i] >= -agent.min_max_action)


    def test_parse_action(self):
        agent = Agent(num_agents = 4, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DQN',
                               comms_scheme = 'None', alphabet_size = 4)
        self.assertEqual(agent.parse_action(0)[0], -0.1)
        self.assertEqual(agent.parse_action(0)[1], -0.1)
        self.assertEqual(agent.parse_action(1)[0], -0.1)
        self.assertEqual(agent.parse_action(1)[1], 0)
        self.assertEqual(agent.parse_action(2)[0], -0.1)
        self.assertEqual(agent.parse_action(2)[1], 0.1)
        self.assertEqual(agent.parse_action(3)[0], 0)
        self.assertEqual(agent.parse_action(3)[1], -0.1)
        self.assertEqual(agent.parse_action(4)[0], 0)
        self.assertEqual(agent.parse_action(4)[1], 0)
        self.assertEqual(agent.parse_action(5)[0], 0)
        self.assertEqual(agent.parse_action(5)[1], 0.1)
        self.assertEqual(agent.parse_action(6)[0], 0.1)
        self.assertEqual(agent.parse_action(6)[1], -0.1)
        self.assertEqual(agent.parse_action(7)[0], 0.1)
        self.assertEqual(agent.parse_action(7)[1], 0)
        self.assertEqual(agent.parse_action(8)[0], 0.1)
        self.assertEqual(agent.parse_action(8)[1], 0.1)
        with self.assertRaises(Exception):
            agent.parse_action(10)

    def test_choose_message(self):
        agent = Agent(num_agents = 4, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DQN',
                               comms_scheme = 'Neighbors', alphabet_size = 4)
        observations = np.random.rand(31).astype(np.float32)
        comms = np.random.rand(2*agent.alphabet_size)
        state = np.concatenate((observations, comms))
        #testing failure
        message, message_code = agent.choose_message(state, failure = True)
        self.assertEqual(message_code, -1)
        self.assertEqual(message.shape[0], agent.alphabet_size)
        for j in range(agent.alphabet_size):
            self.assertEqual(message[j], 0)
        #testing message
        message, message_code = agent.choose_message(state, failure = False)
        self.assertEqual(message.shape[0], agent.alphabet_size)
        self.assertTrue(message_code >= 0 and message_code <= (agent.alphabet_size-1))

    def test_decrement_epsilon(self):
        agent = Agent(num_agents = 4, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DQN',
                               comms_scheme = 'Neighbors', alphabet_size = 4)
        agent.epsilon = 1.0
        agent.decrement_epsilon()
        self.assertEqual(agent.epsilon, 1.0 - agent.eps_dec)
        agent.epsilon = agent.eps_min
        agent.decrement_epsilon()
        self.assertEqual(agent.epsilon, agent.eps_min)
        agent.epsilon = 0
        agent.decrement_epsilon()
        self.assertEqual(agent.epsilon, agent.eps_min)

    def test_memory(self):
        # Testing:
        #           - make_agent_state
        #           - store_transition
        #           - store_comms_transition
        #           - sample_memory
        #           - sample_comms_memory
        agent = Agent(num_agents = 4, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DDQN',
                               comms_scheme = 'Neighbors', alphabet_size = 4)

        for i in range(agent.batch_size):
            observation = np.random.rand(31).astype(np.float32)
            comms = np.random.rand(2*agent.alphabet_size)
            state = agent.make_agent_state(observation, comms)
            action = np.random.choice(agent.num_ops_per_action**agent.num_actions)
            message = np.random.choice(agent.alphabet_size)
            reward = np.random.random()
            state_ = state
            done = False
            agent.store_transition(state, action, reward, state_, done)
            agent.store_comms_transition(state, message, reward, state_, done)
        s, a, r, s_, d = agent.sample_memory()
        self.assertEqual(s.shape[0], agent.batch_size)
        self.assertEqual(s.shape[1], 31+2*agent.alphabet_size)
        self.assertEqual(a.shape[0], agent.batch_size)
        self.assertEqual(r.shape[0], agent.batch_size)
        self.assertEqual(s_.shape[0], agent.batch_size)
        self.assertEqual(s_.shape[1], 31+2*agent.alphabet_size)
        self.assertEqual(d.shape[0], agent.batch_size)

        s, a, r, s_, d = agent.sample_comms_memory()
        self.assertEqual(s.shape[0], agent.batch_size)
        self.assertEqual(s.shape[1], 31+2*agent.alphabet_size)
        self.assertEqual(a.shape[0], agent.batch_size)
        self.assertEqual(r.shape[0], agent.batch_size)
        self.assertEqual(s_.shape[0], agent.batch_size)
        self.assertEqual(s_.shape[1], 31+2*agent.alphabet_size)
        self.assertEqual(d.shape[0], agent.batch_size)

    def testDQN_learn(self):
        agent = Agent(num_agents = 4, num_observations = 31,
                               num_actions = 2, num_ops_per_action = 3,
                               id = 1, learning_scheme = 'DQN',
                               comms_scheme = 'Neighbors', alphabet_size = 4)

        for i in range(agent.batch_size):
            state = np.random.rand(31).astype(np.float32)
            action = np.random.choice(agent.num_ops_per_action**agent.num_actions)
            reward = np.random.random()
            state_ = state
            done = False
            agent.store_transition(state, action, reward, state_, done)

        agent.learn()


if __name__ == '__main__':
    unittest.main()
