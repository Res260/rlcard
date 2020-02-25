''' A toy example of playing Uno with random agents
'''

import rlcard
from rlcard.agents.MyBot import MyBot
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed

# Make environment
env = rlcard.make('uno')
episode_num = 1

# Set a global seed
set_global_seed(0)

# Set up agents
agent_0 = RandomAgent(action_num=env.action_num)
agent_1 = MyBot(action_num=env.action_num)
agent_2 = RandomAgent(action_num=env.action_num)
agent_3 = RandomAgent(action_num=env.action_num)
env.set_agents([agent_0, agent_1])

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=False)
