import numpy as np
from statistics import mean

number_players = 2

# a map of color to its index
COLOR_MAP = {'r': 0, 'g': 1, 'b': 2, 'y': 3}

# a map of trait to its index
TRAIT_MAP = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
             '8': 8, '9': 9, 'skip': 10, 'reverse': 11, 'draw_2': 12,
             'wild': 13, 'wild_draw_4': 14}

WILD = ['r-wild', 'g-wild', 'b-wild', 'y-wild']

WILD_DRAW_4 = ['r-wild_draw_4', 'g-wild_draw_4', 'b-wild_draw_4', 'y-wild_draw_4']

visits = {}

values = {}


def get_visits(state):
    if state not in visits:
        visits[state] = 0
    return visits[state]


def get_values(state):
    if state not in values:
        values[state] = []
    return values[state]


class MyBot(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): the size of the ouput action space
        '''
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        MyBot.montecarlo(state)
        return np.random.choice(state['legal_actions'])

    @staticmethod
    def montecarlo(state):
        current = state
        child_to_choose = None
        max_ucb1 = -999999999999999
        for child in current['legal_actions']:
            child_ucb1 = MyBot.UCB1(child, current['target'])
            if child_ucb1 > max_ucb1:
                child_to_choose = child
                max_ucb1 = child_ucb1

        if get_visits(child_to_choose) == 0:
            MyBot.rollout()
        else:
            # expansion()
            MyBot.rollout()

        # backprop

        return state

    @staticmethod
    def rollout(state):
        state2 = state
        for i in range(5):
            if i % number_players == number_players - 1:
                state2 = MyBot.generate_next_state(state2, state2['hand'])
            else:
                state2 = MyBot.generate_next_state_adversary(state2)
        return 1

    @staticmethod
    def UCB1(action, last_card):

        if get_visits(action) == 0:
            return 999999999999999999999999999999
        ucb1 = mean(get_values(action) or [0]) + (2 * (np.log(get_visits(last_card) / get_visits(action))))
        return ucb1

    @staticmethod
    def generate_next_state(state, action):
        next_state = {'hand': state['hand']}
        next_state['hand'].remove(action) if action in next_state['hand'] else None
        if action == 'draw':
            next_state['hand'].append('reverse')  # changeme
        next_state['target'] = action
        next_state['legal_actions'] = MyBot.get_legal_actions(next_state['target'], next_state['hand'])
        next_state['parent'] = state
        return next_state

    @staticmethod
    def generate_next_state_adversary(state, your_hand):
        next_state = {'hand': your_hand}
        #FAIRE ÇA
        next_state['target'] = action
        next_state['legal_actions'] = MyBot.get_legal_actions(next_state['target'], next_state['hand'])
        next_state['parent'] = state
        return next_state

    @staticmethod
    def get_legal_actions(target, hand):
        wild_flag = 0
        wild_draw_4_flag = 0
        legal_actions = []
        wild_4_actions = []
        if target.type == 'wild':
            for card in hand:
                if card.type == 'wild':
                    #card.color = np.random.choice(UnoCard.info['color'])
                    if card.trait == 'wild_draw_4':
                        if wild_draw_4_flag == 0:
                            wild_draw_4_flag = 1
                            wild_4_actions.extend(WILD_DRAW_4)
                    else:
                        if wild_flag == 0:
                            wild_flag = 1
                            legal_actions.extend(WILD)
                elif card.color == target.color:
                    legal_actions.append(card.str)

        # target is aciton card or number card
        else:
            for card in hand:
                if card.type == 'wild':
                    if card.trait == 'wild_draw_4':
                        if wild_draw_4_flag == 0:
                            wild_draw_4_flag = 1
                            wild_4_actions.extend(WILD_DRAW_4)
                    else:
                        if wild_flag == 0:
                            wild_flag = 1
                            legal_actions.extend(WILD)
                elif card.color == target.color or card.trait == target.trait:
                    legal_actions.append(card.str)
        if not legal_actions:
            legal_actions = wild_4_actions
        if not legal_actions:
            legal_actions = ['draw']
        return legal_actions

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state)
