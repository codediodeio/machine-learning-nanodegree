import random
import pandas as pd
import numpy as np
import itertools
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.actions = Environment.valid_actions    # All possible actions
        self.states = self.list_valid_actions()     # All possible states
        self.q_table = self.init_q_table()          # Q Table with zeros

        # Tuning Parameters
        self.alpha = 0.8      # learning rate           1=sensitive, 0=insensitive
        self.gamma = 0.2      # discount rate           1=long_term, 0=short_term
        self.epsilon = 0.8    # exploration rate        1=random, 0=specific
        self.decay =  0.1     # decay rate of epsilon   1=faster, 0=slower

        # Data logged for debugging/analysis
        self.n_trials = 0
        self.review = pd.DataFrame(columns=['trial', 'deadline', 'reward'])

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.n_trials += 1
        self.epsilon = self.epsilon * (1 - self.decay)
        self.debug() # debug helper

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = (inputs['light'], self.planner.next_waypoint(), inputs['oncoming'], inputs['left'])

        # Select action according to your policy
        action = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.learn(self.state, action, reward)

        # Collect for each turn debugging/analysis
        review_df = pd.DataFrame([[self.n_trials, deadline, reward]], columns=['trial', 'deadline', 'reward'])
        self.review = self.review.append(review_df)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def list_valid_actions(self):
        # Create a list of all possible state combinations
        light_states = ['red', 'green']
        waypoint_states = self.actions
        oncoming_states = Environment.valid_inputs['oncoming']
        left_states = Environment.valid_inputs['oncoming']
        arr = [light_states, waypoint_states, oncoming_states, left_states]
        possible_combos = list(itertools.product(*arr))
        return possible_combos


    def init_q_table(self):
        # Create a pandas dataframe for the Q Table and initilize with all zeros
        shape = (len(self.states), len(self.actions))
        zeros = np.zeros(shape, dtype=float)
        index = pd.MultiIndex.from_tuples(self.states, names=['light', 'waypoint', 'oncoming', 'left'])
        q_table = pd.DataFrame(zeros, index=index, columns=self.actions)
        return q_table

    def get_q_values(self, state):
        # Helper method to return a series of values from the Q Table
        light, waypoint = state[0], state[1]
        oncoming = state[2] or np.nan # convert None to NaN
        left = state[3] or np.nan
        series = self.q_table.loc[(light, waypoint, oncoming, left)]
        return series

    def random_action(self):
        # Helper method to perform a random action
        return random.choice(self.actions)

    def choose_action(self, state):
        # Select the state index in the Q Table, then get max Q Values
        state = self.state
        q_values = self.get_q_values(state)
        q_max = q_values.values.max()

        if (self.epsilon ) > random.random():
            # Choose a random action if epsilon is greater than random value
            action = self.random_action()
        else:
            # Select best action (use random choice to break ties)
            best_actions = q_values[q_values == q_max]
            action = random.choice(best_actions.index)
        return action

    def learn(self, state, action, reward):
        # Query the current q value for state/action pair
        q_values = self.get_q_values(state)
        q_current = q_values[action]

        # Get the max Q value in the next state
        inputs = self.env.sense(self)
        next_state = (inputs['light'], self.planner.next_waypoint(), inputs['oncoming'], inputs['left'])
        q_values_next = self.get_q_values(next_state)
        q_max_next = q_values_next.values.max()

        # Update Q table for the action taken using the Q formula
        q = q_current + self.alpha * ( reward + (self.gamma * q_max_next) - q_current)
        q_values.loc[action] = q.round(5)

    def debug(self):
        print ("Epsilon ---> ", self.epsilon)
        print(self.q_table)
        reward = self.review[self.review['trial'] == self.n_trials-1].reward
        print("Mean --> ", reward.mean())
        print("Min --> ", reward.min())
        print("Max --> ", reward.max())

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
