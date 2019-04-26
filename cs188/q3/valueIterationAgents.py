# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()


    def runValueIteration(self):
        for i in range(self.iterations):
            values = self.values.copy()
            for state in self.mdp.getStates():
                allactionrewards = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    allactionrewards[action] = self.computeQValueFromValues(state, action)
                values[state] = allactionrewards[allactionrewards.argMax()]
            self.values = values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        qval = 0
        for (nextstate, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += prob * (self.mdp.getReward(state, action, nextstate) + self.discount*self.getValue(nextstate))

        return qval


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          Note from JSGL: We are returning indeed None if state == Terminal State
        """

        allactionrewards = util.Counter()

        for action in self.mdp.getPossibleActions(state):
            allactionrewards[action] = self.getQValue(state, action)

        return allactionrewards.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        index = 0
        for i in range(self.iterations):
            state = states[index % len(states)]
            index += 1
            allactionrewards = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                allactionrewards[action] = self.computeQValueFromValues(state, action)
            self.values[state] = allactionrewards[allactionrewards.argMax()]


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Compute predecessors of all state
        predecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for (stateprime, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if stateprime not in predecessors.keys():
                        predecessors[stateprime] = set()
                    predecessors[stateprime].add(state)

        # Initialize an empty priority queue
        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            # Find max absolute value
            if self.mdp.isTerminal(state):
                continue
            current_state_value = self.values[state]
            allactionrewards = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                allactionrewards[action] = self.computeQValueFromValues(state, action)
            max_state_value = allactionrewards[allactionrewards.argMax()]
            diff = abs(current_state_value-max_state_value)

            # Push s in priority queue
            queue.push(state, diff*-1)

        for i in range(self.iterations):
            # If the priority queue is empty, then terminate
            if queue.isEmpty():
                break

            # Pop a state s off the priority queue
            state = queue.pop()

            # If not terminal state, update s value
            if not self.mdp.isTerminal(state):
                allactionrewards = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    allactionrewards[action] = self.computeQValueFromValues(state, action)
                self.values[state] = allactionrewards[allactionrewards.argMax()]

            # For each predecessor p of s, do:
            for p in predecessors[state]:
                # Find max absolute value
                current_state_value = self.values[p]
                allactionrewards = util.Counter()
                for action in self.mdp.getPossibleActions(p):
                    allactionrewards[action] = self.computeQValueFromValues(p, action)
                max_state_value = allactionrewards[allactionrewards.argMax()]
                diff = abs(current_state_value-max_state_value)

                # if diff > theta, push p into priority queue
                if diff > self.theta:
                    queue.update(p, diff*-1)

