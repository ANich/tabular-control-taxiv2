import numpy as np

SARSA, EXPECTEDSARSA, QLEARNING = 'Sarsa', 'ExpectedSarsa', 'QLearning'


class Agent:
    """
    Agent class interacts with the environment using a softmax (Boltzmann)
    exploration policy, and updates an action-value function based on the specified control
    method. For simplicity, we assume tabular states and a discrete action space.

    Parameters
    ---------

    n_states: ``int``
        The number of states.

    n_actions: ``int``
        The number of discrete actions.

    temperature: ``float``
        Tau in softmax action selection. Tau > 0. High temperature usually causes actions 
        to be similarly probable.
        Additional reference: http://www.incompleteideas.net/book/ebook/node17.html

    learning_rate: ``float``
        The learning rate for the value function update. learning_rate > 0.

    discount: ``float``
        Gamma in the discounted return. 0 <= Gamma <= 1. 

    control_method: ``str``
        The method used to update the value function. One of: Sarsa, Expected Sarsa, Q-Learning.

    """

    def __init__(self,
                 n_states: int = 10,
                 n_actions: int = 10,
                 temperature: float = 0.1,
                 learning_rate: float = 0.1,
                 discount: float = 1,
                 control_method: str = SARSA
                 ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.discount = discount
        self.control_method = control_method

        # Initialize the action-value function
        self.q_sa = np.zeros((self.n_states, self.n_actions), dtype=float)

    def exploration_policy(self, state: int) -> int:
        """
        Given a state, this method returns an action index according to the Boltzmann Softmax policy.

        Parameters
        ----------

        state: ``int``
            Index of the given state for which an action is required.

        Returns 
        -------

        The index of the action to take.
        """

        # Get the action values (numpy array of shape: [n_actions])
        action_values = self.q_sa[state]

        softmax_action_values = np.exp(
            action_values / self.temperature) / sum(np.exp(action_values / self.temperature))

        return np.random.choice(self.n_actions, p=softmax_action_values)

    def update_value_function(self, state: int, action: int, next_state: int, reward: int, done: bool) -> None:
        """
        This method delegates responsibility for updating the action value
        to the appropriate control method.

        Parameters
        ----------

        state: ``int``
            The state for which an action-value is being updated. (Usually state just left).

        action: ``int``
            The action for which an action-value is being updated. (Usually action just taken).

        next_state: ``int``
            The state the agent entered after taking action ``action`` in state ``state``. 

        reward: ``int``
            The reward observed from taking action ``action`` in state ``state``.

        done: ``bool``
            Whether we have reached the terminal state or not. This is useful because some algorithms
            e.g. SARSA set action values to 0 after the terminal state.
        """

        if self.control_method == SARSA:
            self.sarsa_update(state, action, next_state, reward, done)

        elif self.control_method == EXPECTEDSARSA:
            self.expected_sarsa_update(state, action, next_state, reward, done)

        elif self.control_method == QLEARNING:
            self.q_learning_update(state, action, next_state, reward, done)

    def sarsa_update(self, state: int, action: int, next_state: int, reward: int, done: bool) -> None:
        """
        This is the update method for on-policy SARSA. In this case we update an action value based on
        the value of taking the next action as dictated by the behaviour (exploration) policy.

        """
        policy_action = self.exploration_policy(next_state)

        # Q(Terminal, .) = 0
        if done:
            action_value_ns = 0
        else:
            action_value_ns = self.q_sa[next_state, policy_action]

        self.q_sa[state, action] = self.q_sa[state, action] + self.learning_rate * \
            (reward + (self.discount *
                       action_value_ns - self.q_sa[state, action]))

    def expected_sarsa_update(self, state: int, action: int, next_state: int, reward: int, done: bool) -> None:
        """
        This is the update method for on-policy Expected SARSA. In this case we update the action values 
        using the expectation of the next action-value under the exploration policy.

        """

        action_values = self.q_sa[next_state]

        # Action probabilities under softmax probabilities:
        action_probabilities = np.exp(
            action_values / self.temperature) / sum(np.exp(action_values / self.temperature))

        self.q_sa[state, action] = self.q_sa[state, action] + self.learning_rate * \
            (reward + (self.discount *
                       np.dot(action_probabilities, self.q_sa[next_state]) - self.q_sa[state, action]))

    def q_learning_update(self, state: int, action: int, next_state: int, reward: int, done: bool) -> None:
        """
        This is the update method for off-policy Q-Learning. In this case we update an action value based on
        the value of taking the next action as dictated by the maximum action-value we could get from the
        next state.

        """

        max_action = np.argmax(self.q_sa[next_state])

        # Q(Terminal, .) = 0
        if done:
            action_value_ns = 0
        else:
            action_value_ns = self.q_sa[next_state, max_action]

        self.q_sa[state, action] = self.q_sa[state, action] + self.learning_rate * \
            (reward + (self.discount *
                       action_value_ns - self.q_sa[state, action]))

    def greedy_policy(self, state: int) -> int:
        """
        Given a state, this function returns an action index according to the (deterministic) policy that is
        greedy w.r.t. the current action-value function.

        Parameters
        ----------

        state: ``int``
            Index of the given state for which an action is required.

        Returns 
        -------

        The index of the action to take.
        """

        return np.argmax(self.q_sa[state])
