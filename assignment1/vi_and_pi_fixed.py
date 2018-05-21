### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)


def policy_evaluation(P,
                      nS,
                      nA,
                      policy,
                      gamma=0.9,
                      max_iteration=1000,
                      tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
        The value function from the given policy.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    v_f = [0.0 for i in range(nS)]
    for iter_c in range(max_iteration):
        max_delta = 0
        for cur_s in range(nS):
            chose_a = policy[cur_s]
            all_outcomes = P[cur_s][chose_a]
            value = 0
            for outcome in all_outcomes:
                prob = outcome[0]
                next_s = outcome[1]
                rew = outcome[2]
                value += prob * (float(rew) + gamma * float(v_f[next_s]))
            delta = value - v_f[cur_s]
            if delta > max_delta:
                max_delta = delta
            v_f[cur_s] = value
        if max_delta < 0.001:
            break
    return v_f


policy_stable = False


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new policy: np.ndarray
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    global policy_stable
    new_policy = np.zeros(nS, dtype='int')
    for cur_s in range(nS):
        max_value = None
        max_a = -1
        for cur_a in range(nA):
            all_outcomes = P[cur_s][cur_a]
            value = 0
            for outcome in all_outcomes:
                prob = outcome[0]
                next_s = outcome[1]
                rew = outcome[2]
                value += prob * (
                    float(rew) + gamma * float(value_from_policy[next_s]))
            if max_value is None or value > max_value:
                max_value = value
                max_a = cur_a
        if max_a != policy[cur_s]:
            policy_stable = False
        new_policy[cur_s] = max_a
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    global policy_stable
    iter_count = 0
    while not policy_stable and iter_count < max_iteration:
        iter_count += 1
        V = policy_evaluation(P, nS, nA, policy, gamma=gamma, tol=tol)
        policy_stable = True
        policy = policy_improvement(P, nS, nA, V, policy, gamma=gamma)

    return V, policy


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    # TODO(gh): implement this
    return V, policy


def example(env):
    """Show an example of gym
    Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
    """
    env.seed(0)
    from gym.spaces import prng
    prng.seed(10)  # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render()


def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
        Policy: np.array of shape [env.nS]
            The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render()
    print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    print env.__doc__
    # print "Here is an example of state, action, reward, and next state"
    # example(env)
    V_vi, p_vi = value_iteration(
        env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    V_pi, p_pi = policy_iteration(
        env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    render_single(env, p_pi)
