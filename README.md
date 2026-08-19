# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python


# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------

def policy_evaluation(policy, env, gamma, theta, max_iter=1000):
    """
    Evaluate a given policy by iteratively updating the state-value function.
    """
    V = np.zeros(n_states)

    for _ in range(max_iter):
        delta = 0.0

        for state in range(n_states):
            v_state = 0.0

            for action, action_prob in enumerate(policy[state]):
                for probability, next_state, reward, terminated in env.P[state][action]:
                    v_state += action_prob * probability * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            delta = max(delta, abs(v_state - V[state]))
            V[state] = v_state

        if delta < theta:
            break
    return V

# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------
def policy_improvement(env, V, gamma):
    """
    Improve the policy greedily with respect to the current value function.
    """
    policy = np.zeros((n_states, n_actions))

    for state in range(n_states):
        action_values = one_step_lookahead(env, state, V, gamma)
        best_action = np.argmax(action_values)
        policy[state, best_action] = 1.0

    return policy

#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------
def policy_iteration(env, gamma, theta, max_iterations=1000):
    """
    Run policy iteration until the policy converges.
    """
    policy = np.ones((n_states, n_actions)) / n_actions
    V = np.zeros(n_states)
    print("-------------------------------------------------")
    print("Before Policy Iteration:")
    print("-------------------------------------------------")
    print_value_function(V)

    for iteration in range(max_iterations):
        V = policy_evaluation(policy, env, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)

        if np.allclose(policy, new_policy):
            print(f"Policy iterations: {iteration + 1}")
            print("-------------------------------------------------")
            print("After Policy Iteration :")
            print("-------------------------------------------------")
            print_value_function(V)
            print_policy(policy)
            return new_policy, V

        policy = new_policy
        print(f"Policy iterations: {iteration +1}")
        print_value_function(V)
        print_policy(policy)
        print()

    print_value_function(V)
    return policy, V




```

## Output

<img width="272" height="655" alt="image" src="https://github.com/user-attachments/assets/3d853336-1b8d-4043-b39b-9b2da146c537" />
<img width="269" height="511" alt="image" src="https://github.com/user-attachments/assets/80eb488a-1ca9-4145-a092-1beec2532d15" />

<img width="288" height="268" alt="image" src="https://github.com/user-attachments/assets/b67694df-4427-4544-9d4a-e4baada759d9" />


---

## Result

```text
The Policy Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The algorithm repeatedly performed policy evaluation and policy improvement until the policy became stable. The resulting value function represents the expected discounted return from each state, while the final policy provides the optimal action for each state under the given environment dynamics.


```
---

## Inference
```text
Policy Iteration successfully solves the finite Markov Decision Process by alternating between evaluating the current policy and improving it greedily. The policy becomes stable after a finite number of iterations, indicating that no further improvement is possible. The experiment demonstrates how Dynamic Programming can be used to obtain an optimal policy when the transition probabilities and rewards of the environment are known.

```
---

