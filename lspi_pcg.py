import gym, pcg
import numpy as np
import lspi

solver = lspi.solvers.LSTDQSolver()
env = gym.make("SapsBattiti-v0")

init_pol = lspi.Policy(lspi.basis_functions.BattitiBasis(20),0.99, 0.1)
max_steps_per_eps = 1200
episodes = 5
rewards = []
lengths = []

for i in range(episodes):
    obs = env.reset()
    # Extracting normalizedHam and normalizedDelta
    obs = obs[4:6]

    samples = [] # collect observations of each episode ...[*]
    done = False
    c_reward = 0
    steps = 0
    while not done:
#         env.render()  # commented to speed up training
        act = init_pol.select_action(obs)
        nobs, r, done, info = env.step(act)
        c_reward += r
        samples.append(lspi.Sample(obs, act, r, nobs, done))
        obs = nobs
        steps += 1
        if steps >= max_steps_per_eps:
            break
    rewards.append(c_reward)
    lengths.append(steps)
    print('{:>6d}: {:>7.1f}'.format(i, c_reward))
    # [*]... to immediately learn from the trajectory of the episode
    init_pol = lspi.learn(samples, init_pol, solver)
env.close()