# reinforcement_learning_tutorial
implement some algorithms using frozen lake from gymnastics

2023.07.08

Currently only using 'FrozenLake-v1'

Working files:
1. policyIteration2.py
2. valueIteration.py
3. Q_learning.py

### Experiment_1: 

<p>An experiment is designed to compare the performances of the agent’s performance under three different policies: policy iteration optimal policy, value iteration optimal policy and a random policy, checking the efficiency of two optimal policy search methods.</p>

Experiment_1 Result:
<p>Policies are tested in lakes with 16 states (a 4x4 grid) and 64 states (a 8x8 grid), and the experiment results are in Table 1 and Table 2.<p>
<p>In the 4x4 size lake, policy iteration achieved a success rate of 73.5%, while value iteration achieved a slightly higher success rate of 74.1%. This suggests that both methods are relatively effective in this smaller environment.</p>
<p>In the larger 8x8 size lake, both policy iteration and value iteration experienced a decline in success rates. Policy itera- tion achieved a success rate of 59.9%, while value iteration achieved a slightly higher success rate of 60.6%. This indi- cates that the performance of both methods decreased in the larger environment.</p>
<p>In comparison to the policy-based methods, the agent that randomly chose actions had significantly lower success rates. In the 4x4 size lake, the random agent achieved a success rate of 1.4%, and in the 8x8 size lake, it only achieved a success rate of 0.1%. This demonstrates the importance of employing structured policies, such as policy iteration and value iteration, to achieve higher success rates.</p>
<p>In summary, the experiment shows that policy iteration and value iteration outperform random action selection in both the 4x4 and 8x8 size lakes, although their effectiveness decreases in the larger environment.</p>
![Alt text](https://github.com/ConnorSiXiong/reinforcement_learning_tutorial/blob/main/IP_VP_exp.png)