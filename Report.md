### Learning Algorithm
I stack with the same approach I used in the second project. I used a DDPG (Deep Deterministic Policy Gradient) algorithm and I implemented a Multi Agenti DDPG.
The two agents are learning separately, but their using the same replay buffer to improve their performance.

#### Network Architecture
As in the previous project, adding layers to the neural network augmented the performance of the algorithm. The final actor neural network has 4 fully connected hidden layers of size 256, 128, 64 and 32.
I used ReLU as the activation function and tanh is used in the final layer to return the action output.
The critic model is a neural network with two hidden layers of size 256 and 128.
I used ReLU as the activation function and tanh is used in the final layer to return the action output.

#### Hyperparameters


### Results

#### Plot of rewards

### Ideas for Future Work
As mentioned in the Deep Reinforcement Learning Nanodegree course Proximal Policy Optimization (PPO),  Asynchronous Actor-Critic Agents (A3C) and Distributed Distributional Deep Determininistic Policy Gradient (D4PG) are some of the best option to improve performance.
To further improve the algorithm, we can refer to the [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778.pdf) article. Truncated Natural Policy Gradient (TNPG) and Trust Region Policy Optimization (TRPO)  (Schulman et al., 2015) are two available options that should improve the learning speed of the algorithm.
