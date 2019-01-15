### Learning Algorithm
I stuck with the same approach I used in the second project. I used a DDPG (Deep Deterministic Policy Gradient) algorithm and I implemented a Multi Agenti DDPG.
As another [Udacity student](https://github.com/blangwallner/Udacity-Deep-Reinforcement-Learning-ND---Project-3---Collaboration-and-Competition) did, the two agents are learning separately, but their using the same replay buffer to improve their performance.

#### Network Architecture
As in the previous project, adding layers to the neural network augmented the performance of the algorithm. The final actor neural network has 4 fully connected hidden layers of size 256, 128, 64 and 32.
I used ReLU as the activation function and tanh is used in the final layer to return the action output.
The critic model is a neural network with two hidden layers of size 256 and 128.
I used ReLU as the activation function and tanh is used in the final layer to return the action output.

#### Hyperparameters

#### Final setup:

buffer_size=10000  
batch_size=1024  
gamma=0.97  
update_every=2  
noise_start=1.0  
noise_decay=1.0  
t_stop_noise=30000  
tau=1e-3  
lr_actor=1e-5  
lr_critic=1e-4  
weight_decay=0.0  

#### Previous setup:

buffer_size=10000  
batch_size=1024  
gamma=0.99  
update_every=2  
noise_start=1.0  
noise_decay=1.0  
t_stop_noise=30000  
tau=1e-3  
lr_actor=1e-4  
lr_critic=1e-3  
weight_decay=0.0  

### Results

The results shows that it takes quite some time to train the algorithm. It took more than 11 hours using the first setup and it took more than 5 hours using the final setup. I trained on my local machine using a NVIDIA GeForce GTX 1070 GPU. The training is quite slow at the begininning and it takes thousands of episodes to get to the average of 0.5 over 100 episodes.

Confronting the previous setup and the final setup, it looks that decreasing the learning rate and gamma stabilizes learning and improves performance.

#### Plot of rewards

##### Final setup:

![fifthtry](/images/fifthtry.PNG)


##### Previous setup:

![fourth_try](/images/Fourthtry.PNG)

### Ideas for Future Work
As mentioned in the Deep Reinforcement Learning Nanodegree course Proximal Policy Optimization (PPO),  Asynchronous Actor-Critic Agents (A3C) and Distributed Distributional Deep Determininistic Policy Gradient (D4PG) are some of the best option to improve performance.
To further improve the algorithm, we can refer to the [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778.pdf) article. Truncated Natural Policy Gradient (TNPG) and Trust Region Policy Optimization (TRPO)  (Schulman et al., 2015) are two available options that should improve the learning speed of the algorithm.
