# CommNet
Communication Neural Network implementation with several environments

## Background
CommNet was originally introduced in [Sukhbaatar et al. 2016](https://arxiv.org/pdf/1605.07736.pdf). The goal of the CommNet architecture is to train multiple agents to learn a task by allowing the agents to communicate to make decisions at each step. At each step, agents have access to a communication channel, and agents recieve a summed transmission of other agents. The messages transmitted are learned during training.
