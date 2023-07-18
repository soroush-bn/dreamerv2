import torch 
import torch.nn as nn
import numpy as np

class DiscreteActionModel(nn.Module):
    """
        A discrete action model that takes in a state and outputs a probability distribution
        over actions.

        Args:
            action_size (int): The number of possible actions.
            deter_size (int): The size of the deterministic part of the state.
            stoch_size (int): The size of the stochastic part of the state.
            embedding_size (int): The size of the embedding layer.
            actor_info (dict): A dictionary of actor hyperparameters.
            expl_info (dict): A dictionary of exploration hyperparameters.
        """
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.model = self._build_model()

    def _build_model(self):
        """
                Builds the model.

                Returns:
                    nn.Sequential: The model.
                """
        model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

        if self.dist == 'one_hot':
            model += [nn.Linear(self.node_size, self.action_size)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model) 

    def forward(self, model_state):
        """
               Performs a forward pass through the model.

               Args:
                   model_state (torch.Tensor): The model state.

               Returns:
                   torch.Tensor: The action and action distribution.
               """
        action_dist = self.get_action_dist(model_state)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist

    def get_action_dist(self, modelstate):
        """
              Gets the action distribution.

              Args:
                  modelstate (torch.Tensor): The model state.

              Returns:
                  torch.distributions.OneHotCategorical: The action distribution.

            creates a one-hot categorical distribution. This distribution can be used to represent a discrete random
            variable that can take on a finite number of values. The probability of each value
            is represented by a one-hot vector, where the value at the index of the value is 1 and all other values are 0.
              """
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)         
        else:
            raise NotImplementedError
            
    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        """
                Adds exploration to the action.

                Args:
                    action (torch.Tensor): The action.
                    itr (int): The iteration.
                    mode (str): The mode (train or eval).

                Returns:
                    torch.Tensor: The action with exploration added.
                """
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action

        raise NotImplementedError