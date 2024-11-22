import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.project_part = params['project_part']
        self.this_agent_number = agent_number  # index for this agent
        if self.project_part == 2:
            self.opponent_number = 1 - agent_number
        else:
            self.opponent_number = agent_number
        self.project_part = params['project_part']
        self.remaining_inventory = params['inventory_limit']
        self.inventory_limit = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        # Reinforcement Learning parameters
        self.state_size = 3 + 1 + 1 + 1  # Covariates (3), remaining inventory (1), time until replenish (1), profit (1)
        self.action_size = 20  # Number of discrete price bins

        self.initial_price_min = 5.0
        self.initial_price_max = 50.0
        self.price_min = self.initial_price_min
        self.price_max = self.initial_price_max
        self.price_bins = np.linspace(self.price_min, self.price_max, self.action_size)

        self.gamma = 0.95           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=5000)
        self.update_target_freq = 5  # Update target network every 5 steps
        self.steps = 0

        # Device configuration for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural Networks
        self.policy_net = self.build_model().to(self.device)
        self.target_net = self.build_model().to(self.device)
        # self.load_pretrained_model('pretrained_model.pth')  # Load the pretrained model

        self.update_target_net()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Variables to store last state and action
        self.last_state = None
        self.last_action = None
        self.profit = 0.0
        self.all_sales = []

        # Initialize price statistics
        self.prices_offered = []
        self.prices_accepted = []
        self.competitors_prices = []

        # logs
        self.rewards_log = []
        self.loss_log = []
        self.epsilon_log = []
        self.price_log = []
        self.sales_log = []
        self.inventory_log = []
        
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model


    def load_pretrained_model(self, filepath):
        try:
            self.epsilon = 0.1  # Reduced from 1.0 to 0.1

            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        except FileNotFoundError:
            print(f"Pretrained model file not found at {filepath} using initial weights.")

    def update_target_net(self):
        # Update target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def action(self, obs):
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs

        # Save new buyer covariates for constructing next_state in _process_last_sale()
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        # Construct current state vector
        if hasattr(self, 'prev_buyer_covariates'):
            buyer_covariates = self.prev_buyer_covariates
        else:
            buyer_covariates = np.zeros(3)  # If no previous covariates, initialize with zeros

        current_state = np.concatenate((
            buyer_covariates,
            [self.remaining_inventory / self.inventory_limit],
            [time_until_replenish / self.inventory_replenish],
            # [inventories[self.opponent_number] / self.inventory_limit],
            [self.profit / (self.price_max * 100)]  # Normalize profit
        ))

        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action = torch.argmax(q_values).item()

        # Map action to price
        # if len(self.all_sales) > 10:
        #     self.price_bins = np.linspace(min(self.price_min, min(self.all_sales)), max(self.price_max, max(self.all_sales)), self.action_size)
        self.adjust_price_range()
        price = self.price_bins[action]

        # Store last state and action for replay buffer
        self.last_state = current_state
        self.last_action = action

        # Save current buyer covariates for next time
        self.prev_buyer_covariates = new_buyer_covariates

        return price

    def _process_last_sale(
            self,
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        # Update internal variables
        self.remaining_inventory = inventories[self.this_agent_number]
        self.profit = state[self.this_agent_number]
        
        # Compute reward
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        if did_customer_buy_from_me:
            # Reward is profit from last sale
            reward = last_sale[1][self.this_agent_number]  # Price we sold at
            self.all_sales.append(reward)
        else:
            reward = -5#abs(last_sale[1][self.this_agent_number]-last_sale[1][self.opponent_number])
            self.all_sales.append(last_sale[1][self.opponent_number])
        
        reward += (self.inventory_limit - self.remaining_inventory) * 0.1

        # Construct next state vector
        if hasattr(self, 'next_buyer_covariates'):
            next_state_covariates = self.next_buyer_covariates
        else:
            next_state_covariates = np.zeros(3)  # Initialize if not available

        next_state = np.concatenate((
            next_state_covariates,
            [self.remaining_inventory / self.inventory_limit],
            [time_until_replenish / self.inventory_replenish],
            # [inventories[self.opponent_number] / self.inventory_limit],
            [self.profit / (self.price_max * 100)]  # Normalize profit
        ))

        # Store experience in replay buffer
        if self.last_state is not None and self.last_action is not None:
            self.memory.append((self.last_state, self.last_action, reward, next_state))

        # Learn from experiences if enough samples in memory
        if len(self.memory) >= self.batch_size:
            self.replay()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.update_target_net()

        # Update price statistics
        if last_sale[1] is not None:
            prices = last_sale[1]
            self.prices_offered.extend(prices)

            if last_sale[0] == self.this_agent_number:
                self.prices_accepted.append(prices[self.this_agent_number])

            # Update competitors' prices
            competitors_prices = np.delete(prices, self.this_agent_number)
            self.competitors_prices.extend(competitors_prices)

    def adjust_price_range(self):
        # Adjust price range based on collected data
        if len(self.prices_offered) > 100:
            # Use percentiles to exclude outliers
            offered_min = np.percentile(self.prices_offered, 5)
            offered_max = np.percentile(self.prices_offered, 95)
            accepted_min = np.percentile(self.prices_accepted, 5) if self.prices_accepted else self.initial_price_min
            accepted_max = np.percentile(self.prices_accepted, 95) if self.prices_accepted else self.initial_price_max

            # Consider both offered and accepted prices
            self.price_min = max(self.initial_price_min, min(offered_min, accepted_min))
            self.price_max = min(self.initial_price_max, max(offered_max, accepted_max))
            self.price_bins = np.linspace(self.price_min, self.price_max, self.action_size)

    def replay(self):
        # Sample minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare batches
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * max_next_q_values

        # Get current Q-values
        q_values = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
