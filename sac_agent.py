import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ===================== ENVIRONMENT =====================
class MaggiMakerEnv:
    """Environment for Maggi making with dataset-based constraints"""
    
    def __init__(self, xgboost_model, dataset_path='maggi_dataset.txt'):
        self.model = xgboost_model
        
        # Load dataset to extract realistic bounds
        self.df = pd.read_csv(dataset_path)
        
        # Calculate action bounds from dataset (with small margin)
        self.action_bounds = self._calculate_bounds_from_dataset()
        
        # Calculate ideal ratios from high-scoring recipes (TasteScore >= 95)
        self.ideal_ratios = self._calculate_ideal_ratios()
        
        self.action_dim = len(self.action_bounds)
        self.state_dim = self.action_dim + 1  # ingredients + previous taste
        
        self.reset()
    
    def _calculate_bounds_from_dataset(self):
        """Extract realistic bounds from dataset"""
        bounds = {}
        columns = ['Maggi_Packets', 'Masala_Sachets', 'Water_ml', 'Onions_g',
                   'ChilliPowder_tbsp', 'Turmeric_tbsp', 'Salt_tbsp', 'CookingTime_min']
        
        for col in columns:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            bounds[col] = (float(min_val), float(max_val))
        
        return bounds
    
    def _calculate_ideal_ratios(self):
        """Calculate ideal ingredient ratios from high-quality recipes"""
        # Filter recipes with TasteScore >= 95
        high_quality = self.df[self.df['TasteScore'] >= 95]
        
        ratios = {}
        
        # Per-packet ratios (ingredients per Maggi packet)
        ratios['masala_per_packet'] = (
            high_quality['Masala_Sachets'] / high_quality['Maggi_Packets']
        ).mean()
        
        ratios['water_per_packet'] = (
            high_quality['Water_ml'] / high_quality['Maggi_Packets']
        ).mean()
        
        ratios['onions_per_packet'] = (
            high_quality['Onions_g'] / high_quality['Maggi_Packets']
        ).mean()
        
        ratios['chilli_per_packet'] = (
            high_quality['ChilliPowder_tbsp'] / high_quality['Maggi_Packets']
        ).mean()
        
        ratios['turmeric_per_packet'] = (
            high_quality['Turmeric_tbsp'] / high_quality['Maggi_Packets']
        ).mean()
        
        ratios['salt_per_packet'] = (
            high_quality['Salt_tbsp'] / high_quality['Maggi_Packets']
        ).mean()
        
        ratios['time_per_packet'] = (
            high_quality['CookingTime_min'] / high_quality['Maggi_Packets']
        ).mean()
        
        return ratios
    
    def reset(self):
        """Reset environment to initial state"""
        self.previous_taste = 50.0  # Start with neutral taste
        self.current_step = 0
        
        # Sample a random recipe from dataset as baseline
        sample_idx = np.random.randint(len(self.df))
        self.baseline_recipe = self.df.iloc[sample_idx][
            ['Maggi_Packets', 'Masala_Sachets', 'Water_ml', 'Onions_g',
             'ChilliPowder_tbsp', 'Turmeric_tbsp', 'Salt_tbsp', 'CookingTime_min']
        ].values
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state (previous taste + normalized baseline)"""
        normalized_baseline = self._normalize_ingredients(self.baseline_recipe)
        state = np.concatenate([[self.previous_taste / 100.0], normalized_baseline])
        return state.astype(np.float32)
    
    def _normalize_ingredients(self, ingredients):
        """Normalize ingredients to [0, 1] range"""
        normalized = []
        for i, (key, (min_val, max_val)) in enumerate(self.action_bounds.items()):
            normalized.append((ingredients[i] - min_val) / (max_val - min_val))
        return np.array(normalized)
    
    def _denormalize_action(self, action):
        """Convert normalized action [-1, 1] to actual ingredient values"""
        ingredients = []
        
        # First, determine number of packets (discrete choice)
        packets_normalized = (action[0] + 1) / 2  # Convert to [0, 1]
        min_packets, max_packets = self.action_bounds['Maggi_Packets']
        packets = np.clip(
            np.round(min_packets + packets_normalized * (max_packets - min_packets)),
            min_packets, max_packets
        )
        ingredients.append(packets)
        
        # Calculate other ingredients based on ideal ratios and packets
        # with small deviations allowed by the action
        
        # Masala Sachets
        ideal_masala = packets * self.ideal_ratios['masala_per_packet']
        masala_deviation = action[1] * 0.3 * ideal_masala  # ±30% deviation
        masala = np.clip(
            ideal_masala + masala_deviation,
            self.action_bounds['Masala_Sachets'][0],
            self.action_bounds['Masala_Sachets'][1]
        )
        ingredients.append(masala)
        
        # Water
        ideal_water = packets * self.ideal_ratios['water_per_packet']
        water_deviation = action[2] * 0.15 * ideal_water  # ±15% deviation
        water = np.clip(
            ideal_water + water_deviation,
            self.action_bounds['Water_ml'][0],
            self.action_bounds['Water_ml'][1]
        )
        ingredients.append(water)
        
        # Onions
        ideal_onions = packets * self.ideal_ratios['onions_per_packet']
        onions_deviation = action[3] * 0.3 * ideal_onions  # ±30% deviation
        onions = np.clip(
            ideal_onions + onions_deviation,
            self.action_bounds['Onions_g'][0],
            self.action_bounds['Onions_g'][1]
        )
        ingredients.append(onions)
        
        # Chilli Powder
        ideal_chilli = packets * self.ideal_ratios['chilli_per_packet']
        chilli_deviation = action[4] * 0.3 * ideal_chilli  # ±30% deviation
        chilli = np.clip(
            ideal_chilli + chilli_deviation,
            self.action_bounds['ChilliPowder_tbsp'][0],
            self.action_bounds['ChilliPowder_tbsp'][1]
        )
        ingredients.append(chilli)
        
        # Turmeric
        ideal_turmeric = packets * self.ideal_ratios['turmeric_per_packet']
        turmeric_deviation = action[5] * 0.3 * ideal_turmeric  # ±30% deviation
        turmeric = np.clip(
            ideal_turmeric + turmeric_deviation,
            self.action_bounds['Turmeric_tbsp'][0],
            self.action_bounds['Turmeric_tbsp'][1]
        )
        ingredients.append(turmeric)
        
        # Salt
        ideal_salt = packets * self.ideal_ratios['salt_per_packet']
        salt_deviation = action[6] * 0.3 * ideal_salt  # ±30% deviation
        salt = np.clip(
            ideal_salt + salt_deviation,
            self.action_bounds['Salt_tbsp'][0],
            self.action_bounds['Salt_tbsp'][1]
        )
        ingredients.append(salt)
        
        # Cooking Time
        ideal_time = packets * self.ideal_ratios['time_per_packet']
        time_deviation = action[7] * 0.25 * ideal_time  # ±25% deviation
        time = np.clip(
            ideal_time + time_deviation,
            self.action_bounds['CookingTime_min'][0],
            self.action_bounds['CookingTime_min'][1]
        )
        ingredients.append(time)
        
        return np.array(ingredients)
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        self.current_step += 1
        
        # Convert normalized action to actual ingredients
        ingredients = self._denormalize_action(action)
        
        # Predict taste using XGBoost model
        ingredients_reshaped = ingredients.reshape(1, -1)
        feature_names = ['Maggi_Packets', 'Masala_Sachets', 'Water_ml', 'Onions_g',
                        'ChilliPowder_tbsp', 'Turmeric_tbsp', 'Salt_tbsp', 'CookingTime_min']
        dmatrix = xgb.DMatrix(ingredients_reshaped, feature_names=feature_names)
        taste = self.model.predict(dmatrix)[0]
        taste = np.clip(taste, 0, 100)
        
        # Calculate reward
        reward = self._calculate_reward(taste, ingredients)
        
        # Update state
        self.previous_taste = taste
        next_state = self._get_state()
        
        # Episode ends after one step (single recipe attempt)
        done = True
        
        info = {
            'taste': taste,
            'ingredients': ingredients,
            'ingredient_names': list(self.action_bounds.keys())
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, taste, ingredients):
        """Calculate reward based purely on taste score"""
        # Main reward comes directly from taste score
        # Normalize to [-1, 1] range to help with learning
        reward = (taste - 50) / 50  
        
        # Add non-linear scaling to emphasize very good or very bad results
        # This creates a natural incentive to find optimal ratios without hard-coding them
        if taste >= 90:
            reward = reward * 2  # Amplify reward for exceptional results
        elif taste <= 30:
            reward = reward * 2  # Amplify penalty for very poor results
            
        return reward


# ===================== REPLAY BUFFER =====================
class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ===================== NEURAL NETWORKS =====================
class Actor(nn.Module):
    """Policy network (actor) for SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Q-value network (critic) for SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        # Q2
        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2


# ===================== SAC AGENT =====================
class SACAgent:
    """Soft Actor-Critic Agent"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        
        return action.cpu().detach().numpy()[0]
    
    def update(self, batch, batch_size):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update Alpha (temperature parameter)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item(), self.alpha
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


# ===================== TRAINING FUNCTION =====================
def log_episode(episode_num, info, reward, epsilon, log_file):
    """Log episode details to file"""
    with open(log_file, 'a') as f:
        f.write(f"\nEpisode {episode_num}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Epsilon (exploration rate): {epsilon:.4f}\n")
        f.write(f"Final Reward: {reward:.4f}\n")
        f.write(f"Taste Score: {info['taste']:.2f}\n")
        f.write("\nIngredient Parameters:\n")
        for name, value in zip(info['ingredient_names'], info['ingredients']):
            f.write(f"{name}: {value:.4f}\n")
        f.write("-" * 50 + "\n")

def train_sac_agent(xgboost_model, dataset_path='maggi_dataset.txt', 
                    num_episodes=2000, batch_size=256, buffer_size=100000):
    """Train SAC agent to optimize Maggi recipe"""
    
    # Initialize environment and agent
    env = MaggiMakerEnv(xgboost_model, dataset_path)
    agent = SACAgent(env.state_dim, env.action_dim)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Training metrics
    episode_rewards = []
    episode_tastes = []
    best_taste = 0
    best_recipe = None
    
    # Epsilon parameters for exploration
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = (epsilon_start - epsilon_end) / (num_episodes * 0.7)  # Decay over 70% of episodes
    
    # Create or clear log file
    log_file = "training_log.txt"
    with open(log_file, 'w') as f:
        f.write("MAGGI RECIPE OPTIMIZATION - TRAINING LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Episodes: {num_episodes}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Initial Exploration Rate (epsilon): {epsilon_start}\n")
        f.write(f"Final Exploration Rate: {epsilon_end}\n")
        f.write("=" * 50 + "\n\n")
    
    print("Starting SAC Training for Maggi Maker (Dataset-Constrained)")
    print(f"State Dim: {env.state_dim}, Action Dim: {env.action_dim}")
    print(f"Device: {agent.device}")
    print(f"\nIdeal Ratios (per packet):")
    for key, value in env.ideal_ratios.items():
        print(f"  {key}: {value:.4f}")
    print()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        # Calculate epsilon for this episode
        epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        
        # Take action with epsilon-greedy exploration
        if np.random.rand() < epsilon:
            action = np.random.uniform(-1, 1, env.action_dim)
        else:
            action = agent.select_action(state)
        
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        
        episode_reward += reward
        episode_rewards.append(episode_reward)
        episode_tastes.append(info['taste'])
        
        # Track best recipe
        if info['taste'] > best_taste:
            best_taste = info['taste']
            best_recipe = {
                name: float(value) 
                for name, value in zip(info['ingredient_names'], info['ingredients'])
            }
        
        # Update agent
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            critic_loss, actor_loss, alpha = agent.update(batch, batch_size)
        
        # Log every episode to file
        log_episode(episode + 1, info, episode_reward, epsilon, log_file)
        
        # Console logging for monitoring progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_taste = np.mean(episode_tastes[-100:])
            max_recent_taste = np.max(episode_tastes[-100:])
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (100 eps): {avg_reward:.2f}")
            print(f"  Avg Taste (100 eps): {avg_taste:.2f}")
            print(f"  Max Recent Taste: {max_recent_taste:.2f}")
            print(f"  Best Taste So Far: {best_taste:.2f}")
            if len(replay_buffer) > batch_size:
                print(f"  Alpha: {alpha:.4f}")
            print()
            
            # Also log summary to file
            with open(log_file, 'a') as f:
                f.write(f"\nSUMMARY AT EPISODE {episode + 1}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Average Reward (last 100): {avg_reward:.2f}\n")
                f.write(f"Average Taste (last 100): {avg_taste:.2f}\n")
                f.write(f"Max Recent Taste: {max_recent_taste:.2f}\n")
                f.write(f"Best Taste Overall: {best_taste:.2f}\n")
                if len(replay_buffer) > batch_size:
                    f.write(f"Current Alpha: {alpha:.4f}\n")
                f.write("-" * 50 + "\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest Taste Achieved: {best_taste:.2f}")
    print("\nOptimal Recipe:")
    for ingredient, amount in best_recipe.items():
        print(f"  {ingredient}: {amount:.2f}")
    
    # Calculate ratios for best recipe
    packets = best_recipe['Maggi_Packets']
    print(f"\nOptimal Ratios (per packet):")
    print(f"  Masala per packet: {best_recipe['Masala_Sachets']/packets:.2f}")
    print(f"  Water per packet: {best_recipe['Water_ml']/packets:.2f} ml")
    print(f"  Onions per packet: {best_recipe['Onions_g']/packets:.2f} g")
    print(f"  Time per packet: {best_recipe['CookingTime_min']/packets:.2f} min")
    
    # Save agent
    agent.save("sac_maggi_agent_constrained.pth")
    print("\nAgent saved to: sac_maggi_agent_constrained.pth")
    
    return agent, best_recipe, episode_rewards, episode_tastes


# ===================== EXAMPLE USAGE =====================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model('maggi_xgboost_model.json')
    
    # Train the agent
    agent, best_recipe, rewards, tastes = train_sac_agent(
        model, 
        dataset_path='maggi_dataset.txt',
        num_episodes=2000
    )
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    
    # Plot raw rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Plot smoothed rewards
    smoothed_rewards = gaussian_filter1d(rewards, sigma=30)
    plt.plot(smoothed_rewards, color='blue', linewidth=2, label='Smoothed Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress - Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot taste scores
    plt.subplot(2, 1, 2)
    plt.plot(tastes, alpha=0.3, color='green', label='Raw Taste Scores')
    
    # Plot smoothed taste scores
    smoothed_tastes = gaussian_filter1d(tastes, sigma=30)
    plt.plot(smoothed_tastes, color='green', linewidth=2, label='Smoothed Taste Scores')
    plt.xlabel('Episode')
    plt.ylabel('Taste Score')
    plt.title('Training Progress - Taste Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()