import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_world_comm_v3
from collections import deque
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import copy
from tqdm import tqdm
import torch.nn.functional as F

def compute_loss(agent, states, actions, rewards, next_states, dones):
    """标准PPO损失计算"""
    # 计算目标值和优势
    with torch.no_grad():
        # 计算V(s)和V(s')
        values = agent.critic(states).squeeze()
        next_values = agent.target_critic(next_states).squeeze()
        
        # 计算目标值和优势
        returns = rewards + agent.gamma * next_values * (1 - dones)
        advantages = returns - values
        
        # 重要：标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 获取旧策略的概率
        old_logits = agent.old_actor(states)
        old_probs = F.softmax(old_logits, dim=1)
        old_log_probs = F.log_softmax(old_logits, dim=1)
        old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    
    # 计算新策略的概率
    logits = agent.actor(states)
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    
    # 计算策略比率
    ratio = torch.exp(action_log_probs - old_action_log_probs)
    
    # PPO的裁剪目标
    clip_range = 0.2  # 标准PPO使用0.2
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    
    # 计算值函数损失
    value_loss = F.mse_loss(agent.critic(states).squeeze(), returns)
    
    # 计算熵
    entropy = -(probs * log_probs).sum(dim=1).mean()
    
    return value_loss, policy_loss, entropy

# 1. 首先是所有辅助函数
def shape_reward(reward, done, step, max_steps):
    """更好的奖励整形"""
    # 保留原始奖励
    shaped_reward = reward
    
    # 为good agents提供更强的导向信号
    if not done:
        # 鼓励生存
        shaped_reward += 0.2
        
    # 避免在接近结束时早退的惩罚    
    if done and step < max_steps - 5:
        # 提前结束给予轻微惩罚
        shaped_reward -= 0.5
        
    return shaped_reward

def save_model(agent, path):
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, path)

def load_model(agent, path):
    checkpoint = torch.load(path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    agent.sync_networks()

def evaluate_agents(env, agents, num_episodes=5):
    """评估智能体性能"""
    eval_rewards = {agent: [] for agent in env.possible_agents if agent.startswith('agent_')}
    
    for episode in range(num_episodes):
        env.reset(seed=episode + 10000)  # 使用不同的种子
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        done = {agent: False for agent in env.possible_agents}
        step = 0
        
        while step < max_steps and not all(done.values()):
            for agent in env.agent_iter():
                if done[agent]:
                    continue
                    
                observation, reward, termination, truncation, info = env.last()
                done[agent] = termination or truncation
                
                if done[agent]:
                    action = None
                else:
                    if agent.startswith('agent_'):
                        # 评估时不使用探索
                        with torch.no_grad():
                            state = torch.FloatTensor(observation).unsqueeze(0)
                            action = torch.argmax(agents[agent].actor(state)).item()
                    else:
                        action = env.action_space(agent).sample()
                
                env.step(action)
                if agent.startswith('agent_'):
                    episode_rewards[agent] += reward
            
            step += 1
        
        for agent in eval_rewards.keys():
            eval_rewards[agent].append(episode_rewards[agent])
    
    # 计算平均奖励
    avg_rewards = {agent: np.mean(rewards) for agent, rewards in eval_rewards.items()}
    return avg_rewards

def plot_rewards(reward_history):
    plt.figure(figsize=(12, 8))
    
    # 第一个子图：20回合移动平均
    plt.subplot(2, 1, 1)
    for agent, rewards in reward_history.items():
        if agent.startswith('agent_'):
            # 计算移动平均，从第20回合开始
            moving_averages = []
            for i in range(20, len(rewards)):
                moving_averages.append(np.mean(rewards[i-20:i]))
            plt.plot(range(20, len(rewards)), moving_averages, 
                    label=f'{agent} Moving Average', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('20-Episode Moving Average Reward')
    plt.title('Training Rewards (20-Episode Moving Average)')
    plt.legend()
    plt.grid(True)
    
    # 第二个子图：原始奖励
    plt.subplot(2, 1, 2)
    for agent, rewards in reward_history.items():
        if agent.startswith('agent_'):
            plt.plot(rewards, label=f'{agent} Raw Rewards', 
                    alpha=0.3, linewidth=1)  # 淡化原始曲线
    
    plt.xlabel('Episode')
    plt.ylabel('Raw Reward')
    plt.title('Raw Training Rewards')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def get_obs_size(env, agent):
    obs_space = env.observation_space(agent)
    if isinstance(obs_space, tuple):
        return sum(space.shape[0] for space in obs_space)
    return obs_space.shape[0]

def log_gradients(model, name):
    """监控梯度"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def process_batch(batch_data, agent, replay_buffer):
    # 设置当前智能体，这是关键修复
    replay_buffer.set_current_agent(agent)
    
    if isinstance(batch_data, list):
        states, actions, rewards, next_states, dones, current_q, td_targets, advantages = zip(*batch_data)
    else:
        states, actions, rewards, next_states, dones, current_q, td_targets, advantages = batch_data
        states = [states]
        actions = [actions]
        rewards = [rewards]
        next_states = [next_states]
        dones = [dones]
        current_q = [current_q]
        td_targets = [td_targets]
        advantages = [advantages]
    
    processed_data = replay_buffer._process_batch(states, actions, next_states)
    states_tensor, actions_tensor, next_states_tensor = processed_data
    
    # 添加数值稳定性检查
    rewards = np.clip(rewards, -10.0, 10.0)  # 裁剪奖励
    td_targets = np.clip(td_targets, -10.0, 10.0)  # 裁剪TD目标
    advantages = np.clip(advantages, -10.0, 10.0)  # 裁剪优势函数
    
    return (
        states_tensor,
        actions_tensor,
        torch.FloatTensor(rewards),
        next_states_tensor,
        torch.FloatTensor(dones),
        torch.FloatTensor(current_q),
        torch.FloatTensor(td_targets),
        torch.FloatTensor(advantages)
    )

# 2. 然后是所有类定义
class Actor(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size)
        )
        # 使用标准PPO初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # 使用标准PPO初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.network(x)

class MAPPOAgent:
    def __init__(self, obs_size, action_size, lr=3e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 分离的Actor和Critic网络
        self.actor = Actor(obs_size, action_size)
        self.old_actor = copy.deepcopy(self.actor)
        self.critic = Critic(obs_size)
        self.target_critic = copy.deepcopy(self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 学习率调度器
        self.min_lr = 1e-5
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=100, gamma=0.95)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=100, gamma=0.95)

        # 将entropy_coef改为Parameter
        self.log_entropy_coef = nn.Parameter(torch.log(torch.tensor(0.01)))  # 初始值为0.01
        self.target_entropy = -0.5  # 目标熵值
        self.entropy_coef_optimizer = optim.Adam([self.log_entropy_coef], lr=1e-4)

    @property
    def entropy_coef(self):
        return self.log_entropy_coef.exp()

    def choose_action(self, observation):
        if isinstance(observation, tuple):
            state = np.concatenate([obs.flatten() for obs in observation])
        else:
            state = observation.flatten()
        
        # 确保状态维度匹配
        state = torch.FloatTensor(state).unsqueeze(0)
        state = (state - state.mean()) / (state.std() + 1e-8)  # 标准化
        
        if random.random() < self.epsilon:
            # 完全随机探索
            if random.random() < 0.3:
                action = random.randrange(self.action_size)
            else:
                # 基于当前策略选择
                with torch.no_grad():
                    logits = self.actor(state)
                    probs = torch.softmax(logits, dim=-1)
                    # 添加噪声以增加探索
                    probs = probs.numpy().squeeze()
                    noise = np.random.normal(0, 0.1, size=probs.shape)
                    probs = np.abs(probs + noise)
                    probs = probs / np.sum(probs)
                    action = np.random.choice(self.action_size, p=probs)
        else:
            with torch.no_grad():
                logits = self.actor(state)
                action = torch.argmax(logits, dim=-1).item()
        
        # 确保动作在有效范围内
        action = max(0, min(action, self.action_size - 1))
        return action

    def sync_target_critic(self, tau=0.01):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sync_old_policy(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_networks(self):
        """同步所有网络"""
        # 同步目标critic
        self.sync_target_critic()
        # 同步old policy
        self.sync_old_policy()
        # 更新学习率
        self.update_learning_rate()

    def update_entropy_coef(self, entropy):
        entropy_loss = -(self.entropy_coef * (entropy - self.target_entropy).detach())
        self.entropy_coef_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_coef_optimizer.step()

    def update_learning_rate(self):
        # 确保学习率不会太小
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_lr)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_lr)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.temp_buffer = []
        self.temp_priorities = []
        self.current_agent = None

    def add(self, experience):
        self.temp_buffer.append(experience)
        self.temp_priorities.append(1.0)  # 新经验给予最高优先级

    def store_episode(self):
        if len(self.buffer) >= self.buffer.maxlen:
            # 随机替换一些旧数据
            replace_size = len(self.temp_buffer)
            indices = random.sample(range(len(self.buffer)), replace_size)
            for idx, (exp, pri) in enumerate(zip(self.temp_buffer, self.temp_priorities)):
                self.buffer[indices[idx]] = exp
                self.priorities[indices[idx]] = pri
        else:
            self.buffer.extend(self.temp_buffer)
            self.priorities.extend(self.temp_priorities)
        self.temp_buffer = []
        self.temp_priorities = []

    def sample(self, batch_size):
        # 基于优先级采样
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, current_q, td_targets, advantages = zip(*batch)
        
        processed_data = self._process_batch(states, actions, next_states)
        states_tensor, actions_tensor, next_states_tensor = processed_data
        
        return (
            states_tensor,
            actions_tensor,
            torch.FloatTensor(rewards),
            next_states_tensor,
            torch.FloatTensor(dones),
            torch.FloatTensor(current_q),
            torch.FloatTensor(td_targets),
            torch.FloatTensor(advantages)
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # 添加小值避免概率为0

    def _process_batch(self, states, actions, next_states):
        """处理批量数据"""
        processed_states = []
        processed_next_states = []
        processed_actions = []
        
        # 处理状态和动作
        for state, action in zip(states, actions):
            if isinstance(state, tuple):
                state = np.concatenate([s.flatten() for s in state])
            else:
                state = state.flatten()
            if len(state) > self.current_agent.obs_size:
                state = state[:self.current_agent.obs_size]
            elif len(state) < self.current_agent.obs_size:
                state = np.pad(state, (0, self.current_agent.obs_size - len(state)))
            processed_states.append(state)
            
            if action is not None:
                processed_actions.append(min(action, self.current_agent.action_size - 1))
            else:
                processed_actions.append(0)

        # 处理下一状态
        for next_state in next_states:
            if isinstance(next_state, tuple):
                next_state = np.concatenate([s.flatten() for s in next_state])
            else:
                next_state = next_state.flatten()
            if len(next_state) > self.current_agent.obs_size:
                next_state = next_state[:self.current_agent.obs_size]
            elif len(next_state) < self.current_agent.obs_size:
                next_state = np.pad(next_state, (0, self.current_agent.obs_size - len(next_state)))
            processed_next_states.append(next_state)

        return (
            torch.FloatTensor(np.array(processed_states)),
            torch.LongTensor(processed_actions),
            torch.FloatTensor(np.array(processed_next_states))
        )

    def __len__(self):
        return len(self.buffer)
    
    def set_current_agent(self, agent):
        self.current_agent = agent
    
    def get_all_data(self):
        return list(self.buffer)

    def update_priorities_from_td_error(self, indices, td_errors):
        """根据TD误差更新优先级"""
        priorities = np.abs(td_errors) + 1e-6  # 避免优先级为0
        self.update_priorities(indices, priorities)

class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def compute_values(self, critic, target_critic, gamma=0.99):
        # 首先处理当前智能体的状态
        if isinstance(self.states[0], tuple):
            states = np.array([np.concatenate([s.flatten() for s in state]) for state in self.states])
            next_states = np.array([np.concatenate([s.flatten() for s in state]) for state in self.next_states])
        else:
            states = np.array([state.flatten() for state in self.states])
            next_states = np.array([state.flatten() for state in self.next_states])
        
        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        
        # 标准化状态
        states_tensor = (states_tensor - states_tensor.mean()) / (states_tensor.std() + 1e-8)
        next_states_tensor = (next_states_tensor - next_states_tensor.mean()) / (next_states_tensor.std() + 1e-8)
        
        with torch.no_grad():
            # 计算当前Q值和目标Q值
            current_q = critic(states_tensor).squeeze()
            target_q = target_critic(next_states_tensor).squeeze()
            
            # 计算TD目标
            rewards = torch.FloatTensor(self.rewards)
            dones = torch.FloatTensor(self.dones)
            td_targets = rewards + gamma * target_q * (1 - dones)
            
            # 计算优势
            advantages = td_targets - current_q
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return current_q.numpy(), td_targets.numpy(), advantages.numpy()

    def get_data(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones
        )

# 3. 最后是主训练代码
# 创建和初始化环境
env = simple_world_comm_v3.env()
env.reset()

# 训练参数
num_episodes = 2000
batch_size = 128    # 增大batch size提高稳定性
buffer_size = 10000 # 减小缓冲区大小，避免过时样本
max_steps = 50      # 增加每个episode的步数
lr = 3e-4           # 使用PPO推荐的学习率
gamma = 0.99        # 标准折扣因子
epsilon_decay = 0.995  # 更快的探索衰减
epsilon_min = 0.05  # 更低的最小探索率
clip_range = 0.2    # 标准PPO裁剪范围
entropy_coef = 0.01 # 增加熵系数促进探索
critic_updates = 1
actor_updates = 1
gradient_accumulation_steps = 1

# Early stopping参数
patience = 200  # 给予更多机会
min_episodes = 500  # 至少训练这么多回合
best_reward = float('-inf')
no_improve_count = 0

# 创建共享经验池
shared_replay_buffer = PrioritizedReplayBuffer(buffer_size=buffer_size)

# 2. 在初始化智能体时使用正确的观察空间大小
agents = {}
for agent in env.possible_agents:
    if agent.startswith('agent_'):
        obs_size = get_obs_size(env, agent)
        action_size = env.action_space(agent).n
        agents[agent] = MAPPOAgent(
            obs_size=obs_size,
            action_size=action_size,
            lr=lr,
            gamma=gamma,
            epsilon=1.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )
    else:
        agents[agent] = None

# 记录每个智能体的奖励历史
reward_history = {agent: [] for agent in env.possible_agents}

# 在主训练循环开始前的监控指标
value_losses = []
policy_losses = []
entropies = []

# 训练循环
for episode in tqdm(range(num_episodes), desc="Training"):
    env.reset()
    step = 0
    episode_rewards = {agent: 0 for agent in env.possible_agents}
    trajectories = {agent: Trajectory() for agent in env.possible_agents if agent.startswith('agent_')}
    all_agents_done = {agent: False for agent in env.possible_agents}
    
    while step < max_steps and not all(all_agents_done.values()):
        step += 1
        for agent in env.agent_iter():
            if all_agents_done[agent]:
                continue
                
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            if done:
                action = None
                all_agents_done[agent] = True
            else:
                if agent.startswith('agent_'):
                    action = agents[agent].choose_action(observation)
                else:
                    action = env.action_space(agent).sample()
            
            env.step(action)
            
            # 只为agent_类型的智能体收集经验
            if agent.startswith('agent_'):
                shaped_reward = shape_reward(reward, done, step, max_steps)
                episode_rewards[agent] += shaped_reward
                
                if not done:
                    next_obs = env.observe(agent)
                    trajectories[agent].add(
                        observation,
                        action,
                        shaped_reward,
                        next_obs,
                        done
                    )
            else:
                episode_rewards[agent] += reward
        
        # 只在每10步打印一次累积奖励
        if step % 10 == 0:
            print(f"\nStep {step} cumulative rewards:")
            for agent, reward in episode_rewards.items():
                print(f"  {agent}: {reward}")
            print()
    
    # 计算每个轨迹的值函数和优势函数
    for agent_id, trajectory in trajectories.items():
        agent = agents[agent_id]
        current_q, td_targets, advantages = trajectory.compute_values(
            agent.critic, 
            agent.target_critic, 
            agent.gamma
        )
        
        # 存储完整的轨迹数据
        for t in range(len(trajectory.states)):
            shared_replay_buffer.add((
                trajectory.states[t],
                trajectory.actions[t],
                trajectory.rewards[t],
                trajectory.next_states[t],
                trajectory.dones[t],
                current_q[t],
                td_targets[t],
                advantages[t]
            ))
    
    # 将临时缓冲区的数据存入主缓冲区
    shared_replay_buffer.store_episode()
    
    # 策略更新阶段 - 更改为标准PPO的多次更新
    if len(shared_replay_buffer) >= batch_size:
        # 获取所有收集的数据
        all_data = shared_replay_buffer.get_all_data()
        # 随机打乱数据
        random.shuffle(all_data)
        # 执行多次更新
        n_updates = 4  # 标准PPO通常进行4-10次更新
        
        for agent_id, agent in agents.items():
            if agent_id.startswith('agent_'):
                for _ in range(n_updates):
                    try:
                        # 从缓冲区中随机采样
                        indices = np.random.randint(0, len(all_data), batch_size)
                        batch = [all_data[i] for i in indices]
                        # 处理数据
                        shared_replay_buffer.set_current_agent(agent)  # 确保设置当前智能体
                        states, actions, rewards, next_states, dones, _, _, _ = process_batch(batch, agent, shared_replay_buffer)
                        
                        # 检查数据形状
                        if len(states) == 0 or states.shape[0] != batch_size:
                            continue
                            
                        # 计算损失
                        value_loss, policy_loss, entropy = compute_loss(agent, states, actions, rewards, next_states, dones)
                        
                        # 检查损失是否为NaN
                        if torch.isnan(value_loss).any() or torch.isnan(policy_loss).any():
                            print(f"Warning: NaN in losses, skipping update for {agent_id}")
                            continue
                            
                        # 计算总损失
                        loss = value_loss * 0.5 + policy_loss - entropy_coef * entropy
                        
                        # 执行更新
                        agent.actor_optimizer.zero_grad()
                        agent.critic_optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪（标准PPO使用0.5）
                        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                        
                        agent.actor_optimizer.step()
                        agent.critic_optimizer.step()
                        
                        # 记录损失值
                        value_losses.append(value_loss.item())
                        policy_losses.append(policy_loss.item())
                        entropies.append(entropy.item())
                    except Exception as e:
                        print(f"Error during update for {agent_id}: {e}")
                        continue
                
                # 在所有更新结束后同步网络
                agent.sync_networks()

        # 在episode结束时更新学习率
        if episode > 0 and episode % 100 == 0:  # 每100个episode更新一次
            for agent_id, agent in agents.items():
                if agent_id.startswith('agent_'):
                    agent.actor_scheduler.step()
                    agent.critic_scheduler.step()

        # 每个episode打印监控信息
        if episode % 10 == 0:
            print(f"\nEpisode {episode} statistics:")
            print(f"Average value loss: {np.mean(value_losses):.4f}")
            print(f"Average policy loss: {np.mean(policy_losses):.4f}")
            print(f"Average entropy: {np.mean(entropies):.4f}")
            print(f"Buffer size: {len(shared_replay_buffer)}")
            
            # 清空监控列表
            value_losses = []
            policy_losses = []
            entropies = []
    
    # 记录每个智能体的回合奖励
    for agent in env.possible_agents:
        reward_history[agent].append(episode_rewards[agent])
        
    print(f"Episode {episode + 1} rewards for agents:", {
        k: v for k, v in episode_rewards.items() if k.startswith('agent_')
    })

    # 在episode结束时添加
    for agent_id, agent in agents.items():
        if agent_id.startswith('agent_'):
            agent.update_epsilon()

    # 计算当前episode的平均奖励
    current_reward = np.mean([reward for agent, reward in episode_rewards.items() if agent.startswith('agent_')])
    
    # Early stopping检查
    if episode < min_episodes:  # 确保最少训练一定回合
        no_improve_count = 0
    else:
        if current_reward > best_reward:
            best_reward = current_reward
            no_improve_count = 0
            # 保存最佳模型
            for agent_id, agent in agents.items():
                if agent_id.startswith('agent_'):
                    save_model(agent, f'best_model_{agent_id}.pth')
        else:
            no_improve_count += 1

    if no_improve_count >= patience and episode >= min_episodes:
        print(f"Early stopping at episode {episode}")
        break

    # 在训练循环中定期评估
    if episode % 50 == 0:
        avg_rewards = evaluate_agents(env, agents)
        print("\nEvaluation results:")
        for agent, reward in avg_rewards.items():
            print(f"{agent}: {reward:.2f}")

# 显示训练结果
plot_rewards(reward_history)

# 关闭环境
env.close()
