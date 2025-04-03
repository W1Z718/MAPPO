import socket
import pickle
import threading
import numpy as np
import torch
import time
from pettingzoo.mpe import simple_world_comm_v3
from realworld1 import (
    MAPPOAgent, PrioritizedReplayBuffer, shape_reward, 
    compute_loss, process_batch, save_model, get_obs_size, evaluate_agents
)
from tqdm import tqdm
import json

# 服务器配置
HOST = '0.0.0.0'  # 监听所有网络接口
PORT_BASE = 5000  # 基础端口号
MAX_BUFFER = 4096 * 16  # 增大缓冲区以处理较大的数据包

# 全局变量
clients = {}
agent_ready = {}
action_buffer = {}
lock = threading.Lock()

class Server:
    def __init__(self):
        # 训练参数
        self.num_episodes = 2000
        self.batch_size = 128
        self.buffer_size = 10000
        self.max_steps = 50
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.patience = 200
        self.min_episodes = 500
        self.best_reward = float('-inf')
        self.no_improve_count = 0
        
        # 初始化环境
        self.env = simple_world_comm_v3.env()
        self.env.reset()
        
        # 初始化经验池
        self.shared_replay_buffer = PrioritizedReplayBuffer(buffer_size=self.buffer_size)
        
        # 初始化智能体
        self.init_agents()
        
        # 初始化训练监控
        self.value_losses = []
        self.policy_losses = []
        self.entropies = []
        self.reward_history = {agent: [] for agent in self.env.possible_agents}
        
    def init_agents(self):
        self.agents = {}
        for agent in self.env.possible_agents:
            if agent.startswith('agent_'):
                obs_size = get_obs_size(self.env, agent)
                action_size = self.env.action_space(agent).n
                self.agents[agent] = MAPPOAgent(
                    obs_size=obs_size,
                    action_size=action_size,
                    gamma=self.gamma,
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    epsilon_min=0.05
                )
            else:
                self.agents[agent] = None
    
    def handle_client(self, conn, agent_id):
        global agent_ready, action_buffer
        print(f"连接已建立: {agent_id}")
        
        try:
            # 发送初始化信息
            agent = self.agents[agent_id]
            init_data = {
                'agent_id': agent_id,
                'obs_size': agent.obs_size,
                'action_size': agent.action_size,
                'actor_state_dict': {k: v.cpu().numpy().tolist() 
                                   for k, v in agent.actor.state_dict().items()},
                'critic_state_dict': {k: v.cpu().numpy().tolist() 
                                    for k, v in agent.critic.state_dict().items()},
            }
            conn.sendall(pickle.dumps(init_data))
            
            # 确认初始化完成
            with lock:
                agent_ready[agent_id] = True
            
            # 主通信循环
            while True:
                data = conn.recv(MAX_BUFFER)
                if not data:
                    break
                
                response = pickle.loads(data)
                cmd = response.get('cmd')
                
                if cmd == 'action':
                    # 接收客户端选择的动作
                    with lock:
                        action_buffer[agent_id] = response.get('action')
                        
                elif cmd == 'sync_done':
                    # 模型同步完成确认
                    with lock:
                        agent_ready[agent_id] = True
                        
                elif cmd == 'disconnect':
                    break
        
        except Exception as e:
            print(f"客户端处理错误 {agent_id}: {e}")
        finally:
            with lock:
                if agent_id in clients:
                    del clients[agent_id]
                if agent_id in agent_ready:
                    del agent_ready[agent_id]
                if agent_id in action_buffer:
                    del action_buffer[agent_id]
            conn.close()
            print(f"客户端连接已关闭: {agent_id}")

    def start_server(self):
        # 每个智能体启动一个服务器套接字
        server_sockets = {}
        agent_ids = [agent for agent in self.env.possible_agents 
                    if agent.startswith('agent_')]
        
        try:
            # 为每个智能体创建一个监听套接字
            for i, agent_id in enumerate(agent_ids):
                port = PORT_BASE + i
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((HOST, port))
                server_socket.listen(1)
                server_sockets[agent_id] = server_socket
                print(f"服务器为 {agent_id} 启动在端口 {port}")
            
            # 等待所有客户端连接
            for agent_id, server_socket in server_sockets.items():
                print(f"等待 {agent_id} 连接...")
                conn, addr = server_socket.accept()
                print(f"{agent_id} 已连接: {addr}")
                
                with lock:
                    clients[agent_id] = conn
                    agent_ready[agent_id] = False
                
                # 为每个客户端启动处理线程
                client_thread = threading.Thread(target=self.handle_client, 
                                               args=(conn, agent_id))
                client_thread.daemon = True
                client_thread.start()
            
            # 等待所有客户端准备就绪
            all_ready = False
            while not all_ready:
                with lock:
                    all_ready = all(agent_ready.get(agent_id, False) 
                                  for agent_id in agent_ids)
                if not all_ready:
                    time.sleep(0.1)
            
            print("所有客户端已连接并初始化，开始训练...")
            self.train()
            
        except Exception as e:
            print(f"服务器错误: {e}")
        finally:
            # 关闭所有连接
            for agent_id, conn in clients.items():
                try:
                    conn.close()
                except:
                    pass
            
            # 关闭所有服务器套接字
            for agent_id, server_socket in server_sockets.items():
                try:
                    server_socket.close()
                except:
                    pass

    def get_action_from_client(self, agent_id, observation):
        """向客户端发送观察并获取动作"""
        global action_buffer
        
        conn = clients.get(agent_id)
        if not conn:
            print(f"警告: {agent_id} 未连接")
            return self.env.action_space(agent_id).sample()
        
        try:
            # 向客户端发送观察
            data = {
                'cmd': 'get_action',
                'observation': observation
            }
            conn.sendall(pickle.dumps(data))
            
            # 等待客户端回应
            max_retries = 10
            for _ in range(max_retries):
                with lock:
                    if agent_id in action_buffer:
                        action = action_buffer[agent_id]
                        del action_buffer[agent_id]
                        return action
                time.sleep(0.01)
            
            # 如果客户端没有响应，则采取随机动作
            print(f"警告: {agent_id} 未在超时前响应，使用随机动作")
            return self.env.action_space(agent_id).sample()
            
        except Exception as e:
            print(f"获取动作时错误 {agent_id}: {e}")
            return self.env.action_space(agent_id).sample()

    def sync_models(self):
        """向所有客户端同步模型参数"""
        for agent_id, agent in self.agents.items():
            if not agent_id.startswith('agent_') or agent_id not in clients:
                continue
                
            conn = clients[agent_id]
            
            # 设置为未就绪状态
            with lock:
                agent_ready[agent_id] = False
            
            try:
                # 准备模型更新数据
                model_data = {
                    'cmd': 'sync_model',
                    'actor_state_dict': {k: v.cpu().numpy().tolist() 
                                       for k, v in agent.actor.state_dict().items()},
                    'critic_state_dict': {k: v.cpu().numpy().tolist() 
                                        for k, v in agent.critic.state_dict().items()},
                    'epsilon': agent.epsilon
                }
                conn.sendall(pickle.dumps(model_data))
                
            except Exception as e:
                print(f"同步模型时错误 {agent_id}: {e}")
        
        # 等待所有客户端确认同步完成
        all_ready = False
        timeout = 5.0  # 设置超时时间（秒）
        start_time = time.time()
        
        while not all_ready and (time.time() - start_time) < timeout:
            with lock:
                all_ready = all(agent_ready.get(agent_id, False) 
                              for agent_id in clients.keys()
                              if agent_id.startswith('agent_'))
            if not all_ready:
                time.sleep(0.1)
        
        if not all_ready:
            print("警告: 部分客户端未在超时前确认模型同步")

    def train(self):
        """主训练循环"""
        for episode in tqdm(range(self.num_episodes), desc="训练中"):
            self.env.reset()
            step = 0
            episode_rewards = {agent: 0 for agent in self.env.possible_agents}
            trajectories = {agent: [] for agent in self.env.possible_agents 
                           if agent.startswith('agent_')}
            all_agents_done = {agent: False for agent in self.env.possible_agents}
            
            # 同步模型到所有客户端
            self.sync_models()
            
            while step < self.max_steps and not all(all_agents_done.values()):
                step += 1
                
                for agent in self.env.agent_iter():
                    if all_agents_done[agent]:
                        continue
                        
                    observation, reward, termination, truncation, info = self.env.last()
                    done = termination or truncation
                    
                    if done:
                        action = None
                        all_agents_done[agent] = True
                    else:
                        if agent.startswith('agent_'):
                            # 从远程客户端获取动作
                            action = self.get_action_from_client(agent, observation)
                        else:
                            # 非玩家智能体使用随机动作
                            action = self.env.action_space(agent).sample()
                    
                    self.env.step(action)
                    
                    # 只为agent_类型的智能体收集经验
                    if agent.startswith('agent_'):
                        shaped_reward = shape_reward(reward, done, step, self.max_steps)
                        episode_rewards[agent] += shaped_reward
                        
                        if not done and action is not None:
                            trajectories[agent].append({
                                'state': observation,
                                'action': action,
                                'reward': shaped_reward,
                                'next_state': self.env.observe(agent) if not done else observation,
                                'done': done
                            })
                    else:
                        episode_rewards[agent] += reward
                
                # 每10步打印一次累积奖励
                if step % 10 == 0:
                    print(f"\n步骤 {step} 累积奖励:")
                    for agent, reward in episode_rewards.items():
                        if agent.startswith('agent_'):
                            print(f"  {agent}: {reward}")
                    print()
            
            # 处理收集的轨迹并存入经验池
            for agent_id, traj in trajectories.items():
                agent = self.agents[agent_id]
                
                if len(traj) > 0:
                    # 将轨迹数据转换为模型可用形式
                    states = [t['state'] for t in traj]
                    actions = [t['action'] for t in traj]
                    rewards = [t['reward'] for t in traj]
                    next_states = [t['next_state'] for t in traj]
                    dones = [float(t['done']) for t in traj]
                    
                    # 转换为张量
                    if isinstance(states[0], tuple):
                        states_np = np.array([np.concatenate([s.flatten() for s in state]) for state in states])
                        next_states_np = np.array([np.concatenate([s.flatten() for s in state]) for state in next_states])
                    else:
                        states_np = np.array([state.flatten() for state in states])
                        next_states_np = np.array([state.flatten() for state in next_states])
                    
                    states_tensor = torch.FloatTensor(states_np)
                    next_states_tensor = torch.FloatTensor(next_states_np)
                    
                    # 标准化状态
                    states_tensor = (states_tensor - states_tensor.mean()) / (states_tensor.std() + 1e-8)
                    next_states_tensor = (next_states_tensor - next_states_tensor.mean()) / (next_states_tensor.std() + 1e-8)
                    
                    with torch.no_grad():
                        # 计算当前Q值和目标Q值
                        current_q = agent.critic(states_tensor).squeeze()
                        target_q = agent.target_critic(next_states_tensor).squeeze()
                        
                        # 计算TD目标
                        rewards_tensor = torch.FloatTensor(rewards)
                        dones_tensor = torch.FloatTensor(dones)
                        td_targets = rewards_tensor + agent.gamma * target_q * (1 - dones_tensor)
                        
                        # 计算优势
                        advantages = td_targets - current_q
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # 将数据存入经验池
                    for t in range(len(traj)):
                        self.shared_replay_buffer.add((
                            states[t],
                            actions[t],
                            rewards[t],
                            next_states[t],
                            float(dones[t]),
                            current_q[t].item(),
                            td_targets[t].item(),
                            advantages[t].item()
                        ))
            
            # 将临时缓冲区的数据存入主缓冲区
            self.shared_replay_buffer.store_episode()
            
            # 策略更新阶段
            if len(self.shared_replay_buffer) >= self.batch_size:
                # 获取所有收集的数据
                all_data = self.shared_replay_buffer.get_all_data()
                # 随机打乱数据
                np.random.shuffle(all_data)
                # 执行多次更新
                n_updates = 4  # 标准PPO通常进行4-10次更新
                
                for agent_id, agent in self.agents.items():
                    if agent_id.startswith('agent_'):
                        for _ in range(n_updates):
                            try:
                                # 从缓冲区中随机采样
                                indices = np.random.randint(0, len(all_data), self.batch_size)
                                batch = [all_data[i] for i in indices]
                                # 处理数据
                                self.shared_replay_buffer.set_current_agent(agent)
                                states, actions, rewards, next_states, dones, _, _, _ = process_batch(batch, agent, self.shared_replay_buffer)
                                
                                # 检查数据形状
                                if len(states) == 0 or states.shape[0] != self.batch_size:
                                    continue
                                    
                                # 计算损失
                                value_loss, policy_loss, entropy = compute_loss(agent, states, actions, rewards, next_states, dones)
                                
                                # 检查损失是否为NaN
                                if torch.isnan(value_loss).any() or torch.isnan(policy_loss).any():
                                    print(f"警告: 损失为NaN，跳过 {agent_id} 的更新")
                                    continue
                                    
                                # 计算总损失
                                loss = value_loss * 0.5 + policy_loss - self.entropy_coef * entropy
                                
                                # 执行更新
                                agent.actor_optimizer.zero_grad()
                                agent.critic_optimizer.zero_grad()
                                loss.backward()
                                
                                # 梯度裁剪
                                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                                
                                agent.actor_optimizer.step()
                                agent.critic_optimizer.step()
                                
                                # 记录损失值
                                self.value_losses.append(value_loss.item())
                                self.policy_losses.append(policy_loss.item())
                                self.entropies.append(entropy.item())
                            except Exception as e:
                                print(f"更新 {agent_id} 时出错: {e}")
                                continue
                        
                        # 在所有更新结束后同步网络
                        agent.sync_networks()
            
            # 在episode结束时更新学习率
            if episode > 0 and episode % 100 == 0:  # 每100个episode更新一次
                for agent_id, agent in self.agents.items():
                    if agent_id.startswith('agent_'):
                        agent.actor_scheduler.step()
                        agent.critic_scheduler.step()
            
            # 输出统计信息
            if episode % 10 == 0 and self.value_losses:
                print(f"\nEpisode {episode} 统计信息:")
                print(f"平均值损失: {np.mean(self.value_losses):.4f}")
                print(f"平均策略损失: {np.mean(self.policy_losses):.4f}")
                print(f"平均熵: {np.mean(self.entropies):.4f}")
                print(f"缓冲区大小: {len(self.shared_replay_buffer)}")
                
                # 清空监控列表
                self.value_losses = []
                self.policy_losses = []
                self.entropies = []
            
            # 记录每个智能体的回合奖励
            for agent in self.env.possible_agents:
                self.reward_history[agent].append(episode_rewards[agent])
                
            print(f"回合 {episode + 1} 智能体奖励:", {
                k: v for k, v in episode_rewards.items() if k.startswith('agent_')
            })
            
            # 在episode结束时更新探索率
            for agent_id, agent in self.agents.items():
                if agent_id.startswith('agent_'):
                    agent.update_epsilon()
            
            # 计算当前episode的平均奖励
            current_reward = np.mean([reward for agent, reward in episode_rewards.items() 
                                     if agent.startswith('agent_')])
            
            # Early stopping检查
            if episode < self.min_episodes:
                self.no_improve_count = 0
            else:
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self.no_improve_count = 0
                    # 保存最佳模型
                    for agent_id, agent in self.agents.items():
                        if agent_id.startswith('agent_'):
                            save_model(agent, f'best_model_{agent_id}.pth')
                else:
                    self.no_improve_count += 1
            
            if self.no_improve_count >= self.patience and episode >= self.min_episodes:
                print(f"Early stopping at episode {episode}")
                break
            
            # 每50回合评估一次
            if episode % 50 == 0:
                avg_rewards = evaluate_agents(self.env, self.agents)
                print("\n评估结果:")
                for agent, reward in avg_rewards.items():
                    if agent.startswith('agent_'):
                        print(f"{agent}: {reward:.2f}")
        
        # 训练结束，通知客户端断开连接
        for agent_id, conn in clients.items():
            try:
                conn.sendall(pickle.dumps({'cmd': 'shutdown'}))
            except:
                pass

if __name__ == "__main__":
    server = Server()
    server.start_server() 