import socket
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import argparse

# 配置参数解析
parser = argparse.ArgumentParser(description='MAPPO客户端')
parser.add_argument('--server', type=str, required=True, help='服务器IP地址')
parser.add_argument('--port', type=int, required=True, help='服务器端口号')
parser.add_argument('--agent-id', type=str, required=True, help='智能体ID')
args = parser.parse_args()

SERVER_IP = args.server
PORT = args.port
AGENT_ID = args.agent_id
MAX_BUFFER = 4096 * 16  # 增大缓冲区以处理较大的数据包

# 定义智能体网络模型
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

    def forward(self, x):
        return self.network(x)

class ClientAgent:
    def __init__(self, obs_size, action_size, agent_id):
        self.obs_size = obs_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.epsilon = 1.0
        self.gamma = 0.99
        
        # 初始化模型
        self.actor = Actor(obs_size, action_size)
        self.critic = Critic(obs_size)
    
    def load_state_dict(self, actor_state_dict, critic_state_dict):
        """从服务器加载模型参数"""
        # 转换列表为张量
        actor_dict = {}
        for key, value in actor_state_dict.items():
            actor_dict[key] = torch.tensor(value)
        
        critic_dict = {}
        for key, value in critic_state_dict.items():
            critic_dict[key] = torch.tensor(value)
        
        # 加载参数
        self.actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)
    
    def choose_action(self, observation):
        """基于当前策略选择动作"""
        if isinstance(observation, tuple):
            state = np.concatenate([obs.flatten() for obs in observation])
        else:
            state = observation.flatten()
        
        # 确保状态维度匹配
        state = torch.FloatTensor(state).unsqueeze(0)
        state = (state - state.mean()) / (state.std() + 1e-8)  # 标准化
        
        if np.random.random() < self.epsilon:
            # 随机探索
            if np.random.random() < 0.3:
                action = np.random.randint(self.action_size)
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

def connect_to_server():
    """连接到服务器"""
    print(f"连接到服务器 {SERVER_IP}:{PORT}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, PORT))
    return sock

def main():
    try:
        # 连接到服务器
        sock = connect_to_server()
        print("已连接到服务器，等待初始化...")
        
        # 接收初始化数据
        data = sock.recv(MAX_BUFFER)
        init_data = pickle.loads(data)
        
        agent_id = init_data['agent_id']
        obs_size = init_data['obs_size']
        action_size = init_data['action_size']
        
        print(f"初始化智能体 {agent_id}")
        print(f"观察空间大小: {obs_size}")
        print(f"动作空间大小: {action_size}")
        
        # 创建客户端智能体
        agent = ClientAgent(obs_size, action_size, agent_id)
        agent.load_state_dict(init_data['actor_state_dict'], init_data['critic_state_dict'])
        
        print("智能体已初始化，进入主循环")
        
        # 主循环
        running = True
        while running:
            try:
                # 接收服务器命令
                data = sock.recv(MAX_BUFFER)
                if not data:
                    print("服务器断开连接")
                    break
                
                server_msg = pickle.loads(data)
                cmd = server_msg.get('cmd')
                
                if cmd == 'get_action':
                    # 接收观察并返回动作
                    observation = server_msg.get('observation')
                    action = agent.choose_action(observation)
                    
                    # 发送动作回服务器
                    response = {
                        'cmd': 'action',
                        'action': action
                    }
                    sock.sendall(pickle.dumps(response))
                    
                elif cmd == 'sync_model':
                    # 更新模型参数
                    agent.load_state_dict(
                        server_msg.get('actor_state_dict'),
                        server_msg.get('critic_state_dict')
                    )
                    agent.epsilon = server_msg.get('epsilon', agent.epsilon)
                    
                    # 确认同步完成
                    response = {
                        'cmd': 'sync_done'
                    }
                    sock.sendall(pickle.dumps(response))
                    
                elif cmd == 'shutdown':
                    print("收到关闭指令")
                    running = False
                    break
                    
            except Exception as e:
                print(f"主循环错误: {e}")
                break
        
    except Exception as e:
        print(f"客户端错误: {e}")
    finally:
        print("客户端关闭")
        try:
            sock.close()
        except:
            pass

if __name__ == "__main__":
    main() 