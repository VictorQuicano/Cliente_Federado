import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 64
CONTEXT_DIM = 32
HIDDEN_DIM = 256

class ContextAwareCritic(nn.Module):
    """
    Crítico estabilizado para DDPG - con protección contra explosión de valores Q
    """
    def __init__(self, state_dim=288, action_dim=128, hidden_dim=256,
                 dropout: float = 0.1, init_gain: float = 0.01,
                 output_init_range: float = 0.003, q_clamp: float = 10.0):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout
        self.init_gain = init_gain
        self.output_init_range = output_init_range
        self.q_clamp = q_clamp
        
        # Network 1 (main)
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer4 = nn.Linear(hidden_dim // 2, 1)
        
        # Normalizaciones de capa (IMPORTANTE para estabilidad)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Network 2 (opcional, para Double DQN-style)
        self.use_double_q = True
        if self.use_double_q:
            self.layer1_2 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.layer2_2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3_2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.layer4_2 = nn.Linear(hidden_dim // 2, 1)
            
            self.ln1_2 = nn.LayerNorm(hidden_dim)
            self.ln2_2 = nn.LayerNorm(hidden_dim)
            self.ln3_2 = nn.LayerNorm(hidden_dim // 2)
        
        # Inicialización MUY conservadora (¡clave para evitar explosión!)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización extremadamente conservadora"""
        gain = self.init_gain
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'layer4' in name or 'layer4_2' in name:  # Capa de salida
                    nn.init.uniform_(module.weight, -self.output_init_range, self.output_init_range)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
    
    def _forward_single_network(self, x, network_id=1):
        """Forward para una red individual"""
        if network_id == 1:
            x = F.relu(self.ln1(self.layer1(x)))
            x = self.dropout(x)
            x = F.relu(self.ln2(self.layer2(x)))
            x = self.dropout(x)
            x = F.relu(self.ln3(self.layer3(x)))
            x = self.dropout(x)
            q_value = self.layer4(x)
        else:
            x = F.relu(self.ln1_2(self.layer1_2(x)))
            x = self.dropout(x)
            x = F.relu(self.ln2_2(self.layer2_2(x)))
            x = self.dropout(x)
            x = F.relu(self.ln3_2(self.layer3_2(x)))
            x = self.dropout(x)
            q_value = self.layer4_2(x)
        
        return q_value
    
    def forward(self, state, action, return_min=False, return_both=False):
        """
        Forward con protección contra valores extremos.
        - return_both=True: devuelve (q1, q2) para entrenar ambas redes (TD3 critic update)
        - return_min=True: devuelve min(q1, q2) para target estimation
        - default: devuelve q1
        """
        if state.shape[-1] != self.state_dim:
            print(f"[CRITIC ERROR] State dim mismatch: expected {self.state_dim}, "
                  f"got {state.shape[-1]}")

        if action.shape[-1] != self.action_dim:
            print(f"[CRITIC ERROR] Action dim mismatch: expected {self.action_dim}, "
                  f"got {action.shape[-1]}")

        # Detectar NaN/Inf
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("[CRITIC WARNING] NaN/Inf in state")
            state = torch.nan_to_num(state)

        if torch.isnan(action).any() or torch.isinf(action).any():
            print("[CRITIC WARNING] NaN/Inf in action")
            action = torch.nan_to_num(action)

        x = torch.cat([state, action], dim=-1)

        if self.use_double_q:
            q1 = self._forward_single_network(x, network_id=1)
            q2 = self._forward_single_network(x, network_id=2)

            q1 = torch.clamp(q1, -self.q_clamp, self.q_clamp)
            q2 = torch.clamp(q2, -self.q_clamp, self.q_clamp)

            if torch.isnan(q1).any() or torch.isnan(q2).any():
                print(f"[CRITIC ERROR] NaN/Inf in Q values! "
                      f"state: μ={state.mean():.3f}, action: μ={action.mean():.3f}")
                q1 = torch.nan_to_num(q1, nan=0.0, posinf=self.q_clamp, neginf=-self.q_clamp)
                q2 = torch.nan_to_num(q2, nan=0.0, posinf=self.q_clamp, neginf=-self.q_clamp)

            if return_both:
                return q1, q2
            if return_min:
                return torch.min(q1, q2)
            return q1
        else:
            q_value = self._forward_single_network(x, network_id=1)
            q_value = torch.clamp(q_value, -self.q_clamp, self.q_clamp)
            if torch.isnan(q_value).any():
                q_value = torch.nan_to_num(q_value, nan=0.0, posinf=self.q_clamp, neginf=-self.q_clamp)
            if return_both:
                return q_value, q_value
            return q_value
    
    def get_gradient_info(self):
        """Información sobre gradientes para debugging"""
        info = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                info[f"{name}_grad_norm"] = grad_norm
                info[f"{name}_grad_mean"] = grad_mean
                info[f"{name}_grad_std"] = grad_std
                
                if torch.isnan(param.grad).any():
                    info[f"{name}_has_nan"] = True
                if torch.max(torch.abs(param.grad)) > 100:
                    info[f"{name}_grad_exploding"] = True
        
        return info
    
    def clip_gradients(self, max_norm=1.0):
        """Clipping de gradientes (llamar después de backward, antes de optimizer.step())"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            max_norm=max_norm,
            norm_type=2.0
        )
        return total_norm.item()