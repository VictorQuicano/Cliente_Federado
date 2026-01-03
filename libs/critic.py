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
    def __init__(self, state_dim=288, action_dim=128, hidden_dim=256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # # print(f"[DEBUG] Critic: state_dim={state_dim}, action_dim={action_dim}")
        
        # ====================
        # CONFIGURACIÓN CRÍTICA
        # ====================
        
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
        self.dropout = nn.Dropout(0.1)
        
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
        gain = 0.01  # ¡Muy pequeño!
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'layer4' in name or 'layer4_2' in name:  # Capa de salida
                    nn.init.uniform_(module.weight, -0.003, 0.003)
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
    
    def forward(self, state, action, return_min=False):
        """
        Forward con protección contra valores extremos.
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            return_min: Si True, retorna el mínimo de las dos redes (como TD3)
            
        Returns:
            q_value: [batch_size, 1] con valores acotados
        """
        # ====================
        # 1. VALIDACIÓN DE INPUTS
        # ====================
        
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
        
        # ====================
        # 2. CONCATENAR
        # ====================
        
        x = torch.cat([state, action], dim=-1)
        
        # Estadísticas para debugging
        state_stats = f"state: μ={state.mean():.3f}±{state.std():.3f}"
        action_stats = f"action: μ={action.mean():.3f}±{action.std():.3f}"
        x_stats = f"concat: μ={x.mean():.3f}±{x.std():.3f}"
        
        # ====================
        # 3. FORWARD A TRAVÉS DE LA(S) RED(ES)
        # ====================
        
        if self.use_double_q:
            q1 = self._forward_single_network(x, network_id=1)
            q2 = self._forward_single_network(x, network_id=2)
            
            # Usar el mínimo para reducir sobreestimación (como en TD3)
            if return_min:
                q_value = torch.min(q1, q2)
            else:
                q_value = q1  # O (q1 + q2) / 2 para promedio
        else:
            q_value = self._forward_single_network(x, network_id=1)
        
        # ====================
        # 4. LIMITAR VALORES Q (¡CRÍTICO!)
        # ====================
        
        # Clamping para evitar explosión
        q_value = torch.clamp(q_value, -10.0, 10.0)
        
        # ====================
        # 5. DEBUGGING
        # ====================
        
        if torch.isnan(q_value).any() or torch.isinf(q_value).any():
            print(f"[CRITIC ERROR] NaN/Inf in Q values!")
            print(f"  {state_stats}")
            print(f"  {action_stats}")
            print(f"  {x_stats}")
            print(f"  Q values before clamp: {q_value}")
            
            # Reemplazar valores inválidos
            q_value = torch.nan_to_num(q_value, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Logging periódico
        if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            print(f"[Critic Debug Step {self.debug_step}]")
            print(f"  {state_stats}")
            print(f"  {action_stats}")
            print(f"  Q range: [{q_value.min():.3f}, {q_value.max():.3f}]")
            print(f"  Q mean: {q_value.mean():.3f}")
        
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