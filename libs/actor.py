import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random
import bisect

# ==================== CONFIGURACIÓN ====================
EMBEDDING_DIM = 64
HISTORY_LENGTH = 10  # Últimos 10 ítems del historial
CONTEXT_DIM = 32  # Dimensión para embeddings de contexto
HIDDEN_DIM = 256

# ==================== MODELO ACTOR CON CONTEXTO POR ÍTEM ====================

class ContextAwareActor(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(ContextAwareActor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # ====================
        # 1. CONTEXT ENCODERS
        # ====================
        self.day_embed = nn.Embedding(7, 8)
        self.month_embed = nn.Embedding(12, 8)
        self.workday_embed = nn.Embedding(2, 8)
        
        # Codificación de hora (8 dimensiones: 4 seno, 4 coseno)
        self.hour_freq = 2 * torch.pi / 24.0
        
        # ====================
        # 2. ITEM PROCESSOR - CORREGIDO
        # ====================
        # ¡IMPORTANTE! embedding_dim + 32 (8*4 embeddings de contexto)
        self.item_processor_input_dim = embedding_dim + 32
        
        self.item_processor = nn.Sequential(
            nn.Linear(self.item_processor_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ====================
        # 3. ATTENTION (opcional)
        # ====================
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
                dropout=0.1
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # ====================
        # 4. DECODER - CORREGIDO
        # ====================
        # El estado combinado es: hidden_dim (historial) + 32 (último contexto)
        self.decoder_input_dim = hidden_dim + 32
        
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, embedding_dim),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización conservadora"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
    
    def encode_hour(self, hour):
        """Codificación senoidal de la hora"""
        # hour: [batch_size] o [batch_size, seq_len]
        hour_normalized = hour.float() / 23.0
        
        # Crear dimensiones para posicional encoding
        angles = self.hour_freq * hour_normalized.unsqueeze(-1) * torch.arange(4).float().to(hour.device)
        
        # Seno y coseno (4 dimensiones cada uno = 8 total)
        hour_sin = torch.sin(angles)
        hour_cos = torch.cos(angles)
        
        hour_encoding = torch.cat([hour_sin, hour_cos], dim=-1)
        return hour_encoding  # [*, 8]
    
    def encode_context(self, context_dict, prefix=""):
        """Codificar contexto con validación"""
        # Obtener dimensiones del batch
        if prefix + 'day_of_week' in context_dict:
            day_tensor = context_dict[prefix + 'day_of_week']
            batch_dims = day_tensor.shape
        else:
            # Buscar sin prefix si no se encuentra
            day_tensor = context_dict['day_of_week']
            batch_dims = day_tensor.shape
        
        # Obtener índices (con manejo de errores)
        try:
            day_idx = context_dict.get(prefix + 'day_of_week', context_dict['day_of_week']).long()
            month_idx = context_dict.get(prefix + 'month', context_dict['month']).long() - 1
            workday_idx = context_dict.get(prefix + 'is_workday', context_dict['is_workday']).long()
            hour = context_dict.get(prefix + 'hour_of_day', context_dict['hour_of_day']).float()
        except KeyError as e:
            print(f"[ERROR] Missing key in context_dict: {e}")
            print(f"Available keys: {list(context_dict.keys())}")
            raise
        
        # Validar rangos
        day_idx = torch.clamp(day_idx, 0, 6)
        month_idx = torch.clamp(month_idx, 0, 11)
        workday_idx = torch.clamp(workday_idx, 0, 1)
        
        # Aplanar para embeddings
        if len(batch_dims) == 1:
            flat_shape = (-1,)
        else:
            flat_shape = (-1, batch_dims[-1])
        
        # Aplicar embeddings
        day_emb = self.day_embed(day_idx.view(-1)).view(*batch_dims, -1)
        month_emb = self.month_embed(month_idx.view(-1)).view(*batch_dims, -1)
        workday_emb = self.workday_embed(workday_idx.view(-1)).view(*batch_dims, -1)
        hour_emb = self.encode_hour(hour).view(*batch_dims, -1)
        
        # Combinar (8 + 8 + 8 + 8 = 32 dimensiones)
        context_rep = torch.cat([day_emb, month_emb, workday_emb, hour_emb], dim=-1)
        
        # Debugging
        if context_rep.shape[-1] != 32:
            print(f"[WARNING] context_rep dim = {context_rep.shape[-1]}, expected 32")
        
        return context_rep
    
    def process_items(self, item_embeddings, item_contexts):
        """Procesar ítems con validación de dimensiones"""
        batch_size, n_items, embed_dim = item_embeddings.shape
        
        # 1. Validar dimensión de embeddings
        if embed_dim != self.embedding_dim:
            print(f"[ERROR] Item embedding dim mismatch: expected {self.embedding_dim}, got {embed_dim}")
            # Intentar ajustar si es posible
            if embed_dim < self.embedding_dim:
                # Padding
                padding = torch.zeros(batch_size, n_items, self.embedding_dim - embed_dim).to(item_embeddings.device)
                item_embeddings = torch.cat([item_embeddings, padding], dim=-1)
            else:
                # Truncar
                item_embeddings = item_embeddings[:, :, :self.embedding_dim]
        
        # 2. Codificar contexto de cada ítem
        item_context_rep = self.encode_context(item_contexts)
        
        # 3. Validar dimensiones antes de concatenar
        if item_context_rep.shape[-1] != 32:
            print(f"[ERROR] item_context_rep dim = {item_context_rep.shape[-1]}, expected 32")
        
        # 4. Concatenar
        combined = torch.cat([item_embeddings, item_context_rep], dim=-1)
        
        # 5. Validar dimensión de entrada al processor
        if combined.shape[-1] != self.item_processor_input_dim:
            print(f"[ERROR] Combined input dim = {combined.shape[-1]}, "
                  f"expected {self.item_processor_input_dim}")
            print(f"  item_embeddings shape: {item_embeddings.shape}")
            print(f"  item_context_rep shape: {item_context_rep.shape}")
            return None
        
        # 6. Procesar
        item_features = self.item_processor(combined)
        
        # 7. Atención opcional
        if self.use_attention:
            attn_output, _ = self.attention(item_features, item_features, item_features)
            item_features = self.attn_norm(item_features + self.dropout(attn_output))
        
        # 8. Pooling
        history_rep = item_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        return history_rep
    
    def forward(self, item_embeddings, item_contexts, last_context):
        """Forward con validación robusta"""
        # ====================
        # 0. VALIDAR INPUTS
        # ====================
        # print(f"[DEBUG forward] item_embeddings shape: {item_embeddings.shape}")
        # print(f"[DEBUG forward] item_contexts keys: {list(item_contexts.keys())}")
        # print(f"[DEBUG forward] last_context keys: {list(last_context.keys())}")
        
        # ====================
        # 1. PROCESAR ÍTEMS
        # ====================
        history_rep = self.process_items(item_embeddings, item_contexts)
        
        if history_rep is None:
            print("[ERROR] Failed to process items")
            # Crear un tensor de fallback
            batch_size = item_embeddings.shape[0]
            history_rep = torch.zeros(batch_size, self.hidden_dim).to(item_embeddings.device)
        
        # ====================
        # 2. CODIFICAR ÚLTIMO CONTEXTO
        # ====================
        last_context_rep = self.encode_context(last_context)
        
        # Si tiene dimensión extra (seq_len=1), removerla
        if last_context_rep.dim() == 3 and last_context_rep.size(1) == 1:
            last_context_rep = last_context_rep.squeeze(1)
        
        # ====================
        # 3. VALIDAR DIMENSIONES
        # ====================
        # print(f"[DEBUG forward] history_rep shape: {history_rep.shape}")
        # print(f"[DEBUG forward] last_context_rep shape: {last_context_rep.shape}")
        
        if history_rep.shape[-1] != self.hidden_dim:
            print(f"[ERROR] history_rep dim: {history_rep.shape[-1]}, expected {self.hidden_dim}")
        
        if last_context_rep.shape[-1] != 32:
            print(f"[ERROR] last_context_rep dim: {last_context_rep.shape[-1]}, expected 32")
        
        # ====================
        # 4. COMBINAR
        # ====================
        combined_state = torch.cat([history_rep, last_context_rep], dim=-1)
        
        # print(f"[DEBUG forward] combined_state shape: {combined_state.shape}")
        # print(f"[DEBUG forward] decoder_input_dim: {self.decoder_input_dim}")
        
        if combined_state.shape[-1] != self.decoder_input_dim:
            print(f"[ERROR] State dim mismatch: {combined_state.shape[-1]}, "
                  f"expected {self.decoder_input_dim}")
            # Ajustar dimensión si es necesario
            if combined_state.shape[-1] > self.decoder_input_dim:
                combined_state = combined_state[:, :self.decoder_input_dim]
            else:
                padding = torch.zeros(combined_state.shape[0], 
                                    self.decoder_input_dim - combined_state.shape[-1]
                                    ).to(combined_state.device)
                combined_state = torch.cat([combined_state, padding], dim=-1)
        
        # ====================
        # 5. DECODIFICAR
        # ====================
        next_item_embedding = self.decoder(combined_state)
        
        # print(f"[DEBUG forward] next_item_embedding shape: {next_item_embedding.shape}")
        
        return combined_state, next_item_embedding
    
    def forward_from_state(self, state):
        """Versión simple para DDPG"""
        # Validar estado
        if state.shape[-1] != self.decoder_input_dim:
            print(f"[WARNING] State dim in forward_from_state: {state.shape[-1]}, "
                  f"expected {self.decoder_input_dim}")
            # Ajustar si es necesario
            if state.shape[-1] > self.decoder_input_dim:
                state = state[:, :self.decoder_input_dim]
            else:
                padding = torch.zeros(state.shape[0], 
                                    self.decoder_input_dim - state.shape[-1]
                                    ).to(state.device)
                state = torch.cat([state, padding], dim=-1)
        
        return self.decoder(state)