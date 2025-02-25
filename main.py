import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass
import os
import secrets

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ParallelAssociativeScan(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, us, vs, etas):
        batch_size, chunk_size, _ = us.shape
        
        etas_expanded = etas.unsqueeze(-1).expand(-1, -1, self.dim)
        
        results = []
        state = torch.zeros(batch_size, self.dim, device=us.device)
        
        for i in range(chunk_size):
            state = etas_expanded[:, i] * state + (us[:, i] - vs[:, i])
            results.append(state)
            
        return torch.stack(results, dim=1)

class ParallelizedLongTermMemory(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        chunk_size: int = 512,
        dropout: float = 0.0,
        init_scale: float = 0.02
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.init_scale = init_scale
        self.chunk_size = chunk_size
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.layers.append(nn.Linear(dim, dim))
            else:
                self.layers.append(nn.Linear(dim, dim))
            
        self.activation = SiLU()
        self.dropout = nn.Dropout(dropout)
        
        self.theta_proj = nn.Linear(dim, 1)
        self.eta_proj = nn.Linear(dim, 1)
        self.alpha_proj = nn.Linear(dim, 1)
        
        self.chunk_theta_proj = nn.Linear(dim, 1)
        self.chunk_eta_proj = nn.Linear(dim, 1)
        self.chunk_alpha_proj = nn.Linear(dim, 1)
        
        self.assoc_scan = ParallelAssociativeScan(dim)
        
        self.register_buffer('momentum', torch.zeros(1, dim))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_scale)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _compute_chunk_params(self, x_chunk):
        chunk_repr = x_chunk.mean(dim=1)
        alpha = torch.sigmoid(self.chunk_alpha_proj(chunk_repr)).unsqueeze(1)
        eta = torch.sigmoid(self.chunk_eta_proj(chunk_repr)).unsqueeze(1)
        theta = torch.sigmoid(self.chunk_theta_proj(chunk_repr)).unsqueeze(1) * 0.1
        return alpha, eta, theta
    
    def _tensorized_mbgd_update(self, x_chunk, layer_weights, beta_values, theta_values):
        batch_size, chunk_size, _ = x_chunk.shape
        
        Xmat = x_chunk.reshape(batch_size, chunk_size, self.dim)
        forward_output = F.linear(Xmat, layer_weights)
        residual = forward_output - Xmat
        
        X_transposed = x_chunk.transpose(1, 2)
        
        grad_matrix = torch.bmm(residual.transpose(1, 2), Xmat)
        
        theta_beta_matrix = (theta_values * beta_values).unsqueeze(-1).unsqueeze(-1)
        grad_update = theta_beta_matrix * grad_matrix
        
        return grad_update.sum(dim=0)
    
    def _parallel_momentum_update(self, gradients, etas):
        return self.assoc_scan(gradients, torch.zeros_like(gradients), etas)
    
    def forward(self, x, update_weights=True, use_chunk_params=True):
        batch_size, seq_len, _ = x.shape
        
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunks.append(x[:, i:end_idx])
        
        all_outputs = []
        
        for chunk_idx, x_chunk in enumerate(chunks):
            chunk_len = x_chunk.size(1)
            
            if update_weights and use_chunk_params:
                alpha, eta, theta = self._compute_chunk_params(x_chunk)
            else:
                token_alphas = torch.sigmoid(self.alpha_proj(x_chunk))
                token_etas = torch.sigmoid(self.eta_proj(x_chunk))
                token_thetas = torch.sigmoid(self.theta_proj(x_chunk)) * 0.1
                alpha = token_alphas.mean(dim=1, keepdim=True)
                eta = token_etas.mean(dim=1, keepdim=True)
                theta = token_thetas.mean(dim=1, keepdim=True)
            
            chunk_output = self._forward_step(x_chunk)
            all_outputs.append(chunk_output)
            
            if update_weights:
                key = x_chunk
                value = x_chunk
                
                pred_value = self._forward_step(key)
                residuals = value - pred_value
                
                beta_values = torch.ones_like(alpha)
                for prev_a in range(chunk_idx):
                    beta_values = beta_values * (1 - alpha)
                
                for j, layer in enumerate(self.layers):
                    if layer.weight.requires_grad:
                        weight_update = self._tensorized_mbgd_update(
                            key, layer.weight, beta_values, theta
                        )
                        
                        if j == 0:
                            momentum_update = self._parallel_momentum_update(
                                weight_update.unsqueeze(0).expand(batch_size, -1, -1),
                                eta.expand(batch_size, chunk_len)
                            )
                            self.momentum = momentum_update[:, -1].mean(dim=0, keepdim=True)
                        
                        layer.weight.data = (1 - alpha).mean() * layer.weight.data + self.momentum
                    
                    if layer.bias is not None and layer.bias.requires_grad:
                        residual_sum = residuals.sum(dim=1).mean(dim=0)
                        layer.bias.data = (1 - alpha).mean() * layer.bias.data - theta.mean() * residual_sum
        
        return torch.cat(all_outputs, dim=1)
    
    def _forward_step(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.depth - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
    
    def retrieve(self, query):
        return self._forward_step(query)
    
    def save_test_time_params(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'layers': [layer.state_dict() for layer in self.layers],
            'momentum': self.momentum
        }
        torch.save(state_dict, path)
    
    def load_test_time_params(self, path):
        if not os.path.exists(path):
            return False
        
        state_dict = torch.load(path)
        for i, layer_state in enumerate(state_dict['layers']):
            self.layers[i].load_state_dict(layer_state)
        self.momentum = state_dict['momentum']
        return True

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, 
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x.transpose(1, 2)

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_conv: bool = True,
        conv_kernel_size: int = 3
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.use_conv = use_conv
        if use_conv:
            self.q_conv = DepthwiseSeparableConv1d(dim, dim, conv_kernel_size, padding=conv_kernel_size//2)
            self.k_conv = DepthwiseSeparableConv1d(dim, dim, conv_kernel_size, padding=conv_kernel_size//2)
            self.v_conv = DepthwiseSeparableConv1d(dim, dim, conv_kernel_size, padding=conv_kernel_size//2)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.use_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.attn_scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        else:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            causal_mask = causal_mask.to(x.device)
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        
        return y

class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 512,
        dropout: float = 0.0,
        use_conv: bool = True,
        conv_kernel_size: int = 3
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.causal_attn = CausalSelfAttention(dim, num_heads, dropout, use_conv, conv_kernel_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        if seq_len <= self.window_size:
            return self.causal_attn(x)
        
        outputs = []
        for i in range(0, seq_len, self.window_size):
            end_idx = min(i + self.window_size, seq_len)
            window = x[:, i:end_idx, :]
            
            window_mask = torch.ones(end_idx - i, end_idx - i, device=x.device)
            window_mask = torch.tril(window_mask).unsqueeze(0).unsqueeze(0)
            
            output = self.causal_attn(window, window_mask)
            outputs.append(output)
        
        return torch.cat(outputs, dim=1)

class PersistentMemory(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 16):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, dim))
    
    def forward(self, x=None):
        batch_size = 1 if x is None else x.shape[0]
        return self.tokens.expand(batch_size, -1, -1)

class MemoryModule(nn.Module):
    def save_checkpoint(self, path):
        if hasattr(self, 'long_term_memory') and isinstance(self.long_term_memory, ParallelizedLongTermMemory):
            self.long_term_memory.save_test_time_params(path)
            return True
        return False
    
    def load_checkpoint(self, path):
        if hasattr(self, 'long_term_memory') and isinstance(self.long_term_memory, ParallelizedLongTermMemory):
            return self.long_term_memory.load_test_time_params(path)
        return False

class TitansMAC(MemoryModule):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        chunk_size: int = 512
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        self.persistent_memory = PersistentMemory(dim, persistent_tokens)
        self.long_term_memory = ParallelizedLongTermMemory(
            dim=dim, 
            depth=memory_depth, 
            chunk_size=chunk_size,
            dropout=dropout
        )
        
        self.layers = nn.ModuleList([
            CausalSelfAttention(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def _create_mac_attention_mask(self, chunk_size, persistent_len, memory_len):
        full_size = persistent_len + memory_len + chunk_size
        mask = torch.zeros(full_size, full_size)
        
        mask[:persistent_len, :persistent_len] = 1
        
        # Allow memory tokens to attend to persistent tokens and themselves
        if memory_len > 0:
            mask[persistent_len:persistent_len+memory_len, :persistent_len+memory_len] = 1
        
        # Allow chunk tokens to attend to all previous tokens
        start_idx = persistent_len + memory_len
        for i in range(chunk_size):
            mask[start_idx+i, :start_idx+i+1] = 1
        
        return mask
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        x = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        outputs = []
        memory_state = None
        
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            
            persistent_mem = self.persistent_memory()
            
            if i > 0 and memory_state is not None:
                query = chunk.mean(dim=1, keepdim=True)
                historical_mem = self.long_term_memory.retrieve(query)
                
                chunk_with_context = torch.cat([persistent_mem, historical_mem, chunk], dim=1)
                
                mac_mask = self._create_mac_attention_mask(
                    chunk.size(1), 
                    persistent_mem.size(1),
                    historical_mem.size(1)
                ).to(x.device)
            else:
                chunk_with_context = torch.cat([persistent_mem, chunk], dim=1)
                
                mac_mask = self._create_mac_attention_mask(
                    chunk.size(1), 
                    persistent_mem.size(1),
                    0
                ).to(x.device)
            
            # Apply attention blocks with residual connections
            attention_output = chunk_with_context
            for layer in self.layers:
                layer_output = layer(attention_output, mac_mask.unsqueeze(0).unsqueeze(0))
                attention_output = attention_output + layer_output
            
            attention_output = self.norm(attention_output)
            
            # Extract only the output corresponding to the chunk
            chunk_output = attention_output[:, -chunk.size(1):]
            outputs.append(chunk_output)
            
            # Update memory with chunk output
            if i + self.chunk_size < seq_len:
                memory_state = self.long_term_memory(chunk_output, update_weights=True)
        
        x = torch.cat(outputs, dim=1)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return logits, loss

class TitansMAG(MemoryModule):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        sliding_window: int = 512
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        self.persistent_memory = PersistentMemory(dim, persistent_tokens)
        self.long_term_memory = ParallelizedLongTermMemory(
            dim=dim, 
            depth=memory_depth, 
            chunk_size=sliding_window,
            dropout=dropout
        )
        
        self.swa = SlidingWindowAttention(dim, num_heads, sliding_window, dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def _create_mag_attention_mask(self, seq_len, persistent_len, window_size):
        total_len = persistent_len + seq_len
        
        sliding_window_mask = torch.zeros(total_len, total_len)
        
        # Allow persistent tokens to attend to each other
        sliding_window_mask[:persistent_len, :persistent_len] = 1
        
        # Apply sliding window attention pattern for sequence tokens
        for i in range(persistent_len, total_len):
            window_start = max(persistent_len, i - window_size + 1)
            sliding_window_mask[i, :persistent_len] = 1  # Always attend to persistent tokens
            sliding_window_mask[i, window_start:i+1] = 1  # Attend to window
            
        return sliding_window_mask
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        x = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        persistent_mem = self.persistent_memory()
        x_with_persistent = torch.cat([persistent_mem, x], dim=1)
        
        # Create sliding window mask for MAG variant
        mag_mask = self._create_mag_attention_mask(
            seq_len, 
            persistent_mem.size(1),
            self.swa.window_size
        ).to(x.device)
        
        # Apply sliding window attention with persistent memory
        swa_output = self.swa(x_with_persistent)[:, persistent_mem.size(1):]
        swa_output = self.norm1(swa_output)
        
        # Apply long-term memory module
        memory_output = self.long_term_memory(x)
        memory_output = self.norm2(memory_output)
        
        # Gate the two outputs
        gate_input = torch.cat([swa_output, memory_output], dim=-1)
        gate_value = self.gate(gate_input)
        
        x = gate_value * swa_output + (1 - gate_value) * memory_output
        
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return logits, loss

class TitansMAL(MemoryModule):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        sliding_window: int = 512
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        self.persistent_memory = PersistentMemory(dim, persistent_tokens)
        self.long_term_memory = ParallelizedLongTermMemory(
            dim=dim, 
            depth=memory_depth, 
            chunk_size=sliding_window,
            dropout=dropout
        )
        
        self.swa = SlidingWindowAttention(dim, num_heads, sliding_window, dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        x = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        persistent_mem = self.persistent_memory()
        x_with_persistent = torch.cat([persistent_mem, x], dim=1)
        
        # First pass through memory module
        memory_output = self.long_term_memory(x_with_persistent)[:, persistent_mem.size(1):]
        memory_output = self.norm1(memory_output)
        
        # Then pass through attention
        swa_output = self.swa(memory_output)
        swa_output = self.norm2(swa_output)
        
        logits = self.head(swa_output)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return logits, loss

class EfficientMemoryManager:
    def __init__(self, max_memory_tokens=2000000, chunk_size=10000, eviction_policy='lru'):
        self.max_memory_tokens = max_memory_tokens
        self.chunk_size = chunk_size
        self.eviction_policy = eviction_policy
        self.memory_chunks = {}
        self.access_times = {}
        self.importance_scores = {}
    
    def store(self, key, memory_chunk, importance=None):
        if len(self.memory_chunks) * self.chunk_size >= self.max_memory_tokens:
            self._evict_chunk()
        
        self.memory_chunks[key] = memory_chunk
        self.access_times[key] = 0
        if importance is not None:
            self.importance_scores[key] = importance
        else:
            self.importance_scores[key] = 1.0
    
    def retrieve(self, key):
        if key in self.memory_chunks:
            self.access_times[key] += 1
            return self.memory_chunks[key]
        return None
    
    def _evict_chunk(self):
        if self.eviction_policy == 'lru':
            # Least recently used
            key_to_evict = min(self.access_times, key=self.access_times.get)
        elif self.eviction_policy == 'lfu':
            # Least frequently used
            key_to_evict = min(self.access_times, key=lambda k: self.access_times[k])
        elif self.eviction_policy == 'importance':
            # Least important (weighted by recency and importance)
            scores = {k: self.importance_scores[k] * (1 + self.access_times[k]) for k in self.memory_chunks}
            key_to_evict = min(scores, key=scores.get)
        else:
            key_to_evict = secrets.choice(list(self.memory_chunks.keys()))
        
        del self.memory_chunks[key_to_evict]
        del self.access_times[key_to_evict]
        if key_to_evict in self.importance_scores:
            del self.importance_scores[key_to_evict]

class TitansModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        window_size: int = 512,
        enable_efficient_memory: bool = False,
        max_memory_tokens: int = 2000000
    ):
        super().__init__()
        self.model_type = model_type.lower()
        
        if self.model_type == 'mac':
            self.model = TitansMAC(
                vocab_size=vocab_size,
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                memory_depth=memory_depth,
                persistent_tokens=persistent_tokens,
                chunk_size=window_size
            )
        elif self.model_type == 'mag':
            self.model = TitansMAG(
                vocab_size=vocab_size,
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                memory_depth=memory_depth,
                persistent_tokens=persistent_tokens,
                sliding_window=window_size
            )
        elif self.model_type == 'mal':
            self.model = TitansMAL(
                vocab_size=vocab_size,
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                memory_depth=memory_depth,
                persistent_tokens=persistent_tokens,
                sliding_window=window_size
            )
        elif self.model_type == 'lmm':
            self.model = ParallelizedLongTermMemory(
                dim=dim,
                depth=memory_depth,
                chunk_size=window_size,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.enable_efficient_memory = enable_efficient_memory
        if enable_efficient_memory:
            self.memory_manager = EfficientMemoryManager(max_memory_tokens)
    
    def forward(self, x, targets=None):
        return self.model(x, targets)
    
    def save_test_time_state(self, path):
        if hasattr(self.model, 'save_checkpoint'):
            return self.model.save_checkpoint(path)
        return False
    
    def load_test_time_state(self, path):
        if hasattr(self.model, 'load_checkpoint'):
            return self.model.load_checkpoint(path)
        return False

def create_titans_model(
    model_type: str,
    vocab_size: int,
    dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    max_seq_len: int = 4096,
    dropout: float = 0.0,
    memory_depth: int = 2,
    persistent_tokens: int = 16,
    window_size: int = 512,
    enable_efficient_memory: bool = False,
    max_memory_tokens: int = 2000000
):
    return TitansModel(
        model_type=model_type,
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
        memory_depth=memory_depth,
        persistent_tokens=persistent_tokens,
        window_size=window_size,
        enable_efficient_memory=enable_efficient_memory,
        max_memory_tokens=max_memory_tokens
    )
