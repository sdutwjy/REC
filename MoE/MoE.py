import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from modules.crossvit import CrossAttentionBlock

class GateNetwork(nn.Module):
    def __init__(self, num_experts=4, init_type='random'):
        super().__init__()
        assert num_experts == 4, "num_experts must be 4"
        assert init_type in ["fixed", "learnable", "random"], "init_type must be either 'fixed', 'learnable' or 'random'"

        # Initialize gate weights for 4 experts
        init_weight = torch.Tensor([0.25, 0.25, 0.25, 0.25])  # Evenly distribute the initial weights

        # Set the gate parameter based on the initialization type
        if init_type == 'fixed':
            self.gate = nn.Parameter(init_weight, requires_grad=False)  # Fixed weights
        elif init_type == 'learnable':
            self.gate = nn.Parameter(init_weight)  # Learnable weights
        else:
            self.gate = nn.Parameter(torch.randn(num_experts))  # Random initialization
        
    def forward(self):
        return F.softmax(self.gate, dim=0)  # Ensure the weights sum to 1 using softmax
    
class TGateNetwork(nn.Module):
    def __init__(self, num_experts=4, init_type='random', temperature=4.0):
        super().__init__()
        assert num_experts == 4, "num_experts must be 4"
        assert init_type in ["fixed", "learnable", "random"], "init_type must be either 'fixed', 'learnable' or 'random'"

        # Initialize gate weights for 4 experts
        init_weight = torch.Tensor([0.25, 0.25, 0.25, 0.25])  # Evenly distribute the initial weights

        if init_type == 'fixed':
            self.gate = nn.Parameter(init_weight, requires_grad=False)  # Fixed weights
        elif init_type == 'learnable':
            self.gate = nn.Parameter(init_weight)  # Learnable weights
        else:
            # Xavier initialization for random initialization
            self.gate = nn.Parameter(torch.randn(num_experts) * math.sqrt(2. / num_experts))  # Xavier initialization
        
        self.temperature = temperature  # Temperature for scaling the gate weights

    def forward(self):
        # Apply temperature scaling to gate weights before softmax
        return F.softmax(self.gate / self.temperature, dim=0)  # Softmax with temperature
    
class topk_GateNetwork(nn.Module):
    def __init__(self, num_experts=4, init_type='random', top_k=1):
        super().__init__()
        assert num_experts == 4, "num_experts must be 4"
        assert init_type in ["fixed", "learnable", "random"], "init_type must be either 'fixed', 'learnable' or 'random'"

        # Initialize gate weights for 4 experts
        init_weight = torch.Tensor([0.25, 0.25, 0.25, 0.25])  # Evenly distribute the initial weights

        if init_type == 'fixed':
            self.gate = nn.Parameter(init_weight, requires_grad=False)  # Fixed weights
        elif init_type == 'learnable':
            self.gate = nn.Parameter(init_weight)  # Learnable weights
        else:
            # Xavier initialization for random initialization
            self.gate = nn.Parameter(torch.randn(num_experts) * math.sqrt(2. / num_experts))  # Xavier initialization

        self.top_k = top_k  # Number of top experts to select

    def forward(self):
        softmax_weights = F.softmax(self.gate, dim=0)
        
        # Select top-k experts based on the softmax weights
        top_k_values, top_k_indices = torch.topk(softmax_weights, self.top_k)

        # Zero-out all but the top-k experts
        sparse_weights = torch.zeros_like(softmax_weights)
        sparse_weights[top_k_indices] = top_k_values

        return sparse_weights

class share_GateNetwork(nn.Module):
    def __init__(self, num_experts=4, init_type='random'):
        super().__init__()
        assert num_experts == 4, "num_experts must be 4"
        assert init_type in ["fixed", "learnable", "random"], "init_type must be either 'fixed', 'learnable' or 'random'"

        init_weight = torch.Tensor([0.3, 0.3, 0.2, 0.2])

        if init_type == 'fixed':
            self.gate = nn.Parameter(init_weight, requires_grad=False)
        elif init_type == 'learnable':
            self.gate = nn.Parameter(init_weight)
        else:
            self.gate = nn.Parameter(torch.randn(num_experts))

        # Define a shared memory for information exchange between experts
        self.shared_memory = nn.Parameter(torch.randn(1, 1, 768))  # Example memory of size [1, 1, hidden_size]

    def forward(self, expert_outputs):
        # Apply gate mechanism
        gate_weights = F.softmax(self.gate, dim=0)

        # Ensure memory_contribution has the same batch size, num_experts, seq_len, hidden_size as expert_outputs
        batch_size, num_experts, seq_len, hidden_size = expert_outputs.shape
        
        # Expand shared memory to match the required dimensions
        memory_contribution = self.shared_memory.expand(batch_size, num_experts, seq_len, hidden_size)

        # Combine the shared memory contribution with the expert outputs
        combined_output = expert_outputs + memory_contribution  # Add memory to expert outputs

        # Adjust gate_weights to be compatible for broadcasting
        # gate_weights = gate_weights.view(1, num_experts, 1, 1)  # Shape [1, num_experts, 1, 1] for broadcasting

        # Ensure gate_weights has the shape [batch_size, num_experts, 1] for broadcasting
        # gate_weights = gate_weights.view(1, 4, 1)  # Shape: [1, 4, 1]
        # gate_weights = gate_weights.expand(combined_output.size(0), -1, 1)  # Expand to [batch_size, num_experts, 1]

        # Apply einsum with the adjusted gate_weights
        output = torch.einsum("bnik,n->bik", combined_output, gate_weights)  # Remove extra dimensions



        return output
    
class CrossAttention2(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Cross-Attention Module
        Args:
            embed_dim: The embedding dimension for both query and key-value pairs.
            num_heads: Number of attention heads.
            dropout: Dropout rate applied to the attention scores.
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for query, key, value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: Tensor of shape (batch_size, query_len, embed_dim)
            key: Tensor of shape (batch_size, key_len, embed_dim)
            value: Tensor of shape (batch_size, key_len, embed_dim)
            key_padding_mask: Optional mask for the key, shape (batch_size, key_len)
                              where 1 indicates padding positions.
        Returns:
            Tensor of shape (batch_size, query_len, embed_dim) with cross-attended outputs.
        """
        batch_size, query_len, embed_dim = query.size()
        _, key_len, _ = key.size()

        # Project query, key, value
        query = self.query_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5  # Scaled dot-product

        # Apply key padding mask (if provided)
        if key_padding_mask is not None:
            # Expand mask to (batch_size, num_heads, 1, key_len)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            key_padding_mask = torch.randint(0, 2, (batch_size, key_len)).to(device)
            attention_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(device)
            attention_scores = attention_scores.masked_fill(attention_mask == 1, float("-inf"))

        # Normalize scores with softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Compute weighted sum of value vectors
        attention_output = torch.matmul(attention_probs, value)  # Shape: (batch_size, num_heads, query_len, head_dim)

        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)
        attention_output = self.out_proj(attention_output)

        return attention_output


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.1, concat=True):
        """
        Graph Attention Network (GAT) Layer
        Args:
            in_features: Number of input features per node.
            out_features: Number of output features per node.
            num_heads: Number of attention heads.
            dropout: Dropout rate for attention weights.
            concat: Whether to concatenate the multi-head outputs (True) or average them (False).
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Linear layer for learning node embeddings (shared across heads)
        self.weight = nn.Parameter(torch.Tensor(in_features, num_heads * out_features))

        # Attention weights for each head
        self.attn_weight_src = nn.Parameter(torch.Tensor(num_heads, out_features, 1))
        self.attn_weight_dst = nn.Parameter(torch.Tensor(num_heads, out_features, 1))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_weight_src)
        nn.init.xavier_uniform_(self.attn_weight_dst)

    def forward(self, x, adj):
        """
        Forward pass for GAT Layer.
        Args:
            x: Node features, shape (batch_size, num_nodes, in_features).
            adj: Adjacency matrix, shape (batch_size, num_nodes, num_nodes).
        Returns:
            Updated node features, shape (batch_size, num_nodes, out_features).
        """
        batch_size, num_nodes, _ = x.size()

        # Step 1: Linear transformation of input features
        h = torch.matmul(x, self.weight)  # Shape: (batch_size, num_nodes, num_heads * out_features)
        h = h.view(batch_size, num_nodes, self.num_heads, self.out_features)  # Shape: (batch_size, num_nodes, num_heads, out_features)

        # Step 2: Compute attention scores
        src_attn = torch.matmul(h, self.attn_weight_src).squeeze(-1)  # Shape: (batch_size, num_nodes, num_heads)
        dst_attn = torch.matmul(h, self.attn_weight_dst).squeeze(-1)  # Shape: (batch_size, num_nodes, num_heads)

        # Broadcast to compute pairwise attention
        scores = src_attn.unsqueeze(2) + dst_attn.unsqueeze(1)  # Shape: (batch_size, num_nodes, num_nodes, num_heads)

        # Mask invalid edges (if adjacency matrix is used)
        if adj is not None:
            mask = (adj == 0).unsqueeze(-1)  # Shape: (batch_size, num_nodes, num_nodes, 1)
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply softmax to normalize attention scores
        attention = F.softmax(scores, dim=2)  # Shape: (batch_size, num_nodes, num_nodes, num_heads)
        attention = self.dropout(attention)  # Apply dropout

        # Step 3: Weighted aggregation of node features
        h_prime = torch.matmul(attention.transpose(2, 3), h)  # Shape: (batch_size, num_nodes, num_heads, out_features)

        # Step 4: Concatenate or average heads
        if self.concat:
            h_prime = h_prime.reshape(batch_size, num_nodes, self.num_heads * self.out_features)  # Concatenate heads
        else:
            h_prime = h_prime.mean(dim=2)  # Average heads

        return h_prime



# class MoE(nn.Module):
#     def __init__(self, num_experts, experts, gate_init_type='random', query_data="image"):
#         super().__init__()
#         assert query_data in ["image", "text"], "query_data must be either 'image' or 'text'"
        
#         self.num_experts = num_experts
#         self.query_data = query_data
#         self.experts = experts
#         self.gate = GateNetwork(num_experts, init_type=gate_init_type)

#     def forward(self, input_ids, query_embeds, attention_mask, encoder_hidden_states, encoder_attention_mask, return_dict=True):
#         gate = self.gate()  # Get gating weights and ensure they sum to 1 using softmax

#         # Collecting the outputs of experts
#         expert_outputs = []
        
#         # Expert 1
#         expert_outputs.append(self.experts(
#             input_ids=input_ids,
#             query_embeds=query_embeds,
#             attention_mask=attention_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             return_dict=return_dict
#         ).last_hidden_state[:, :query_embeds.size(1), :])

#         # Expert 2 (image or text-specific adjustments)
#         if self.num_experts >= 2 or (self.num_experts == 4 and self.query_data == "image"):
#             expert_outputs.append(self.experts(
#                 query_embeds=query_embeds,
#                 attention_mask=attention_mask[:, :query_embeds.size(1)],
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 return_dict=return_dict
#             ).last_hidden_state[:, :query_embeds.size(1), :])
        
#         # Expert 3 (image or text-specific adjustments)
#         if self.num_experts >= 3 or (self.num_experts == 4 and self.query_data == "text"):
#             expert_outputs.append(self.experts(
#                 input_ids=input_ids,
#                 query_embeds=query_embeds,
#                 attention_mask=attention_mask,
#                 return_dict=return_dict
#             ).last_hidden_state[:, :query_embeds.size(1), :])

#         # Expert 4 (can be any specialized expert)
#         if self.num_experts >= 4:
#             expert_outputs.append(self.experts(
#                 input_ids=input_ids,  # Adjust this if needed for expert 4
#                 query_embeds=query_embeds,
#                 attention_mask=attention_mask,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 return_dict=return_dict
#             ).last_hidden_state[:, :query_embeds.size(1), :])

#         # Stack and apply weighted sum for 4 experts
#         expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: [batch_size, num_experts, seq_len, hidden_size]
#         output = torch.einsum("bnik,n->bik", expert_outputs, gate)  # Shape: [batch_size, seq_len, hidden_size]

#         return output


# class GateNetwork(nn.Module):
#     def __init__(self, num_experts=1, init_type='random'):
#         super().__init__()
#         assert num_experts == 1, "num_experts 必须是 1"  # 确保是一个专家

#         # 仅为一个专家初始化一个门控参数
#         self.gate = nn.Parameter(torch.Tensor([1.0]))  # 只有一个门控权重

#     def forward(self):
#         return torch.sigmoid(self.gate)  # 使用 sigmoid 确保门控值在 0 和 1 之间

class EnhancedGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(EnhancedGatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 扩展隐藏层的维度
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_experts)  # 输出每个专家的权重
        
        self.softmax = nn.Softmax(dim=1)  # 确保权重总和为1
        self.dropout = nn.Dropout(0.3)  # 添加dropout防止过拟合
        
    def forward(self, input_features):
        # 增加更多的非线性层次
        x = F.relu(self.fc1(input_features))
        x = self.dropout(x)  # dropout
        x = F.relu(self.fc2(x))
        
        # 使用残差连接
        residual = x  # 残差
        x = F.relu(self.fc3(x))
        
        # 最终输出层
        weights = self.softmax(self.fc4(x))
        return weights

class ExpertFusionNetworkWithEnhancedGating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(ExpertFusionNetworkWithEnhancedGating, self).__init__()
        self.gating_network = EnhancedGatingNetwork(input_dim, num_experts)
        # self.expert_models = nn.ModuleList(expert_models)  # 专家模型列表
    
    def forward(self, input_features):
        # 通过增强的门控网络生成专家权重
        weights = self.gating_network(input_features)  # [batch_size, num_experts]
        
        
        # 对专家输出进行加权融合
        output = torch.einsum('bnik,n->bik', input_features, weights)  # 计算加权和
        return output



class MoE(nn.Module):
    def __init__(self, num_experts, experts, gate_init_type='random', query_data="image"):
        super().__init__()
        assert query_data in ["image", "text"], "query_data 必须是 'image' 或 'text'"

        self.num_experts = num_experts
        self.query_data = query_data
        self.experts = experts
        self.gate = GateNetwork(num_experts=4, init_type=gate_init_type)  # 仅为一个专家设置门控

    def forward(self, input_ids, query_embeds, attention_mask, encoder_hidden_states, encoder_attention_mask, return_dict=True):
        gate = self.gate()  # 获取门控值

        # 由于只有一个专家，直接获取该专家的输出
        expert_output = self.experts(
            input_ids=input_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict
        ).last_hidden_state[:, :query_embeds.size(1), :]  # 从单个专家获取输出

        # 使用门控对专家输出进行加权（在这个设置中，门控值始终为 1）
        output = expert_output * gate  # 将输出按门控值进行缩放（门控值为 1）

        return output


class FMoE(MoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crossattn = CrossAttentionBlock(768, 8, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, drop=0.1, drop_path=0.1)

    def forward(self, input_ids, query_embeds, attention_mask, encoder_hidden_states, encoder_attention_mask, return_dict=True):
        gate = self.gate()  # Get gating weights and ensure they sum to 1 using softmax

        # Collecting the outputs of experts
        expert_outputs = []
        expert_outputs.append(self.experts(
            input_ids=input_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict
        ).last_hidden_state[:, :query_embeds.size(1), :])

        if self.num_experts == 3 or (self.num_experts == 2 and self.query_data == "image"):
            expert_outputs.append(self.experts(
                query_embeds=query_embeds,
                attention_mask=attention_mask[:, :query_embeds.size(1)],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict
            ).last_hidden_state[:, :query_embeds.size(1), :])
        
        if self.num_experts == 3 or (self.num_experts == 2 and self.query_data == "text"):
            expert_outputs.append(self.experts(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                return_dict=return_dict
            ).last_hidden_state[:, :query_embeds.size(1), :])

        if self.num_experts == 2: 
            output = self.crossattn(expert_outputs[0] * gate[0], expert_outputs[1] * gate[1])
        else:
            output_1 = self.crossattn(expert_outputs[0] * gate[0], expert_outputs[1] * gate[1])
            output_2 = self.crossattn(expert_outputs[0] * gate[0], expert_outputs[2] * gate[2])
            output = self.crossattn(output_1, output_2)

        return output
