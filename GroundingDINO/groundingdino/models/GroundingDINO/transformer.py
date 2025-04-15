# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional

import logging
import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torch.nn import functional as F
from groundingdino.util.misc import inverse_sigmoid
from MoE.MoE import MoE, GateNetwork, TGateNetwork, topk_GateNetwork, share_GateNetwork, CrossAttention2, GATLayer, ExpertFusionNetworkWithEnhancedGating
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)

from dmix.dmix import HybridTokenMixer

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        learnable_tgt_init=False,
        # two stage
        two_stage_type="no",  
        embed_init_tgt=False,
        # for text
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
        # experts=None,  # 这里传入专家模块
        num_experts=4,  # 设置专家数量为4
        gate_init_type="random",  # 门控初始化方式
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4
        
        # self.moe = MoE(num_experts=num_experts, experts=self.encoder, gate_init_type=gate_init_type)

        # choose encoder layer type
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )

        if use_text_enhancer: # True
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer: # True
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint, # True
            use_transformer_ckpt=use_transformer_ckpt, # True
        )
        self.qformer = Blip2QformerCirAlignPrompt()
        # self.moe = MoE(num_experts=num_experts, experts=self.qformer, gate_init_type=gate_init_type)

        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0: # 6
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model)) 
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt # True
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"): 
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard": 
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        
        self.dmix = HybridTokenMixer(
            dim=256,
            kernel_size=3,
            num_groups=2,
            num_heads=1,
            sr_ratio=1,
        )

        self.cross_attention = CrossAttentionLayer(d_model)
        self.num_experts = num_experts
        self.gate = GateNetwork(num_experts=4, init_type="random")
        # self.gate = TGateNetwork(num_experts=4, init_type="random")
        # self.gate = topk_GateNetwork(num_experts=4, init_type="random")
        # self.gate = share_GateNetwork(num_experts=4, init_type="random")
        # self.ehancegate = ExpertFusionNetworkWithEnhancedGating(input_dim=768, num_experts=4)
        self.crossattention = CrossAttention2(256, 8)
        # self.graph_attention = GATLayer()
        
        print(f"Total added parameters for cross attention: {sum(p.numel() for p in self.cross_attention.parameters())}")

        self.expert_pool = ExpertPool(
            num_experts=4,
            input_dim=768,
            hidden_dim=512,
            output_dim=256
        )
        self.feature_interaction = FeatureInteraction(256)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, text_dict=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi] # list of 4: [(bs, 256,100,137) (bs,256,50,69), (bs, 256,25,35), (bs, 256,13,18)]
            - masks: List of multi masks [bs, hi, wi]      # list of 4: [(bs,100,137), (bs,50,69), (bs,25,35), (bs,13,18)]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = [] 
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # print(f"Level {lvl}: src.shape = {mask.shape}")

            src = src.flatten(2).transpose(1, 2)  
            mask = mask.flatten(1)  
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  
            if self.num_feature_levels > 1 and self.level_embed is not None: 
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            
        # if src_flatten[0].shape[0] != src_flatten[1].shape[0]:
        #     max_bs = max([src.shape[0] for src in src_flatten])
        #     max_bs_mask = max([mask.shape[0] for mask in mask_flatten])
        #     max_bs_masks = max([masks[0].shape[0] for masks in masks])
            

        #     # Step 2: 填充src_flatten中每个元素的batch size，直到它们的batch size一致
        #     src_flatten_filled = []
        #     mask_flatten_filled = []
        #     masks_filled = []
            
        #     for masks_p in masks:
        #         bs, h, w = masks_p.shape
        #         if bs < max_bs_masks:
        #             # 计算需要填充的尺寸
        #             padding = (0, 0, 0, 0, 0, max_bs_masks - bs)  # batch维度填充
        #             masks_p = F.pad(masks_p, padding, value=0)  # 填充batch维度
        #         masks_filled.append(masks_p)

        #     for src in src_flatten:
        #         bs, c, hw = src.shape
        #         if bs < max_bs:
        #             # 计算需要填充的尺寸
        #             padding = (0, 0, 0, 0, 0, max_bs - bs)  # batch维度填充
        #             src = F.pad(src, padding, value=0)  # 填充batch维度
        #         src_flatten_filled.append(src)
                
        #     for mask in mask_flatten:
        #         bs, wh = mask.shape
        #         if bs < max_bs_mask:
        #             # 计算需要填充的尺寸
        #             padding = (0, 0, 0, max_bs_mask - bs)  # batch维度填充
        #             mask = F.pad(mask, padding, value=0)  # 填充batch维度
        #         mask_flatten_filled.append(mask)
        #     src_flatten = src_flatten_filled
        #     mask_flatten = mask_flatten_filled
        #     maskss = masks_filled
        # mask_p = mask_flatten_filled

        # Step 3: 将填充后的src_flatten元素在第一维度拼接
        src_flatten = torch.cat(src_flatten, 1)  # 在第1维度拼接，确保batch size一致
        mask_flatten = torch.cat(mask_flatten, 1)  # [6, 20906]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [6, 20906, 256]
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )  # [4, 2]
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )  # [4]

        # if masks[0].shape[0] != masks[1].shape[0]:
        #     valid_ratios = torch.stack([self.get_valid_ratio(m) for m in maskss], 1)   # [6, 4, 2]
        # else:
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)   # [6, 4, 2]

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None
        memory = src_flatten   # [6, 20906, 256]
        memory_text = text_dict["encoded_text"]  # [6, 5, 256]
        # """ encoder
        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder( 
            src_flatten,    # [6, 20906, 256]
            pos=lvl_pos_embed_flatten,   # [6, 20906, 256]
            level_start_index=level_start_index,  # [4]
            spatial_shapes=spatial_shapes,  # [4, 2]
            valid_ratios=valid_ratios,  # [6, 4, 2]
            key_padding_mask=mask_flatten,   #[6, 20906]
            memory_text=text_dict["encoded_text"],     # [6, 5, 256]
            text_attention_mask=~text_dict["text_token_mask"],  # [6, 5]
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict["position_ids"],   # # [6, 5]
            text_self_attention_masks=text_dict["text_self_attention_masks"],   # [6, 5, 5]
        )   # [6, 20906, 256]     [6, 5, 256]

        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        # """
        text_dict["encoded_text"] = memory_text   # [6, 5, 256]
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if memory.isnan().any() | memory.isinf().any():
        #         import ipdb; ipdb.set_trace()

        txt_embs = text_dict["encoded_text"]  # [6, 5, 256]

        if self.two_stage_type == "standard": 
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )   # [6, 20906, 256]    [6, 20906, 4]
            output_memory = self.enc_output_norm(self.enc_output(output_memory))  # [6, 20906, 256]

            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)   # [6, 20906, 256]
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

            topk_logits = enc_outputs_class_unselected[:, :, 0]  # [6, 20906]

            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # [6, 20906, 4]
            topk = self.num_queries    # 900

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # [6, 900]

            lower_idxes, higher_idxes = split_tokens(topk_proposals)    #  [6, 810]    [6, 90]
            lower_tokens = torch.gather(output_memory, 1, lower_idxes.unsqueeze(-1).expand(-1, -1, self.d_model))  # [6, 810, 256]
            higher_tokens = torch.gather(output_memory, 1, higher_idxes.unsqueeze(-1).expand(-1, -1, self.d_model))   # [6, 90, 256]

            text_subject_mask = text_dict["text_subject_mask"]   # [6, 5]
            text_context_mask = text_dict["text_context_mask"]   # # [6, 5]

            # Extracting the tokens using the mask
            subject_text_tokens = [text[mask] for text, mask in zip(text_dict["encoded_text"], text_subject_mask)] 
            context_text_tokens = [text[mask] for text, mask in zip(text_dict["encoded_text"], text_context_mask)]
            max_size = text_dict["encoded_text"].size(1)   # 5
            padded_subject_text_tokens = torch.stack([F.pad(t, (0, 0, 0, max_size - t.size(0))) for t in subject_text_tokens])  # [6, 5, 256]
            padded_context_text_tokens = torch.stack([F.pad(t, (0, 0, 0, max_size - t.size(0))) for t in context_text_tokens])   # [6, 5, 256]
            
            subject_mask = torch.stack([torch.cat([torch.ones(t.size(0)).to(t.device), torch.zeros(max_size - t.size(0)).to(t.device)]) for t in subject_text_tokens])  # [6, 5]
            context_mask = torch.stack([torch.cat([torch.ones(t.size(0)).to(t.device), torch.zeros(max_size - t.size(0)).to(t.device)]) for t in context_text_tokens])  # [6, 5]

            # lower_tokens = self.cross_attention(lower_tokens, padded_subject_text_tokens, V_mask=subject_mask)
            # if context_mask.sum() > 2: # not all [CLS][SEP] tokens
            #     higher_tokens = self.cross_attention(higher_tokens, padded_context_text_tokens, V_mask=context_mask) 

            # updated_lower_tokens = self.cross_attention(lower_tokens, higher_tokens)
            # output_memory = output_memory.scatter(1, lower_idxes.unsqueeze(-1).expand(-1, -1, 256), updated_lower_tokens)
            
            # moe_attri_output = self.moe(input_ids=None, 
            #                   query_embeds=padded_context_text_tokens, 
            #                   attention_mask=context_mask, 
            #                   encoder_hidden_states=lower_tokens, 
            #                   encoder_attention_mask=mask_flatten, 
            #                   return_dict=True)
        
            # moe_global_output = self.moe( input_ids=None, 
            #                   query_embeds=padded_context_text_tokens, 
            #                   attention_mask=context_mask, 
            #                   encoder_hidden_states=higher_tokens, 
            #                   encoder_attention_mask=mask_flatten, 
            #                   return_dict=True)
        
            # moe_spatial_output = self.moe(input_ids=None, 
            #                   query_embeds=lower_tokens, 
            #                   attention_mask=attn_mask, 
            #                   encoder_hidden_states=higher_tokens, 
            #                   encoder_attention_mask=mask_flatten, 
            #                   return_dict=True)
        
            # moe_local_output = self.moe( input_ids=None, 
            #                   query_embeds=padded_subject_text_tokens, 
            #                   attention_mask=subject_mask, 
            #                   encoder_hidden_states=lower_tokens, 
            #                   encoder_attention_mask=mask_flatten, 
            #                   return_dict=True)
            
            # target_dim = moe_attri_output.size(1)
            # moe_spatial_output = moe_spatial_output[:, :target_dim, :]
            
            gate = self.gate()
            # Collecting the outputs of experts
            expert_outputs = self.qformer(
                visual_features=lower_tokens,
                text_features=padded_context_text_tokens
            )

            # Expert 2 (image or text-specific adjustments)
            if self.num_experts >= 2 :
                expert_outputs2 = self.qformer(
                    visual_features=higher_tokens,
                    text_features=padded_context_text_tokens
                )
            
            # Expert 3 (image or text-specific adjustments)
            if self.num_experts >= 3 :
                expert_outputs3 = self.qformer(
                    visual_features=lower_tokens,
                    text_features=higher_tokens,
                )

            # Expert 4 (can be any specialized expert)
            if self.num_experts >= 4:
                expert_outputs4 = self.qformer(
                    visual_features=lower_tokens,
                    text_features=padded_subject_text_tokens,
                )
                
            # 确保所有输出都是3D张量 [batch, seq_len, dim]
            expert_outputs_list = [expert_outputs]
            if self.num_experts >= 2:
                expert_outputs_list.append(expert_outputs2)
            if self.num_experts >= 3:
                expert_outputs_list.append(expert_outputs3)
            if self.num_experts >= 4:
                expert_outputs_list.append(expert_outputs4)
                
            # 安全检查维度并适配
            for i in range(len(expert_outputs_list)):
                # 如果输出只有2D，扩展为3D
                if len(expert_outputs_list[i].shape) == 2:
                    expert_outputs_list[i] = expert_outputs_list[i].unsqueeze(1)
            
            # 获取最大序列长度和特征维度
            max_seq_len = max(output.size(1) for output in expert_outputs_list)
            feature_dims = [output.size(2) for output in expert_outputs_list]
            target_dim = max(feature_dims)  # 使用最大特征维度

            # 填充序列长度和特征维度
            padded_expert_outputs = []
            for expert_output in expert_outputs_list:
                # 获取当前输出的维度
                batch_size, seq_len, feat_dim = expert_output.shape
                
                # 计算需要填充的长度
                pad_seq = max_seq_len - seq_len
                pad_feat = target_dim - feat_dim
                
                # 只有在需要填充时才应用填充
                if pad_seq > 0 or pad_feat > 0:
                    padded_output = F.pad(expert_output, (0, pad_feat, 0, pad_seq, 0, 0))
                else:
                    padded_output = expert_output
                    
                padded_expert_outputs.append(padded_output)

            # 堆叠所有专家输出
            expert_outputs = torch.stack(padded_expert_outputs, dim=1)  # Shape: [batch_size, num_experts, max_seq_len, feat_dim]
            
            # 应用特征交互
            batch_size, num_experts, seq_len, feat_dim = expert_outputs.shape
            expert_outputs_reshaped = expert_outputs.view(batch_size * num_experts, seq_len, feat_dim)
            
            # 应用特征交互
            processed_outputs = self.feature_interaction(expert_outputs_reshaped)
            
            # 重新整形回原始形状
            processed_outputs = processed_outputs.view(batch_size, num_experts, seq_len, feat_dim)
            
            # 合并所有专家的输出（使用门控网络或简单平均）
            try:
                # 尝试使用门控权重
                gate_weights = self.gate().view(-1, 1, 1)  # [num_experts, 1, 1]
                output_memory = torch.sum(processed_outputs * gate_weights.unsqueeze(0), dim=1)  # [batch_size, seq_len, feat_dim]
            except (RuntimeError, ValueError) as e:
                # 回退到简单平均
                output_memory = torch.mean(processed_outputs, dim=1)  # [batch_size, seq_len, feat_dim]

            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # [6, 900, 4]

            refpoint_embed_ = refpoint_embed_undetach.detach()  # [6, 900, 4]
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # [6, 900, 4] 
            
            # 确保output_memory的序列长度足够
            required_seq_len = topk_proposals.max().item() + 1
            current_seq_len = output_memory.shape[1]
            
            # 如果需要，填充output_memory
            if current_seq_len < required_seq_len:
                padding_size = required_seq_len - current_seq_len
                output_memory = F.pad(output_memory, (0, 0, 0, padding_size), mode='constant', value=0)
            
            # 检查特征维度
            feature_dim = output_memory.shape[2]
            if feature_dim != 256:
                # 创建一个线性层来调整特征维度
                device = output_memory.device
                linear_layer = nn.Linear(feature_dim, 256).to(device)
                output_memory = linear_layer(output_memory)
            
            # 现在根据topk_proposals收集记忆
            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )

            if self.embed_init_tgt: 
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None: 
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        
        img_embs = tgt 

        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, nq, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder( 
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model # list of 6
        # references: n_dec+1, bs, nq, query_dim # list of 7
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard": # yes
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal, img_embs, txt_embs


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0: 
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None: 
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share 
                ) 
            if feature_fusion_layer is not None: 
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                ) 
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint 
        self.use_transformer_ckpt = use_transformer_ckpt 

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers: 
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model

        self.ref_anchor_head = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            # q = k = self.with_pos_embed(tgt, tgt_query_pos)
            # tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            with torch.no_grad():
                q = k = self.with_pos_embed(tgt, tgt_query_pos).detach()
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type, 
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
    )


class Blip2QformerCirAlignPrompt(Blip2Base):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        mlp_hidden_dim=256,
    ):
        super().__init__()
        
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        
        self.mlp = MLP(
            input_dim=self.visual_encoder.num_features,  # 视觉特征维度
            hidden_dim=mlp_hidden_dim,                  # MLP 隐藏层维度
            output_dim=embed_dim,                       # 输出维度
            num_layers=2                                # MLP 层数
        )
        
        self.mlp_projector = MLPProjector(
            input_dim=embed_dim,  # 视觉特征维度
            hidden_dim=self.visual_encoder.num_features,
            output_dim=embed_dim  # 原BLIP-2的embed_dim
        )
    
    def forward(self, visual_features, text_features):
        # 检查维度，如果需要，调整视觉特征的维度
        if hasattr(self, 'mlp') and self.mlp is not None:
            # 检查并保存原始形状
            original_shape = visual_features.shape
            
            # 尝试调整视觉特征维度
            try:
                # 如果视觉特征是3D[batch, seq_len, dim]，尝试直接使用
                if len(visual_features.shape) == 3:
                    input_dim = self.mlp.layers[0].in_features
                    feature_dim = visual_features.shape[2]
                    
                    # 如果维度不匹配，创建适配层
                    if feature_dim != input_dim:
                        adapter = nn.Linear(feature_dim, input_dim).to(visual_features.device)
                        visual_features = adapter(visual_features)
            
                # 应用MLP
                visual_features = self.mlp(visual_features)
            except RuntimeError as e:
                # 判断是否需要重塑张量
                if len(visual_features.shape) == 2:
                    # 可能是被展平的3D张量，尝试恢复
                    if hasattr(self, '_last_batch_size') and hasattr(self, '_last_seq_len'):
                        try:
                            visual_features = visual_features.reshape(self._last_batch_size, self._last_seq_len, -1)
                        except:
                            pass
                            
                # 创建临时适配层
                adapter = nn.Linear(visual_features.shape[-1], self.mlp.layers[0].in_features).to(visual_features.device)
                visual_features = adapter(visual_features)
                visual_features = self.mlp(visual_features)
        
        # 记住当前的批次和序列长度，以便下次可能需要恢复形状
        if len(visual_features.shape) == 3:
            self._last_batch_size, self._last_seq_len = visual_features.shape[0], visual_features.shape[1]
        
        # 通过MLP投影器进行特征融合
        fusion_output = self.mlp_projector(
            visual_features=visual_features,
            text_features=text_features,
        )
        return fusion_output

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, lower_tokens, higher_tokens, lower_mask=None, V_mask=None):
        Q = lower_tokens.transpose(0, 1)
        K = higher_tokens.transpose(0, 1)
        V = higher_tokens.transpose(0, 1)
        attn_output, _ = self.cross_attention(Q, K, V, key_padding_mask=V_mask)
        # Add & norm
        updated_lower_tokens = self.norm(attn_output.transpose(0, 1) + lower_tokens)
        return updated_lower_tokens


def split_tokens(topk_proposals):
    sorted = torch.sort(topk_proposals, dim=1, descending=False)[0]
    num_lower = int(0.9 * topk_proposals.size(1)) 
    lower_idxes = sorted[:, :num_lower]
    higher_idxes = sorted[:, num_lower:]
    
    return lower_idxes, higher_idxes

class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp_visual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mlp_combined = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, visual_features, text_features):
        """ 将视觉特征投影到文本空间，自适应处理不同维度输入 """
        # 保存原始形状以便后面恢复
        original_shape = visual_features.shape
        
        if text_features is None:
            # 维持原始形状的批次和序列长度维度
            if len(original_shape) == 3:
                # 保持批次和序列长度维度[batch, seq_len, dim]
                visual_features_reshaped = visual_features
                result = self.mlp_visual(visual_features_reshaped)
                return result
            else:
                return self.mlp_visual(visual_features)
        
        # 检查维度是否匹配
        try:
            # 维持两个输入的批次和序列长度维度
            if len(original_shape) == 3 and len(text_features.shape) == 3:
                # 需要处理序列长度不同的情况
                batch_size, visual_seq_len, visual_dim = visual_features.shape
                _, text_seq_len, text_dim = text_features.shape
                
                # 如果视觉和文本特征的最后一个维度不相等，需要调整
                if visual_dim != text_dim and visual_dim + text_dim == self.mlp_combined[0].in_features:
                    # 为每个视觉特征找到对应的文本特征
                    # 方法1：复制文本特征以匹配视觉特征的序列长度
                    expanded_text = text_features.unsqueeze(2).expand(batch_size, text_seq_len, visual_seq_len, text_dim)
                    expanded_text = expanded_text.transpose(1, 2).reshape(batch_size * visual_seq_len, text_seq_len, text_dim)
                    
                    # 重塑视觉特征以匹配展开的文本特征
                    reshaped_visual = visual_features.reshape(batch_size * visual_seq_len, visual_dim)
                    
                    # 对于每个视觉特征，选择第一个文本特征进行连接
                    first_text_feature = expanded_text[:, 0, :]
                    combined = torch.cat([reshaped_visual, first_text_feature], dim=-1)
                    result = self.mlp_combined(combined)
                    
                    # 恢复原始形状
                    return result.reshape(batch_size, visual_seq_len, -1)
                else:
                    # 维度不正确，返回仅处理过的视觉特征
                    return self.mlp_visual(visual_features)
            else:
                # 标准情况：直接连接特征
                combined_features = torch.cat([visual_features, text_features], dim=-1)
                return self.mlp_combined(combined_features)
        except RuntimeError as e:
            # 如果连接失败，只使用视觉特征
            if len(original_shape) == 3:
                # 检查是否需要reshape
                if visual_features.shape[0] * visual_features.shape[1] == visual_features.numel() // original_shape[2]:
                    # 数据可能被展平了，需要恢复原始形状
                    visual_features = visual_features.reshape(original_shape)
            
            result = self.mlp_visual(visual_features)
            return result

class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.mlp_visual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mlp_combined = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, visual_features, text_features):
        """ 自适应处理不同维度输入 """
        original_shape = visual_features.shape
        
        if text_features is None:
            if len(original_shape) == 3:
                return self.mlp_visual(visual_features)
            else:
                return self.mlp_visual(visual_features)
        
        # 检查维度是否匹配
        try:
            # 处理3D输入
            if len(original_shape) == 3 and len(text_features.shape) == 3:
                batch_size, visual_seq_len, visual_dim = visual_features.shape
                _, text_seq_len, text_dim = text_features.shape
                
                # 如果视觉和文本特征的最后一个维度不相等，需要调整
                if visual_dim != text_dim and visual_dim + text_dim == self.mlp_combined[0].in_features:
                    # 扩展文本特征以匹配视觉特征的序列长度
                    expanded_text = text_features.unsqueeze(2).expand(batch_size, text_seq_len, visual_seq_len, text_dim)
                    expanded_text = expanded_text.transpose(1, 2).reshape(batch_size * visual_seq_len, text_seq_len, text_dim)
                    
                    # 重塑视觉特征
                    reshaped_visual = visual_features.reshape(batch_size * visual_seq_len, visual_dim)
                    
                    # 使用第一个文本特征
                    first_text_feature = expanded_text[:, 0, :]
                    combined = torch.cat([reshaped_visual, first_text_feature], dim=-1)
                    result = self.mlp_combined(combined)
                    
                    # 恢复原始形状
                    return result.reshape(batch_size, visual_seq_len, -1)
                else:
                    # 维度不正确，只使用视觉特征
                    return self.mlp_visual(visual_features)
            else:
                # 标准情况
                combined_features = torch.cat([visual_features, text_features], dim=-1)
                return self.mlp_combined(combined_features)
        except RuntimeError as e:
            if len(original_shape) == 3:
                # 检查是否需要reshape
                if visual_features.shape[0] * visual_features.shape[1] == visual_features.numel() // original_shape[2]:
                    # 数据可能被展平了，需要恢复原始形状
                    visual_features = visual_features.reshape(original_shape)
                    
            return self.mlp_visual(visual_features)

class ExpertPool(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.experts = nn.ModuleList([
            MLPExpert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        self.gate = GateNetwork(num_experts=num_experts)
        
    def forward(self, visual_features, text_features):
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(visual_features, text_features))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_weights = self.gate()
        return torch.einsum("bnik,n->bik", expert_outputs, gate_weights)

class FeatureInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, features):
        # 增加特征之间的交互
        attn_output, _ = self.cross_attention(features, features, features)
        return self.norm(attn_output + features)