import os
import json
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from utils.logger import LOGGER
from .general_module import TokenMasker, MMGeneralModule, Contra_head, Match_head
from utils.distributed import all_gather_with_grad, concat_all_gather, all_gather_list
from torch.nn import LayerNorm as LayerNorm
from easydict import EasyDict as edict
from utils.volume import volume_computation4,volume_computation3, volume_computation5, volume_computation2, volume_computation_masked, present_from_feats, gallery_self_volume


class GRAM(MMGeneralModule):
    """ VLP pretraining """
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        self.construct_vision_encoder()
        self.construct_audio_encoder()
        self.construct_multimodal_encoder()

        contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_s = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_a = Contra_head(self.audio_dim, contra_dim)
        self.contra_head_d = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        # ---- hypergraph (GatedHGNN) knobs; defaults => stage A == plain GRAM (byte-identical) ----
        self.stage = getattr(self.config, 'stage', 'A')
        self.w_doc  = float(getattr(self.config, 'w_doc', 0.0))
        # no separate w_xdoc term: the post-graph volume loss is loss_area (see _hg_refine / forward_ret)
        self.w_reg  = float(getattr(self.config, 'w_reg', 0.0))
        # Weight on the volume loss computed over the UN-refined features -- the representation
        # inference actually scores, since the hypergraph is by design absent at test time. 0.0 is
        # the original behaviour (refined features carry the ONLY retrieval loss), which leaves the
        # deployed representation trained only indirectly. See forward_ret for the full argument.
        self.w_raw  = float(getattr(self.config, 'w_raw', 0.0))
        # Distil the REFINED embedding into the raw one: the encoder learns to produce, from a clip
        # ALONE, what the graph produced using that clip's neighbours. This is what makes the method
        # inductive -- neighbourhood knowledge ends up in the weights, so inference needs no graph,
        # no gallery and no extra compute, and the deployed model generalises the way GRAM's does.
        # w_raw alone does NOT do this: it only asserts "the raw features must also retrieve well",
        # a one-scalar-per-clip constraint that never tells the encoder WHAT the graph contributed.
        # Distillation gives a full target vector per clip per modality instead.
        self.w_distill = float(getattr(self.config, 'w_distill', 0.0))
        self.semantic_edges = bool(getattr(self.config, 'semantic_edges', False))
        # Per-clip missing-modality masking in the Gramian volume. This is INDEPENDENT of the
        # hypergraph (it applies in stage A too, and at inference where the graph is off), so it
        # needs its own switch to be ablated. False => present=None => vanilla GRAM, where a
        # zero-filled modality makes the Gram matrix singular and the volume collapses to 0.
        self.masked_volume = bool(getattr(self.config, 'masked_volume', True))
        # Apply the presence mask to SEMANTIC edges as well as doc edges (see semantic_incidence).
        # False restores the earlier behaviour where the graph imputed missing modalities.
        self.sem_edge_presence_mask = bool(getattr(self.config, 'sem_edge_presence_mask', True))
        # Weight semantic edges by caption cosine (w_ij = cos(c_i, c_j)) instead of a flat 1.0.
        # False = the earlier binary graph, where a 0.31 neighbour counted as much as a 0.95 one.
        self.sem_edge_weighted = bool(getattr(self.config, 'sem_edge_weighted', True))
        # Build the hypergraph over the whole DDP batch rather than per-GPU shard. The contrastive
        # loss is already global (local queries vs gathered gallery); this makes the GRAPH global
        # too, so a document's neighbours are drawn from world_size x more candidates.
        self.global_graph = bool(getattr(self.config, 'global_graph', False))
        # WHERE semantic edges get their similarity from.
        #   'caption' : caption kNN. Cannot be used at inference -- the captions are the queries, so
        #               wiring the gallery graph with them leaks the query. Hence training-only.
        #   'fusion'  : the clip's own fused modality descriptor (see _fusion_src). Query-free, so
        #               the SAME graph can be built at train and at test time.
        self.sem_edge_src = str(getattr(self.config, 'sem_edge_src', 'caption'))
        # Apply the hypergraph at INFERENCE, over the whole gallery. Requires sem_edge_src='fusion'
        # for the semantic half to be legal; doc edges alone are always legal.
        self.infer_graph = bool(getattr(self.config, 'infer_graph', False))
        # IMPUTE a missing modality instead of dropping it. The presence mask disconnects a missing
        # vertex from its hyperedges in BOTH directions, so it sends nothing and receives nothing and
        # stays exactly zero -- the graph cannot reconstruct it, and only the masked volume can cope
        # by scoring the clip at lower arity. impute_missing makes the incidence asymmetric: the
        # vertex still sends nothing, but receives its doc's edge summary, so it is reconstructed
        # from the modalities the clip DOES have. The refined volume is then full-arity and directly
        # comparable with complete clips -- which is the point: GRAM's volume is undefined when a
        # modality is absent (singular Gram matrix -> vol 0 -> ties at the top), and imputation makes
        # it defined rather than merely masking it away.
        self.impute_missing = bool(getattr(self.config, 'impute_missing', False))
        # Train on the SUBSPACE DISTANCE rather than the raw volume:
        #     vol(c, z1..zk) = vol(z1..zk) * dist(c, span(z1..zk))
        # The raw-volume loss can be reduced two ways: align the span with the caption (wanted), or
        # shrink vol(z1..zk) by pulling the clip's modalities together (degeneracy). The hypergraph
        # is text-blind by design, so the SECOND lever is the only one it can pull unilaterally --
        # and the measured traces show it pulling it (gate rises while validation falls). Dividing
        # the score by vol(z1..zk) removes that lever algebraically: the degeneracy direction has
        # exactly zero gradient, and the only way any module can reduce the loss is by genuinely
        # rotating the clip's subspace toward its caption. Backprop flows THROUGH the normaliser,
        # so span collapse is actively penalised, not just unrewarded.
        self.train_vol_norm = bool(getattr(self.config, 'train_volume_normalized', False))
        self._gc_edges = 0
        self.knn_k = int(getattr(self.config, 'knn_k', 4))
        self.edge_dropout = float(getattr(self.config, 'edge_dropout', 0.3))
        # optional similarity floor for the semantic kNN; None = plain mutual-kNN
        _ss = getattr(self.config, 'sem_sim_std', None)
        self.sem_sim_std = float(_ss) if _ss is not None else None
        if self.stage == 'B':
            from .hypergraph import GatedHGNN
            self.hgnn = GatedHGNN(contra_dim, n_layers=int(getattr(self.config, 'hgnn_layers', 2)),
                                  gate_init=float(getattr(self.config, 'gate_init', 1.0)),
                                  per_mod_gate=bool(getattr(self.config, 'per_mod_gate', False)))
        self.itm_head = Match_head(self.multimodal_dim)
        self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_vision_sample_num, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_audio_sample_num, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.beam_size  = config.beam_size
        self.itm_ratio = config.itm_ratio   
        self.max_omni_caption_len = config.max_omni_caption_len
        self.max_caption_len = config.max_caption_len
        self.max_subtitle_len = config.max_subtitle_len





   
    def construct_multimodal_encoder(self):    
        
        from .text_encoders.bert.bert import BertForMaskedLM, BertConfig
     
        bertconfig = BertConfig.from_pretrained("./pretrained_weights/bert/bert-base-uncased")
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True
        self.multimodal_encoder = BertForMaskedLM.from_pretrained("./pretrained_weights/bert/bert-base-uncased",config = bertconfig )
        self.multimodal_dim = 768

        if self.config.checkpointing:
            self.multimodal_encoder._set_gradient_checkpointing(self.multimodal_encoder.bert.encoder, True)

        from transformers import BertTokenizer


        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert/bert-base-uncased')
        self.multimodal_encoder.tokenizer.bos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.eos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.multimodal_encoder.tokenizer.pad_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.multimodal_encoder.tokenizer.mask_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        self.text_masker = TokenMasker(mask_token = self.multimodal_encoder.tokenizer.mask_token_id, range_start=106, range_end = 30522)

        

    def batch_get(self, batch, key):
        if key in batch:
            return batch[key]


        elif key == 'caption_tokens':

            caption_tokens = self.multimodal_encoder.tokenizer(batch.raw_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt").to(torch.device('cuda'))
         
            batch[key] = caption_tokens
        
        elif key == 'subtitle_tokens':
         
            subtitle_tokens = self.multimodal_encoder.tokenizer(batch.raw_subtitles,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_subtitle_len,
                                                    return_tensors="pt")
            subtitle_tokens = subtitle_tokens.to(torch.device('cuda'))
            batch[key] = subtitle_tokens
                                        

        elif key == 'vision_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.vision_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        
        elif key == 'audio_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.audio_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens

        elif key == 'omni_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.omni_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_omni_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        elif key == 'caption_output':
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output



        elif key == 'vision_caption_output':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output


        elif key == 'audio_caption_output':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output

       
        elif key == 'subtitle_output':
            subtitle_tokens = self.batch_get(batch, 'subtitle_tokens')
            input_ids = subtitle_tokens.input_ids
            attention_mask = subtitle_tokens.attention_mask
            subtitle_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = subtitle_output
  

        elif key == 'vision_output':
            vision_output = self.forward_vision_encoder(batch.vision_pixels)
            batch[key] = vision_output
            
        elif key == 'depth_output':
            depth_output = self.forward_vision_encoder(batch.depth_pixels)
            batch[key] = depth_output
        
        elif key == 'audio_output':
            audio_output = self.forward_audio_encoder(batch.audio_spectrograms) 
            batch[key] = audio_output


        elif key == 'condition_feats_v':
            vision_output = self.batch_get(batch, 'vision_output')
            condition_feats_v = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_v
            
        elif key == 'condition_feats_d':
            vision_output = self.batch_get(batch, 'depth_output')
            condition_feats_d = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_d
            
        elif key == 'condition_feats_a':
            audio_output = self.batch_get(batch, 'audio_output')
            condition_feats_a = self.get_multimodal_forward_input_audio(audio_output)
            batch[key] = condition_feats_a

        elif key == 'condition_feats_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            condition_feats_s = self.get_multimodal_forward_input_subtitle(subtitle_output)
            batch[key] = condition_feats_s

        elif key == 'condition_feats_va':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_va = torch.cat((condition_feats_v, condition_feats_a),dim=1)
            batch[key] = condition_feats_va

        elif key == 'condition_feats_vs':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vs = torch.cat((condition_feats_v, condition_feats_s),dim=1)
            batch[key] = condition_feats_vs

        elif key == 'condition_feats_vas':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s),dim=1)
            batch[key] = condition_feats_vas
            
        elif key == 'condition_feats_vasd':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_d = self.batch_get(batch, 'condition_feats_d')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s, condition_feats_d),dim=1)
            batch[key] = condition_feats_vas



        elif key == 'feat_v':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            feat_v = self.contra_head_v(vision_output_pooled)
            batch['u_v'] = feat_v   # pre-norm proj for hypergraph loss_reg (stage B; unused in A)
            feat_v = F.normalize(feat_v,dim=-1)
            batch[key] = feat_v
        
        elif key == 'feat_d':
            depth_output = self.batch_get(batch, 'depth_output')
            depth_output_pooled = self.pool_vision_for_contra(depth_output)
            feat_d = self.contra_head_d(depth_output_pooled)  # depth projection head (mirrors the vision contra head)
            batch['u_d'] = feat_d
            feat_d = F.normalize(feat_d,dim=-1)
            batch[key] = feat_d
        
        elif key == 'feat_a':
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_a = self.contra_head_a(audio_output_pooled)
            batch['u_a'] = feat_a
            feat_a = F.normalize(feat_a,dim=-1)
            batch[key] = feat_a

        elif key == 'feat_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_s = self.contra_head_s(subtitle_output_pooled)
            batch['u_s'] = feat_s
            feat_s = F.normalize(feat_s,dim=-1)
            batch[key] = feat_s

        elif key == 'feat_t':
            caption_output = self.batch_get(batch, 'caption_output')
            caption_output_pooled = self.pool_text_for_contra(caption_output)
            feat_t = self.contra_head_t(caption_output_pooled)
            batch['u_t'] = feat_t
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_va':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_va = torch.cat((vision_output_pooled, audio_output_pooled), dim=1)
            feat_va = self.contra_head_va(feat_va)
            feat_va = F.normalize(feat_va,dim=-1)
            batch[key] = feat_va

        elif key == 'feat_vs': 
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vs = torch.cat((vision_output_pooled, subtitle_output_pooled), dim=1)
            feat_vs = self.contra_head_vs(feat_vs)
            feat_vs = F.normalize(feat_vs,dim=-1) 
            batch[key] = feat_vs

        elif key == 'feat_vas':     
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vas = torch.cat((vision_output_pooled, audio_output_pooled, subtitle_output_pooled), dim=1)
            feat_vas = self.contra_head_vas(feat_vas)
            feat_vas = F.normalize(feat_vas,dim=-1)
            batch[key] = feat_vas  
            
            

        elif key == 'feat_t_omni_caption':
            caption_tokens = self.batch_get(batch, 'omni_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_vision_caption':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_audio_caption':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        return batch[key] 


    def forward(self, batch, task, compute_loss=True):
        batch = edict(batch)
        ### gram-27m pretraining
        #if 'vision_captions' in batch or 'audio_captions' in batch or 'omni_captions' in batch:
        #    assert compute_loss
        #    return self.forward_vast27m(batch, task)

        ### other datasets pretraining or finetuning
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret(batch, task, compute_loss=compute_loss)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap(batch, task, compute_loss=compute_loss)
                output_ls.append(cap_dict)

            elif task.startswith('qa'):
                qa_dict = self.forward_qa(batch, task, compute_loss=compute_loss)
                output_ls.append(qa_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict


    def forward_vast27m(self, batch, task):
     
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret_vast27m(batch, task)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap_vast27m(batch, task)
                output_ls.append(cap_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):
            
        slice_output = self.multimodal_encoder.bert(input_ids = slice_input_ids,
                                                    attention_mask = slice_attention_mask,
                                                    encoder_hidden_states=slice_multimodal_vision_input).last_hidden_state
        slice_scores = F.softmax(self.itm_head(slice_output[:,0]),dim=1)[:,1]

        return slice_scores


    @staticmethod
    def _fusion_src(z, mask, present=None):
        """Query-independent per-clip descriptor used to WIRE semantic edges.

        Mean of the clip's present modalities, L2-normed, detached. It contains no text, which is
        the whole point: semantic edges built from captions cannot be used at inference, because
        the captions ARE the queries and wiring the gallery graph with them leaks the query into
        the gallery representation. That is why the caption-based edges are gated behind
        self.training -- and it is why the graph's only novel component (inter-document structure)
        was structurally unavailable at test time, leaving only the doc hyperedge, which largely
        duplicates the cross-modal coupling the Gramian volume already performs.

        Wiring from the gallery side instead makes the edges legal at inference. It is transductive
        -- the graph sees the whole gallery at once -- which is standard and disclosed in video
        retrieval (QB-Norm, dual softmax, inverted softmax all use gallery-side test-time
        structure), but it never sees a query.
        """
        st = torch.stack([z[m].float() for m in mask], dim=1)              # (B, k1, D)
        if present is None:
            s = st.mean(dim=1)
        else:
            w = present.unsqueeze(-1).float()
            s = (st * w).sum(1) / w.sum(1).clamp(min=1.0)
        return F.normalize(s, dim=-1).detach()

    @torch.no_grad()
    def refine_gallery(self, feats_list, order, present=None):
        """Apply the hypergraph to a WHOLE gallery at inference. Returns refined tensors.

        Unlike the training path this runs on the full gallery in one graph (1k-5k clips) rather
        than a 256-document batch, so the semantic neighbourhood is drawn from far more candidates.
        Text is neither refined nor used to build edges, so queries stay out of it entirely.
        """
        from .hypergraph import doc_incidence, mutual_knn_adj, semantic_incidence
        mask = tuple(order)
        z = {m: f for m, f in zip(mask, feats_list)}
        B = feats_list[0].shape[0]
        device = feats_list[0].device
        if present is None:
            present = self._presence(feats_list, mask)
        present = present.float().to(device)
        H_doc = doc_incidence(B, mask, device, present=present).float()
        H_sem = None
        if self.semantic_edges:
            src = self._fusion_src(z, mask, present)
            adj = mutual_knn_adj(src, k=self.knn_k, edge_dropout=0.0, training=False,
                                 sim_std=self.sem_sim_std, weighted=self.sem_edge_weighted)
            H_sem = semantic_incidence(adj, B, mask, device,
                                       present=present if self.sem_edge_presence_mask else None)
        z32 = {m: z[m].float() for m in mask}
        H_recv = None
        if self.impute_missing:
            _hd_r = doc_incidence(B, mask, device, present=None).float()
            _hs_r = semantic_incidence(adj, B, mask, device, present=None) if H_sem is not None else None
            H_recv = _hd_r if _hs_r is None else torch.cat([_hd_r, _hs_r], dim=1)
        z_hat, _h, _hp = self.hgnn(z32, mask, H_doc, H_sem, present=present, H_recv=H_recv)
        return [z_hat[m].to(feats_list[i].dtype) for i, m in enumerate(mask)]

    def _vol_score(self, vol, gal_feats, present, dim):
        """vol -> score used in the contrastive loss.

        train_volume_normalized divides by the gallery clips' own modality volume, turning the raw
        Gramian volume into dist(c, span(clip)) -- see the flag's comment in __init__. `dim` is the
        axis of `vol` indexed by gallery clips (1 for volume, 0 pre-transpose handled by caller
        passing the already-transposed matrix with dim=0's data on rows -> use dim accordingly).
        """
        if self.train_vol_norm:
            gv = gallery_self_volume(gal_feats, present=present).clamp(min=1e-3)
            vol = vol / (gv.unsqueeze(0) if dim == 1 else gv.unsqueeze(1))
        return vol / self.contra_temp

    @staticmethod
    def _presence(feats, order, has_audio=None):
        """(B, L) per-clip modality presence for gallery tensors `feats`, ordered by `order`.

        The norm test is the fallback: a zero feature means the modality is absent. It CANNOT see a
        naturally missing audio track, though -- a missing .wav gives a zero SPECTROGRAM, and BEaTs
        applies LayerNorm, whose output for a constant input is its bias beta, so the encoder emits
        a fixed non-zero vector and the L2-normed feature has norm 1 like any other. `has_audio` is
        what the loader actually observed (data/IndexAnno.py) and overrides the audio column when
        supplied. Shapes are checked rather than assumed: a mismatch would silently mark the wrong
        clips absent, which is worse than falling back to the norm test.
        """
        p = torch.stack([(f.float().norm(dim=-1) > 0.5).float() for f in feats], dim=1)
        if has_audio is not None and 'A' in order:
            ha = has_audio.reshape(-1).to(p.device)
            if ha.shape[0] == p.shape[0]:
                p[:, list(order).index('A')] = ha.to(p.dtype)
        return p

    def _hg_refine(self, feats, t_frozen=None, use_semantic=False, has_audio=None):

        from .hypergraph import doc_incidence, mutual_knn_adj, semantic_incidence
        mask = tuple(m for m in ('V', 'A', 'S', 'D') if m in feats)
        B_local = feats[mask[0]].shape[0]
        device = feats[mask[0]].device
        # GLOBAL GRAPH: build ONE hypergraph over the whole DDP batch instead of one per shard.
        # Only the 512-d contrastive embeddings are gathered (~1.5 MB/step for T+V+A over 4 ranks),
        # never encoder hidden states. all_gather_with_grad is required, not concat_all_gather: a
        # refined embedding depends on every document in the graph, so a no-grad gather would
        # silently drop the cross-rank terms of d(z_hat_i)/d(z_j).
        _ws = dist.get_world_size() if dist.is_initialized() else 1
        go_global = bool(self.global_graph) and _ws > 1
        if go_global:
            _rank = dist.get_rank()
            feats = {m: all_gather_with_grad(v) for m, v in feats.items()}
            if t_frozen is not None:
                # topology only, and mutual_knn_adj detaches anyway -> a no-grad gather is correct
                t_frozen = concat_all_gather(t_frozen)
        if go_global and has_audio is not None:
            has_audio = concat_all_gather(has_audio.reshape(-1))   # align with the gathered feats
        B = feats[mask[0]].shape[0]
        # per-clip modality presence. A missing modality is masked out of the doc edge (no messages)
        # and the fusion mean (no dilution), so the graph's relation-building is unaffected by absent
        # modalities. All-present -> ones -> unchanged.
        pres = self._presence([feats[m] for m in mask], mask, has_audio)   # (B, k1)
        H_doc = doc_incidence(B, mask, device, present=pres).float()
        H_sem = None
        # Edge source: captions (training-only, leaks at inference) or the query-free gallery
        # fusion (legal everywhere). With 'fusion' the graph is IDENTICAL at train and test time,
        # which is what makes refine_gallery a faithful deployment of what was trained.
        if self.sem_edge_src == 'fusion':
            t_frozen = self._fusion_src({m: feats[m] for m in mask}, mask, pres)
        if use_semantic and self.semantic_edges and t_frozen is not None \
                and (self.training or self.sem_edge_src == 'fusion'):
            _stats = {} if (self._gc_edges % 50 == 1 and dist.get_rank() == 0) else None
            adj = mutual_knn_adj(t_frozen.detach(), k=self.knn_k,
                                 edge_dropout=self.edge_dropout, training=True,
                                 sim_std=self.sem_sim_std, stats=_stats,
                                 weighted=self.sem_edge_weighted,
                                 gen_seed=(987654321 + self._gc_edges) if go_global else None)
            self._gc_edges += 1
            if _stats:
                # edge_cos vs batch_cos is the signal: edges no more similar than the batch average
                # mean the semantic wiring is not selecting genuine relations.
                _z = ((_stats['edge_cos'] - _stats['batch_cos']) / _stats['batch_cos_std']
                      if _stats['batch_cos_std'] > 1e-6 else float('nan'))
                print(f"[EDGES] step~{self._gc_edges}: B={_stats['B']} k={_stats['k']} "
                      f"edges={_stats['edges']:.0f} deg={_stats['deg_mean']:.2f} "
                      f"isolated={_stats['isolated']}/{_stats['B']} | "
                      f"edge_cos={_stats['edge_cos']:.3f} batch_cos={_stats['batch_cos']:.3f}"
                      f"(sd {_stats['batch_cos_std']:.3f}) z=+{_z:.2f}", flush=True)
            H_sem = semantic_incidence(adj, B, mask, device,
                                       present=pres if self.sem_edge_presence_mask else None)
            if H_sem is not None:
                H_sem = H_sem.float()
        with torch.cuda.amp.autocast(enabled=False):
            z32 = {m: feats[m].float() for m in feats}
            H_recv = None
        if self.impute_missing:
            # Same hyperedges, no presence factor: a missing vertex is a RECEIVER of its doc edge
            # (and of any semantic edges) while still contributing nothing to them.
            _hd_r = doc_incidence(B, mask, device, present=None).float()
            _hs_r = semantic_incidence(adj, B, mask, device, present=None) if H_sem is not None else None
            H_recv = _hd_r if _hs_r is None else torch.cat([_hd_r, _hs_r], dim=1)
        z_hat, h, h_prenorm = self.hgnn(z32, mask, H_doc, H_sem, present=pres, H_recv=H_recv)
        if go_global:
            # GatherLayer concatenates rank-ordered, so this rank owns rows [rank*B_local, ...).
            # Slicing back keeps every caller downstream (losses, the later gathers) unchanged.
            _sl = slice(_rank * B_local, (_rank + 1) * B_local)
            z_hat = {m: v[_sl] for m, v in z_hat.items()}
            h, h_prenorm = h[_sl], h_prenorm[_sl]
        return z_hat, h, h_prenorm

    def forward_ret(self, batch, task, compute_loss=True):

        if isinstance(batch.raw_captions[0],list): #### test
            batch.raw_captions = [i for j in batch.raw_captions for i in j]
        subtasks = task.split('%')[1:]
        if compute_loss:
            loss_dict={}
            loss_itc = []
            loss_itm = []
            loss_area = []

            #Extract text features
            feat_t = self.batch_get(batch,'feat_t')
            #Extract visual features
            feat_v = self.batch_get(batch,'feat_v')
            #Extract audio features
            feat_a = self.batch_get(batch,'feat_a')
            #Extract subtitles features
            if "raw_subtitles" in batch.keys():
                feat_s = self.batch_get(batch,'feat_s')
            #extract depth features
            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch,'feat_d')

            # ---- Hypergraph refinement is on the main path: the volume loss below is computed on
            # the REFINED embeddings. The graph is deliberately absent at inference (it exists to
            # shape the encoders during training, not to be part of the deployed model), which
            # creates a train/test mismatch this block has to account for:
            #
            #   training  optimises the encoders so that HGNN(encoder(x)) retrieves well
            #   inference scores encoder(x) directly, with no HGNN
            #
            # With the gate at 0.7-0.8, HGNN(x) differs substantially from x, so the encoders are
            # free to offload work onto a module that will not be there -- and the representation
            # inference actually uses never receives a direct retrieval gradient. That is the
            # standard failure mode for a discarded auxiliary module, and it matches the measured
            # result (stage B straddles stage A: +0.5 at 5 epochs, -1.2 at 1).
            #
            # w_raw > 0 adds the SAME volume loss on the un-refined features (see below), so the
            # inference-path representation is trained directly and the graph becomes a genuine
            # regulariser on top of it rather than a substitute for it. w_raw = 0 is the original
            # refined-only behaviour.
            hg_h = hg_h_prenorm = None
            _raw = None
            # Loader-reported audio presence (data/IndexAnno.py). Absent from older configs and
            # from datasets with no audio -> None -> every presence test falls back to the norm.
            _ha = batch['has_audio'] if 'has_audio' in batch else None
            # gallery modality order, mirroring how _g/_gT are built below
            _ord = ['V', 'A']
            if "raw_subtitles" in batch.keys():
                _ord.append('S')
                if "depth_pixels" in batch.keys():
                    _ord.append('D')
            if self.stage == 'B':
                # LEAK-FREE: refine ONLY non-text (gallery) modalities; feat_t stays the raw anchor.
                _f = {'V': feat_v, 'A': feat_a}
                if "raw_subtitles" in batch.keys(): _f['S'] = feat_s
                if "depth_pixels" in batch.keys():  _f['D'] = feat_d
                if self.w_raw > 0 or self.w_distill > 0:
                    _raw = dict(_f)      # pre-refinement copies: raw-path loss and/or distillation target
                z_hat, hg_h, hg_h_prenorm = self._hg_refine(_f, t_frozen=feat_t, use_semantic=True,
                                                            has_audio=_ha)
                self._gc=getattr(self,"_gc",0)+1
                if self._gc%50==1 and dist.get_rank()==0: print(f"[GATE] step~{self._gc}: {self.hgnn.gates.detach().float().tolist()}",flush=True)
                feat_v, feat_a = z_hat['V'], z_hat['A']        # feat_t UNCHANGED (raw)
                if 'S' in z_hat: feat_s = z_hat['S']
                if 'D' in z_hat: feat_d = z_hat['D']

            feat_t_all = concat_all_gather(feat_t)
            feat_v_all = concat_all_gather(feat_v)
            feat_a_all = concat_all_gather(feat_a)
            if "raw_subtitles" in batch.keys():
                feat_s_all = concat_all_gather(feat_s)
            if "depth_pixels" in batch.keys():
                feat_d_all = concat_all_gather(feat_d)
            # additional modality features are gathered above as needed
            
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids_collate = concat_all_gather(input_ids)
            attention_mask_collate = concat_all_gather(attention_mask)


            #       VOLUME LOSS COMPUTATION
            #           VOLUME ITC

            #Volume (Text, batch_all)
            # per-clip missing-modality masking, ON in training too (no filter now -> a clip may lack a
            # modality; its volume is taken over what it has). Complete clip -> present all-ones ->
            # volume_computation{3,4,5} byte-for-byte. Same masked volume + same presence rule as eval.
            if "raw_subtitles" in batch.keys():
                if "depth_pixels" in batch.keys():
                    _g = [feat_v_all,feat_a_all,feat_s_all,feat_d_all]
                else:
                    _g = [feat_v_all,feat_a_all,feat_s_all]
            else:
                _g = [feat_v_all,feat_a_all]
            _ha_all = concat_all_gather(_ha.reshape(-1)) if _ha is not None else None
            # Verify the loader flag is actually arriving: without it the audio column falls back to
            # a norm test that can never be 0 (BEaTs' LayerNorm bias), so the mask silently no-ops.
            if self._gc_edges % 50 == 1 and (not dist.is_initialized() or dist.get_rank() == 0):
                _na = int((_ha_all < 0.5).sum().item()) if _ha_all is not None else -1
                print(f"[PRESENCE] loader has_audio={'yes' if _ha is not None else 'NO -> norm fallback'}"
                      f"  audio_absent={_na}/{0 if _ha_all is None else _ha_all.shape[0]}"
                      f"  masked_volume={self.masked_volume}", flush=True)
            _pres_g = self._presence(_g, _ord, _ha_all) \
                if (self.masked_volume and not self.impute_missing) else None
            volume = self._vol_score(
                volume_computation_masked(feat_t, _g, present=_pres_g), _g, _pres_g, dim=1)
            #AreaT (Video,batch_all)
            if "raw_subtitles" in batch.keys():
                if "depth_pixels" in batch.keys():
                    _gT = [feat_v,feat_a,feat_s,feat_d]
                else:
                    _gT = [feat_v,feat_a,feat_s]
            else:
                _gT = [feat_v,feat_a]
            _pres_gT = self._presence(_gT, _ord, _ha) \
                if (self.masked_volume and not self.impute_missing) else None
            volumeT = self._vol_score(
                volume_computation_masked(feat_t_all, _gT, present=_pres_gT).T, _gT, _pres_gT, dim=0)
            rank = dist.get_rank()
            bs = feat_t.size(0)
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(volume.device)

            loss = (
                    F.cross_entropy(-volume, targets, label_smoothing=0.1) #d2a
                    + F.cross_entropy(-volumeT, targets, label_smoothing=0.1) #a2d
            ) / 2

            loss_area.append(loss)

            # ---- raw-path retrieval loss: the SAME Gramian-volume objective on the UN-refined
            # features, i.e. exactly what inference will score. Without this the encoders are only
            # ever asked to be good after refinement (see the note above the _hg_refine call), so
            # the deployed representation is optimised only indirectly. Same targets, same temp,
            # same masking, same modality order as the refined path -- the only difference is which
            # tensors go in.
            if _raw is not None and self.w_raw > 0:
                _rl = [_raw[m] for m in ('V', 'A', 'S', 'D') if m in _raw]     # local
                _ra = [concat_all_gather(x) for x in _rl]                      # gathered
                _pres_ra = self._presence(_ra, _ord, _ha_all) if self.masked_volume else None
                _vol_r = self._vol_score(
                    volume_computation_masked(feat_t, _ra, present=_pres_ra), _ra, _pres_ra, dim=1)
                _pres_rl = self._presence(_rl, _ord, _ha) if self.masked_volume else None
                _vol_rT = self._vol_score(
                    volume_computation_masked(feat_t_all, _rl, present=_pres_rl).T, _rl, _pres_rl, dim=0)
                loss_dict['loss_area_raw'] = self.w_raw * (
                    F.cross_entropy(-_vol_r, targets, label_smoothing=0.1)
                    + F.cross_entropy(-_vol_rT, targets, label_smoothing=0.1)
                ) / 2

            # ---- distillation: pull the raw embedding toward the refined one ------------------
            # stop-grad on the target, so the graph is a TEACHER: gradients shape the encoder to
            # imitate the refined vector, rather than letting the graph drift toward whatever the
            # encoder already produces (which would make the objective trivially satisfiable).
            # Both sides are already L2-normed, so cosine is a dot product; 1-cos in [0, 2].
            if _raw is not None and self.w_distill > 0:
                _dk = [m for m in ('V', 'A', 'S', 'D') if m in _raw and m in z_hat]
                if _dk:
                    _d = sum((1.0 - (F.normalize(_raw[m].float(), dim=-1)
                                     * F.normalize(z_hat[m].detach().float(), dim=-1)).sum(-1)).mean()
                             for m in _dk) / len(_dk)
                    loss_dict['loss_distill'] = self.w_distill * _d

            # hypergraph auxiliaries (loss_area above is the graph's retrieval loss)
            if self.stage == 'B':
                u = {'T': batch['u_t'], 'V': batch['u_v'], 'A': batch['u_a']}
                if "raw_subtitles" in batch.keys(): u['S'] = batch['u_s']
                if "depth_pixels" in batch.keys():  u['D'] = batch['u_d']
                # loss_doc: text anchor vs doc-embedding h, arity-2 Gramian volume (same geometry as loss_area)
                h_all = all_gather_with_grad(hg_h)
                ztl_all = concat_all_gather(feat_t).float()
                vol_t2h = volume_computation2(feat_t.float(), h_all) / self.contra_temp
                vol_h2t = volume_computation2(hg_h, ztl_all) / self.contra_temp
                loss_doc = (F.cross_entropy(-vol_t2h, targets, label_smoothing=0.1)
                            + F.cross_entropy(-vol_h2t, targets, label_smoothing=0.1)) / 2
                # loss_reg: VICReg-style variance hinge on pre-norm feats + h_prenorm
                def _vh(x):
                    std = torch.sqrt(x.float().var(dim=0) + 1e-4)
                    return F.relu(1.0 - std).mean()
                loss_reg = sum(_vh(u[m]) for m in u) + _vh(hg_h_prenorm)
                loss_dict['loss_doc']  = self.w_doc  * loss_doc
                loss_dict['loss_reg']  = self.w_reg  * loss_reg


            #   AREA VID ITM
            #vid_sim = feat_t @ feat_v_all.T
            #vid_simT = feat_v @ feat_t_all.T
                     
            condition_feats = self.batch_get(batch, f'condition_feats_va')#self.batch_get(batch, f'condition_feats_v')
            condition_feats_collate = all_gather_with_grad(condition_feats)
            with torch.no_grad():
                weights_t2cond = F.softmax(-(volume), dim=1) + 1e-4
                weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                weights_cond2t = F.softmax(-(volumeT), dim=1) + 1e-4
                weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

            condition_feats_neg = []
            for b in range(bs): 
                neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                condition_feats_neg.append(condition_feats_collate[neg_idx])
            condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                text_ids_neg.append(input_ids_collate[neg_idx])
                text_atts_neg.append(attention_mask_collate[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
            input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
            attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
            condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
            output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                                        attention_mask = attention_mask_1,
                                        encoder_hidden_states=condition_feats).last_hidden_state
            batch_size = condition_feats_neg.shape[0]
            logits = self.itm_head(output[:,0].half())
            ground_truth = torch.zeros(batch_size*3).long().cuda()
            ground_truth[:batch_size] = 1
            loss = F.cross_entropy(logits,ground_truth) #itm (dtm)
            loss_itm.append(self.itm_ratio * loss)

            

            for task in subtasks:


                # #### compute_itc
                # assert task in ['tv','ta','tva','tvs','tvas']
                # #feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                # #feat_cond_all = concat_all_gather(feat_cond)
                # sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1,0))
                # sim_cond2t = sim_cond2t / self.contra_temp
                # sim_t2cond = torch.matmul(feat_t, feat_cond_all.permute(1,0))
                # sim_t2cond = sim_t2cond / self.contra_temp  # [batch_size, batch_size*num_gpu]
                # rank = dist.get_rank()
                # bs = feat_t.size(0)
                # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_cond.device)
                # loss = (
                #     F.cross_entropy(sim_cond2t, targets, label_smoothing=0.1)
                #     + F.cross_entropy(sim_t2cond, targets, label_smoothing=0.1)
                # ) / 2
                
                loss_itc.append(torch.tensor(0))#*loss)

                #### compute_itm
         
                # condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                # condition_feats_collate = all_gather_with_grad(condition_feats)
                # with torch.no_grad():
                #     weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                #     weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                #     weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                #     weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                # condition_feats_neg = []
                # for b in range(bs): 
                #     neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                #     condition_feats_neg.append(condition_feats_collate[neg_idx])
                # condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                # text_ids_neg = []
                # text_atts_neg = []
                # for b in range(bs):
                #     neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                #     text_ids_neg.append(input_ids_collate[neg_idx])
                #     text_atts_neg.append(attention_mask_collate[neg_idx])

                # text_ids_neg = torch.stack(text_ids_neg, dim=0)
                # text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
                # input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
                # attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
                # condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
                # output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                #                             attention_mask = attention_mask_1,
                #                             encoder_hidden_states=condition_feats).last_hidden_state
                # batch_size = condition_feats_neg.shape[0]
                # logits = self.itm_head(output[:,0].half())
                # ground_truth = torch.zeros(batch_size*3).long().cuda()
                # ground_truth[:batch_size] = 1
                # loss = F.cross_entropy(logits,ground_truth)
                #loss_itm.append(torch.tensor(0))#*(self.itm_ratio * loss))

            loss_itc = sum(loss_itc)/len(loss_itc)
            loss_dict['loss_itc'] = loss_itc          
            loss_itm = sum(loss_itm)/len(loss_itm)
            loss_dict['loss_itm'] = loss_itm
            loss_area = sum(loss_area)/len(loss_area)
            loss_dict['loss_area'] = loss_area
            return loss_dict
          
        else:

            evaluation_dict = {}
            feat_t = self.batch_get(batch,'feat_t')
            feat_v = feat_a = feat_s = feat_d = None
            if "vision_pixels" in batch.keys():          # audio-only (AudioCaps T-A): no video -> skip feat_v
                feat_v = self.batch_get(batch,'feat_v')
            feat_a = self.batch_get(batch,'feat_a')
            if "raw_subtitles" in batch.keys():
                feat_s = self.batch_get(batch,'feat_s')
            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch,'feat_d')


            # eval / validation is GRAM-faithful: NO hypergraph refinement here. The graph is applied
            # on the main path during TRAINING only; at inference feat_v/a/s/d stay the raw GRAM
            # embeddings, fed unchanged to the volume/ITM computation below (identical to GRAM).

            evaluation_dict['feat_t'] = feat_t
            if feat_v is not None: evaluation_dict['feat_v'] = feat_v
            evaluation_dict['feat_a'] = feat_a
            if feat_s is not None: evaluation_dict['feat_s'] = feat_s
            if feat_d is not None: evaluation_dict['feat_d'] = feat_d


            caption_tokens = self.batch_get(batch,'caption_tokens')
            evaluation_dict['input_ids'] = caption_tokens.input_ids
            evaluation_dict['attention_mask'] = caption_tokens.attention_mask
            for task in subtasks:
                #### compute_itc
                assert task in ['tv','ta','tva','tvs','tvas','tvasd']
                # feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                # evaluation_dict[f'feat_cond_{task}'] = feat_cond

                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                evaluation_dict[f'condition_feats_{task}'] = condition_feats

            return evaluation_dict

    def forward_cap(self, batch, task, compute_loss=True):
        subtasks = task.split('%')[1:]

        if compute_loss:

            loss_dict = {}
            loss_ls_cap = []

            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])

            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                output = self.multimodal_encoder(input_ids = input_ids,
                                                attention_mask = attention_mask,
                                                encoder_hidden_states=condition_feats,
                                                labels = txt_labels)
                loss_ls_cap.append(output.loss)

            loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
            loss_dict['loss_cap'] = loss_cap
            return loss_dict

        else:
            evaluation_dict = {}
            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')

                batch_size = condition_feats.shape[0]
                if self.config.captioner_mode:
                    batch_size *=self.config.generate_nums

                init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
                init_attention_mask = init_input_ids.new_ones(batch_size, 1, 1)
                
                if self.config.captioner_mode:
                    condition_feats = condition_feats.unsqueeze(1).expand(-1, self.config.generate_nums,-1,-1).reshape(-1,*condition_feats.shape[1:])
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                            attention_mask=init_attention_mask,
                                                            do_sample = True,
                                                            top_k = 10,
                                                            encoder_hidden_states=condition_feats,
                                                            max_new_tokens=self.max_caption_len,
                                                            eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                            pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 

                                                        
                else:
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                        attention_mask=init_attention_mask,
                                        encoder_hidden_states=condition_feats,
                                        max_new_tokens=self.max_caption_len,
                                        num_beams=self.beam_size,
                                        eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                        pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id,
                                        length_penalty=0.6) 
                                                        
                outputs_newgen = outputs[:,1:]
                captions = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
                evaluation_dict[f'generated_captions_{task}'] = captions

            return evaluation_dict



    def forward_qa(self, batch, task, compute_loss=True):
        subtasks = task.split('%')[1:]
        raw_questions = batch.raw_questions
        raw_answers = batch.raw_answers
      
        if isinstance(raw_questions[0],list): #### test
            # raw_batch_size = len(raw_questions)
            num_questions = [len(i) for i in raw_questions]
            raw_questions = [j for d in raw_questions for j in d]

        question_tokens = self.multimodal_encoder.tokenizer(raw_questions,
                                                            padding="max_length",
                                                            truncation=True,
                                                            max_length=self.max_caption_len,
                                                            return_tensors="pt").to(torch.device('cuda'))

        question_tokens_ids, question_tokens_mask = question_tokens.input_ids, question_tokens.attention_mask

        if compute_loss:

            loss_dict = {}
            loss_ls_qa = []

            answer_tokens = self.multimodal_encoder.tokenizer(raw_answers,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=10,
                                                    return_tensors="pt")
    
            answer_tokens = answer_tokens.to(torch.device('cuda'))
            answer_tokens_ids, answer_tokens_mask = answer_tokens.input_ids, answer_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(answer_tokens_ids, 0.99)
            input_ids = torch.cat((question_tokens_ids,input_ids),dim=1)
            attention_mask = torch.cat((question_tokens_mask,answer_tokens_mask),dim=1)
            dummy_labels = (-100*torch.ones_like(question_tokens_ids)).cuda()
            txt_labels = torch.cat((dummy_labels,txt_labels),dim=1)

            #### part-causal attention mask
            question_len = question_tokens_ids.shape[1]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, question_len: seq_len, question_len: seq_len] = torch.tril(attention_mask[:, question_len: seq_len, question_len: seq_len])
            attention_mask[:, :question_len, question_len:seq_len] = 0


            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                output = self.multimodal_encoder(input_ids = input_ids,
                                attention_mask = attention_mask,
                                encoder_hidden_states=condition_feats,
                                labels = txt_labels)
                loss_ls_qa.append(output.loss)
            
            loss_qa = sum(loss_ls_qa)/len(loss_ls_qa)
            loss_dict['loss_qa'] = loss_qa
            return loss_dict
        
        else:
            evaluation_dict = {} 
            batch_size = sum(num_question)
            init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
            init_input_ids = torch.cat((question_tokens['input_ids'],init_input_ids),dim=1)
            question_len = question_tokens['input_ids'].shape[1]
            seq_len = init_input_ids.shape[1]
            attention_mask = question_tokens['attention_mask'].unsqueeze(1).expand(-1, question_len, -1).clone()
            init_attention_mask = self.multimodal_encoder.update_attention_mask(attention_mask)


            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
            
                condition_feats_expand = []
                for i in range(condition_feats.shape[0]):
                    condition_feats_expand.append( condition_feats[i:i+1].expand(num_questions[i],-1,-1))
                condition_feats = torch.cat(condition_feats_expand,dim=0)
                batch_size = condition_feats.shape[0]
                       
                outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                            attention_mask=init_attention_mask,
                                                            encoder_hidden_states=condition_feats,
                                                            max_new_tokens=10,
                                                            num_beams=self.beam_size,
                                                            eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                            pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 
                
                outputs_newgen = outputs[:,seq_len:]
                answers = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
                print(answers)
                evaluation_dict[f'generated_answers_{task}'] = answers


            return evaluation_dict




    def forward_cap_vast27m(self, batch, task):


        subtasks = task.split('%')[1:]
        loss_dict = {}
        loss_ls_cap = []

        for task in subtasks:
            assert task in ['tv','ta','tva','tvs','tvas']
            if task == 'tv':
                caption_tokens = self.batch_get(batch, 'vision_caption_tokens')

            elif task == 'ta':
                caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
                
            else:
                caption_tokens = self.batch_get(batch, 'omni_caption_tokens')

            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])
            condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
            output = self.multimodal_encoder(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            encoder_hidden_states=condition_feats,
                                            labels = txt_labels)
            loss_ls_cap.append(output.loss)

        loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
        loss_dict['loss_cap'] = loss_cap
        return loss_dict


    def forward_ret_vast27m(self, batch, task):
        

        subtasks = task.split('%')[1:]  
        if compute_loss:
            loss_dict={}
            loss_itc = []
            loss_itm = []
 
            for task in subtasks:
                #### compute_itc
                assert task in ['tv','ta','tva','tvs','tvas']
                if task == 'tv':
                    feat_t = self.batch_get(batch,'feat_t_vision_caption')
                    caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
                elif task == 'ta':
                    feat_t = self.batch_get(batch,'feat_t_audio_caption')
                    caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
                else:
                    feat_t = self.batch_get(batch,'feat_t_omni_caption')
                    caption_tokens = self.batch_get(batch, 'omni_caption_tokens', txt_len = self.max_omni_caption_len)
               
                feat_t_all = concat_all_gather(feat_t) 

                input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
                input_ids_collate = concat_all_gather(input_ids)
                attention_mask_collate = concat_all_gather(attention_mask)

                feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                feat_cond_all = concat_all_gather(feat_cond)
                sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1,0))
                sim_cond2t = sim_cond2t / self.contra_temp
                sim_t2cond = torch.matmul(feat_t, feat_cond_all.permute(1,0))
                sim_t2cond = sim_t2cond / self.contra_temp  # [batch_size, batch_size*num_gpu]
                rank = dist.get_rank()
                bs = feat_t.size(0)
                targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_cond.device)
                loss = (
                    F.cross_entropy(sim_cond2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2cond, targets, label_smoothing=0.1)
                ) / 2
                
                loss_itc.append(loss)

                #### compute_itm
         
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                condition_feats_collate = all_gather_with_grad(condition_feats)
                with torch.no_grad():
                    weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                    weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                    weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                    weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                condition_feats_neg = []
                for b in range(bs): 
                    neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                    condition_feats_neg.append(condition_feats_collate[neg_idx])
                condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                text_ids_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                    text_ids_neg.append(input_ids_collate[neg_idx])
                    text_atts_neg.append(attention_mask_collate[neg_idx])

                text_ids_neg = torch.stack(text_ids_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
                input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
                attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
                condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
                output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                                            attention_mask = attention_mask_1,
                                            encoder_hidden_states=condition_feats).last_hidden_state
                batch_size = condition_feats_neg.shape[0]
                logits = self.itm_head(output[:,0].half())
                ground_truth = torch.zeros(batch_size*3).long().cuda()
                ground_truth[:batch_size] = 1
                loss = F.cross_entropy(logits,ground_truth)
                loss_itm.append(self.itm_ratio * loss)

            loss_itc = sum(loss_itc)/len(loss_itc)
            loss_dict['loss_itc'] = loss_itc          
            loss_itm = sum(loss_itm)/len(loss_itm)
            loss_dict['loss_itm'] = loss_itm

            return loss_dict
          
