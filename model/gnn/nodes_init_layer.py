import torch
import torch.nn as nn
from model.bert.get_tokenizer import get_tokenlizer, get_pretrained_language_model
from model.bert.bertwarper import generate_bert_fetures

class NodesInitLayer(nn.Module):
    ''' Initialize semantic, boundingbox and shape embeddings of each node.
        Output fused node embeddings.
    '''
    def __init__(self, sg_conf,
                       shape_emb_dim, 
                       online_bert):
        super(NodesInitLayer, self).__init__()
        self.sg_conf = sg_conf
        self.shape_emb_dim = shape_emb_dim

        # BERT
        self.online_bert = online_bert
        if online_bert:
            self.tokenizer = get_tokenlizer('bert-base-uncased')
            self.bert = get_pretrained_language_model('bert-base-uncased')
            self.bert.pooler.dense.weight.requires_grad_(False)
            self.bert.pooler.dense.bias.requires_grad_(False)             
        
        # Semantic
        self.mlp_semantic = torch.nn.Sequential(
            torch.nn.Linear(sg_conf.bert_dim,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,sg_conf.semantic_dim))
        self.mlp_box = torch.nn.Sequential(
            torch.nn.Linear(3,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,sg_conf.box_dim))        
        self.instance_input_dim = sg_conf.semantic_dim + sg_conf.box_dim

        if sg_conf.fuse_shape and sg_conf.fuse_stage=='early':
            self.instance_input_dim += sg_conf.semantic_dim
            self.mlp_shape = nn.Linear(shape_emb_dim,
                                       sg_conf.semantic_dim)
        assert sg_conf.fuse_stage in ['early','late'], 'Invalid fuse stage'
        self.feat_projector = nn.Linear(self.instance_input_dim, 
                                        sg_conf.node_dim) 
        
    def forward(self, data_dict:dict, 
                shape_embeddings:torch.Tensor):
        ''' Initialize nodes with instance features.
        '''
        if 'semantic_embeddings' in data_dict and self.online_bert==False:
            semantic_embeddings = data_dict['semantic_embeddings']
        else:
            semantic_embeddings = generate_bert_fetures(self.tokenizer,
                                                        self.bert,
                                                        data_dict['labels'],
                                                        CUDA=True)
        semantic_feats = self.mlp_semantic(semantic_embeddings)
        box_feats = self.mlp_box(data_dict['boxes'])
        output_feat = [semantic_feats, box_feats]
        if self.sg_conf.fuse_shape and self.sg_conf.fuse_stage=='early':
            output_feat.append(self.mlp_shape(shape_embeddings))
        output_feat = torch.cat(output_feat,dim=1)
        output_feat = self.feat_projector(output_feat)
        
        return output_feat
    
