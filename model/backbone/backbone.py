import time
import torch
import torch.nn as nn
from model.utils.tictoc import TicToc
from model.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample

def encode_batch_scenes_points(backbone:nn.Module,
                        batch_graph_dict:dict):
    ''' Read a batch of scene pairs.
    Encode all of their fine points and features.
    '''
    stack_feats_f = []
    stack_feats_f_batch = [torch.tensor(0).long()]
    
    for scene_id in torch.arange(batch_graph_dict['batch_size']):
        tictoc = TicToc()
        feats = batch_graph_dict['batch_features'][scene_id].detach()
        data_dict = batch_graph_dict['batch_points'][scene_id]

        # 2. KPFCNN Encoder
        feats_list = backbone(feats,
                                   data_dict['points'],
                                   data_dict['neighbors'],
                                   data_dict['subsampling'],
                                   data_dict['upsampling'])
        duration = tictoc.toc()
        #
        points_f = data_dict['points'][1]
        feats_f = feats_list[0]
        assert feats_f.shape[0] == points_f.shape[0], 'feats_f and points_f shape are not match'
        
        stack_feats_f.append(feats_f) # (P+Q,C)
        stack_feats_f_batch.append(stack_feats_f_batch[-1]+feats_f.shape[0])
    
    #
    stack_feats_f = torch.cat(stack_feats_f,dim=0) # (P+Q,C)
    stack_feats_f_batch = torch.stack(stack_feats_f_batch,dim=0) # (B+1,)
    
    return {'feats_f':stack_feats_f,
            'feats_batch':stack_feats_f_batch}, duration
    
class KPConvFPN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm, strided=True
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats,
                    points_list,
                    neighbors_list,
                    subsampling_list,
                    upsampling_list):
        '''
        Read a scene pair. 
        Encode all the fine points and features.
        '''
        feats_list = []

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list


# This a copy of the KPConvFPN. It is used to compute the FLOPS of the backbone.
# It accepts input in one list.
class TmpKPConvFPN(KPConvFPN):
    # the init function is the same as the KPConvFPN
    def forward(self, feats,
                    points0,
                    points1,
                    points2,
                    points3,
                    neighbors0,
                    neighbors1,
                    neighbors2,
                    neighbors3,
                    subsampling0,
                    subsampling1,
                    subsampling2,
                    upsampling0,
                    upsampling1,
                    upsampling2):
        '''
        Read a scene pair. 
        Encode all the fine points and features.
        '''
        feats_list = []

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points0, points0, neighbors0)
        feats_s1 = self.encoder1_2(feats_s1, points0, points0, neighbors0)

        feats_s2 = self.encoder2_1(feats_s1, points1, points0, subsampling0)
        feats_s2 = self.encoder2_2(feats_s2, points1, points1, neighbors1)
        feats_s2 = self.encoder2_3(feats_s2, points1, points1, neighbors1)

        feats_s3 = self.encoder3_1(feats_s2, points2, points1, subsampling1)
        feats_s3 = self.encoder3_2(feats_s3, points2, points2, neighbors2)
        feats_s3 = self.encoder3_3(feats_s3, points2, points2, neighbors2)

        feats_s4 = self.encoder4_1(feats_s3, points3, points2, subsampling2)
        feats_s4 = self.encoder4_2(feats_s4, points3, points3, neighbors3)
        feats_s4 = self.encoder4_3(feats_s4, points3, points3, neighbors3)

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling2)
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling1)
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)
        
        feats_list.reverse()
        return feats_list
