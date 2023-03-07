import torch.nn as nn

class MAGICALNet(nn.Module):
    """Custom CNN for MAGICAL policies."""
    def __init__(self, observation_space, image_dim=256, target_dim=16, width=2):

        super().__init__()

        w = width
        def conv_block(i, o, k, s, p, b=False):
            return [
                # batch norm has its own bias, so don't add one to conv layers by default
                nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p, bias=b,
                          padding_mode='zeros'),
                nn.ReLU(),
                #nn.BatchNorm2d(o)
            ]

        #img_shape = observation_space['past_obs'].shape
        img_shape = observation_space.shape

        conv_layers = [
            *conv_block(i=img_shape[0], o=32*w, k=5, s=1, p=2, b=True),
            *conv_block(i=32*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
        ]
        # final FC layer to make feature maps the right size
        test_tensor = torch.zeros((1,) + img_shape)
        for layer in conv_layers:
            test_tensor = layer(test_tensor)
        fc_in_size = np.prod(test_tensor.shape)
        reduction_layers = [
            nn.Flatten(),
            nn.Linear(fc_in_size, image_dim),
            # Stable Baselines will add extra affine layer on top of this reLU
            nn.ReLU(),
        ]
        layers = [*conv_layers, *reduction_layers]
        self.image_feature_layer = nn.Sequential(*layers)

        self.position_feature_layer = nn.Linear(2, TARGET_FEAT_DIM)
        self.colour_embeddings = nn.Embedding(len(en.SHAPE_COLOURS), TARGET_FEAT_DIM)
        self.shape_embeddings  = nn.Embedding(len(en.SHAPE_TYPES), TARGET_FEAT_DIM)

        #self.features_dim = image_dim + target_dim * 3
        self.features_dim = image_dim

    def forward(self, x, traj_info=None):

        visual_feat = self.image_feature_layer(x)

        return visual_feat
