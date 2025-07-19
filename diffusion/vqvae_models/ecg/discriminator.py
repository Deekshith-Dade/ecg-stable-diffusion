import torch
import torch.nn as nn
import warnings


class ECGNetworkWarning(UserWarning):
    pass


class spatialResidualBlock(nn.Module):
    def __init__(self, in_channels=(64, 64), out_channels=(64, 64), kernel_size=7, stride=1, groups=1, bias=True, padding=3, dropout=False):
        super(spatialResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels[0],
                               out_channels=out_channels[0],
                               kernel_size=(kernel_size, 1),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(padding, 0))
        self.batchNorm1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=in_channels[1],
                               out_channels=out_channels[1],
                               kernel_size=(kernel_size, 1),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(padding, 0))
        self.batchNorm2 = nn.BatchNorm2d(out_channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.drop = nn.Dropout()

        if in_channels[0] != out_channels[-1]:
            self.resampleInput = nn.Sequential(nn.Conv2d(in_channels=in_channels[0],
                                                         out_channels=out_channels[-1],
                                                         kernel_size=(1, 1),
                                                         bias=bias,
                                                         padding=0),
                                               nn.BatchNorm2d(out_channels[-1]))
        else:
            self.resampleInput = None

    def forward(self, X):
        if self.resampleInput is not None:
            identity = self.resampleInput(X)
        else:
            identity = X
        features = self.conv1(X)
        features = self.batchNorm1(features)
        features = self.relu(features)

        features = self.conv2(features)
        features = self.batchNorm2(features)
        if self.dropout:
            features = self.drop(features)
        features += identity
        features = self.relu(features)
        return features


class temporalResidualBlock(nn.Module):
    def __init__(self, in_channels=(64, 64), out_channels=(64, 64), kernel_size=3, stride=1, groups=1, bias=True, padding=1, dropout=False):
        super(temporalResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels[0],
                               out_channels=out_channels[0],
                               kernel_size=(1, kernel_size),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(0, padding))
        self.conv2 = nn.Conv2d(in_channels=in_channels[1],
                               out_channels=out_channels[1],
                               kernel_size=(1, kernel_size),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(0, padding))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.drop = nn.Dropout()
        self.batchNorm1 = nn.BatchNorm2d(out_channels[0])
        self.batchNorm2 = nn.BatchNorm2d(out_channels[1])

        if in_channels[0] != out_channels[-1]:
            self.resampleInput = nn.Sequential(nn.Conv2d(in_channels=in_channels[0],
                                                         out_channels=out_channels[-1],
                                                         kernel_size=(1, 1),
                                                         bias=bias,
                                                         padding=0),
                                               nn.BatchNorm2d(out_channels[-1]))
        else:
            self.resampleInput = None

    def forward(self, X):
        if self.resampleInput is not None:
            identity = self.resampleInput(X)
        else:
            identity = X
        features = self.conv1(X)
        features = self.batchNorm1(features)
        features = self.relu(features)

        features = self.conv2(features)
        features = self.batchNorm2(features)
        if self.dropout:
            features = self.drop(features)

        features += identity
        features = self.relu(features)
        return features


class ECG_SpatioTemporalNet(torch.nn.Module):
    def __init__(self, temporalResidualBlockParams, spatialResidualBlockParams, firstLayerParams, lastLayerParams, integrationMethod='add', problemType='Binary'):
        super(ECG_SpatioTemporalNet, self).__init__()
        self.firstLayer = nn.Sequential(nn.Conv2d(in_channels=firstLayerParams['in_channels'],
                                                  out_channels=firstLayerParams['out_channels'],
                                                  kernel_size=firstLayerParams['kernel_size'],
                                                  bias=firstLayerParams['bias'],
                                                  padding=(0, int(firstLayerParams['kernel_size'][1]/2))),
                                        nn.BatchNorm2d(
                                            firstLayerParams['out_channels']),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d((1, firstLayerParams['maxPoolKernel'])))
        self.residualBlocks_time = self._generateResidualBlocks(
            **temporalResidualBlockParams)  # type: ignore

        self.residualBlocks_space = self._generateResidualBlocks(
            **spatialResidualBlockParams)  # type: ignore
        if integrationMethod == 'add':
            integrationChannels = temporalResidualBlockParams['out_channels'][-1][-1]
        elif integrationMethod == 'concat':
            integrationChannels = temporalResidualBlockParams['out_channels'][-1][-1] + \
                spatialResidualBlockParams['out_channels'][-1][-1]
        else:
            warnings.warn(
                f'Unknown concatenation method. Defaulting to addtion.', ECGNetworkWarning)
            integrationChannels = temporalResidualBlockParams['out_channels'][-1][-1]

        self.integrationBlock = nn.Sequential(nn.Conv2d(in_channels=integrationChannels,
                                                        out_channels=integrationChannels,
                                                        kernel_size=(3, 3),
                                                        bias=firstLayerParams['bias'],
                                                        padding=1),
                                              nn.BatchNorm2d(
                                                  integrationChannels),
                                              nn.Dropout(),
                                              nn.ReLU(inplace=True))

        self.finalLayer = nn.Sequential(nn.AdaptiveAvgPool2d(lastLayerParams['maxPoolSize']),
                                        nn.Flatten(),
                                        nn.Linear(in_features=lastLayerParams['maxPoolSize'][0]*lastLayerParams['maxPoolSize'][1]*integrationChannels,
                                                  out_features=1))
        self.integrationMethod = integrationMethod
        self.problemType = problemType
        if problemType == 'Binary':
            self.finalLayer = nn.Sequential(*self.finalLayer, nn.Sigmoid())
        elif problemType == 'Regression':
            self.finalLayer = nn.Sequential(*self.finalLayer, nn.ReLU())
        elif problemType == 'RegressionSoftPlus':
            self.finalLayer = nn.Sequential(*self.finalLayer, nn.Softplus())
        elif problemType == 'BCELogits':
            pass  # exclude adding the relu or sigmoid

    def forward(self, X):
        resInputs = self.firstLayer(X)
        spatialFeatures = self.residualBlocks_space(resInputs)
        temporalFeatures = self.residualBlocks_time(resInputs)
        if self.integrationMethod == 'add':
            linearInputs = spatialFeatures + temporalFeatures
        elif self.integrationMethod == 'concat':
            # concatenate the channels
            linearInputs = torch.cat(
                (spatialFeatures, temporalFeatures), dim=1)
        else:
            warnings.warn(
                f'Unknown concatenation method. Defaulting to addtion.', ECGNetworkWarning)
            linearInputs = spatialFeatures + temporalFeatures
        linearInputs = self.integrationBlock(linearInputs)
        output = self.finalLayer(linearInputs)
        return output

    def _generateResidualBlocks(self, numLayers, in_channels, out_channels, kernel_size, dropout, bias, padding, blockType):
        layerList = []
        for layerIx in range(numLayers):
            if blockType == 'Temporal':
                layerList.append(temporalResidualBlock(in_channels=in_channels[layerIx],
                                                       out_channels=out_channels[layerIx],
                                                       kernel_size=kernel_size[layerIx],
                                                       dropout=dropout[layerIx],
                                                       bias=bias,
                                                       padding=padding[layerIx]))
            if blockType == 'Spatial':
                layerList.append(spatialResidualBlock(in_channels=in_channels[layerIx],
                                                      out_channels=out_channels[layerIx],
                                                      kernel_size=kernel_size[layerIx],
                                                      dropout=dropout[layerIx],
                                                      bias=bias,
                                                      padding=padding[layerIx]))
        return nn.Sequential(*layerList)


class ECGPatchDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator for ECG signals as 2D maps.
    Input: (B, 1, 8, 5000)
    """

    def __init__(self, conv_channels=[64, 128, 256],
                 kernels=[(3, 15), (3, 15), (3, 15), (3, 15)],
                 strides=[(1, 4), (2, 4), (2, 4), (1, 1)],
                 paddings=[(1, 7), (1, 7), (1, 7), (1, 7)]):
        super().__init__()
        self.im_channels = 1
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i+1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i != 0 else True),
                nn.BatchNorm2d(
                    layers_dim[i+1]) if i != len(layers_dim)-2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim)-2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 1, 8, 5000)
    disc = ECGPatchDiscriminator()
    out = disc(x)
    print(out.shape)
