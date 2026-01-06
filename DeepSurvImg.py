# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from unet3Dmodel import UNet3DCut, UNet3D
import torch
import torch.nn as nn
import torchvision.models
# import pytorchvideo.models
# import torchsummary
import resnet3d
# from models.resnet_simclr import ResNetSimCLR
from monai.networks.nets import resnet18, DenseNet169

def mexHat(x): return torch.mul((1-x**2),torch.exp(-0.5*x**2))
def gh(x, w, b, t, a):
  return torch.matmul(torch.matmul(mexHat(x - t),torch.diag(a)),torch.transpose(w, 0, 1)) + b
def sai(x, n, m):
  return (2**(-0.5*m))*mexHat((2**m)*x - n)


# def replace_relu_with_wavelet(model):
#     # For activations in ResNet Blocks
#     model[0].activation = WN(in_features=64 * 128 * 50 * 50)
#     wavelet_sizes = [64 * 128 * 25 * 25, 64 * 128 * 25 * 25, 256 * 128 * 25 * 25]
#     for i in range(3):
#         model[1].res_blocks[i].branch2.act_a = WN(in_features=wavelet_sizes[0])
#         model[1].res_blocks[i].branch2.act_b = WN(in_features=wavelet_sizes[1])
#         model[1].res_blocks[i].activation = WN(in_features=wavelet_sizes[2])
#     return model
def replace_relu_with_wavelet(model):
    # For activations in ResNet Blocks
    model.relu = WN(in_features=0)
    model.layer1[0].relu = WN(in_features=0)
    model.layer2[0].relu = WN(in_features=0)
    model.layer3[0].relu = WN(in_features=0)
    model.layer4[0].relu = WN(in_features=0)

    return model
def replace_relu_with_wavelet_unet(model):
    # For activations in ResNet Blocks
    wavelet_sizes = [0,0,0,0,0,0,0,0,0,0,0,0]
    # wavelet_sizes = [128*50*50,128*50*50, 64*25*25, 64*25*25,
    #                  32*12*12, 32*12*12, 16*6*6, 16*6*6]
    # wavelet_sizes = [50*50,50*50, 25*25, 25*25,
    #                  12*12, 12*12, 6*6, 6*6]
    counter = 0
    for i in range(6):
        for j in range(2):
            model.encoders[i].basic_module[j][2] = WN(in_features=wavelet_sizes[counter])
            counter += 1
    return model


def replace_relu_with_wavelet_fc(model):
    # For linear layers
    wavelet_sizes = [0,0,0,0]
    for ii, i in enumerate([2, 5, 8, 11]): #for classsurve:[4, 7, 10, 13]
        if hasattr(model, 'fc'):
            model.fc[i] = WN(wavelet_sizes[ii])
        else:
            model[i] = WN(wavelet_sizes[ii])

    return model
def replace_relu_with_wavelet_fc_unet(model):
    # For linear layers
    wavelet_sizes = [0,0,0,0]
    for ii, i in enumerate([3, 6, 9, 12]): #for classsurve:[4, 7, 10, 13]
        if hasattr(model, 'fc'):
            model.fc[i] = WN(wavelet_sizes[ii])
        else:
            model[i] = WN(wavelet_sizes[ii])

    return model
class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class
        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()

        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.
        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class DynamicMLP(nn.Module):
    def __init__(self, num_layers, input_dim=128, output_dim=10, dropout_prob=0.5, use_tanh=False):
        super(DynamicMLP, self).__init__()

        # Check if input_dim can be halved num_layers times
        if input_dim % (2 ** num_layers) != 0:
            raise ValueError("Input dimension must be divisible by 2 a sufficient number of times to allow halving.")

        layers = []
        current_dim = input_dim

        # Hidden layers
        for _ in range(num_layers):
            next_dim = current_dim // 2
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            current_dim = next_dim  # Update current_dim for next layer

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # Optional Tanh activation
        if use_tanh:
            layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def create_dynamic_mlp(num_layers, input_dim=128, output_dim=10, dropout_prob=0.5, use_tanh=False):
    """
    Creates a PyTorch model with the specified number of linear layers, where each
    subsequent layer has half the number of nodes as the previous one.

    Args:
        num_layers (int): Number of linear layers.
        input_dim (int, optional): Dimension of input features. Default is 128.
        output_dim (int, optional): Dimension of output layer. Default is 10.
        dropout_prob (float, optional): Dropout probability. Default is 0.5.
        use_tanh (bool, optional): Whether to add a Tanh activation function to the output. Default is False.

    Returns:
        nn.Module: A PyTorch model.
    """
    return DynamicMLP(num_layers, input_dim, output_dim, dropout_prob, use_tanh)


class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config, modeltype = "Resnet"):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        # self.drop = config['drop']
        # self.norm = config['norm']
        # self.dims = config['dims']
        self.activation = config['activation']
        self.modeltype = config['modeltype']
        # builds network
        self.model = self._build_network(modeltype)

    def _build_network(self, modeltype = "Resnet"):
        ''' Performs building networks according to parameters'''
        if self.modeltype=="2D":
            model = torchvision.models.resnet18()
            model.fc = nn.Sequential(torch.nn.Linear(512,256, bias=True),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     torch.nn.Linear(256,128, bias=True),
                                     nn.ReLU(),
                                     torch.nn.Linear(128, 64, bias=True),
                                     nn.ReLU(),
                                     torch.nn.Linear(64, 1, bias=True),
                                     nn.Tanh())
            model.conv1 = torch.nn.Conv2d(1,64,(7,7),(2,2), (3,3), bias=False)
        elif self.modeltype=="3D":
            if modeltype == "Resnet":
                model = resnet3d.generate_model(model_depth=10,n_classes=1,n_input_channels=1,shortcut_type='B',conv1_t_size=7,conv1_t_stride=1,no_max_pool=False,widen_factor=1.0)
                # model.fc = nn.Sequential(*[
                #     nn.Linear(512, 256), nn.LayerNorm(256),
                #     nn.ReLU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64), nn.LayerNorm(64),
                #     nn.ReLU(), nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
                #     nn.Linear(32, 1)], nn.Tanh())
                model.fc = nn.Sequential(*[nn.Linear(512,1), nn.Tanh()])
            elif modeltype == "Densenet":
                model = DenseNet169(pretrained=False, out_channels=1, in_channels=1, spatial_dims=3)
                model.class_layers.out = torch.nn.Sequential(
                    torch.nn.Linear(in_features=model.class_layers.out.in_features, out_features=1), torch.nn.Tanh())

            # model = UNet3DCut(in_channels=1, out_channels=1, f_maps=16, num_levels=6)
            # # Unet
            # self.mlp = nn.Sequential(*[
            #     nn.Flatten(start_dim=1), nn.Linear(512, 256), nn.LayerNorm(256),
            #     nn.ReLU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64), nn.LayerNorm(64),
            #     nn.ReLU(), nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            #     nn.Linear(32, 1)])

            # if self.activation == "wavelet":
            #     # model = replace_relu_with_wavelet_unet(model)
            #     model = replace_relu_with_wavelet(model)
            #     # self.mlp = replace_relu_with_wavelet_fc_unet(self.mlp)
            #     model.fc = replace_relu_with_wavelet_fc(model.fc) # also works for new resnet
        return model

    def forward(self, X):
        pred = self.model(X)
        # X = X.view(-1, 128, 576)
        # pred = self.mlp(pred)
        return pred

class DeepSurvSSLOnly(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config, activation=None):
        super(DeepSurvSSLOnly, self).__init__()
        # parses parameters of network from configuration
        self.config = config
        self.activation = config['activation']
        self.modeltype = config['modeltype']
        # builds network
        self.classes = config["num_classes"]
        self.starting_feats = config['start_feats']
        self.activation = activation
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''

        if self.modeltype == "SSL_Pre":
            layer_sizes = [self.starting_feats, self.starting_feats//2, self.starting_feats//4, self.starting_feats//8,
                           self.starting_feats//16, self.starting_feats//32, self.starting_feats//64]
            model = nn.Sequential(*[
                nn.Linear(layer_sizes[0], layer_sizes[1]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[1]), nn.ReLU(),
                nn.Linear(layer_sizes[1], layer_sizes[2]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[2]), nn.ReLU(),
                nn.Linear(layer_sizes[2], layer_sizes[3]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[3]), nn.ReLU(),
                nn.Linear(layer_sizes[3], layer_sizes[4]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[4]),nn.ReLU(),
                nn.Linear(layer_sizes[4], layer_sizes[5]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[5]), nn.ReLU(),
                nn.Linear(layer_sizes[5], layer_sizes[6]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[6]), nn.ReLU(),
                nn.Linear(layer_sizes[6], self.classes)])
        elif self.modeltype == "SSL_Full" or self.modeltype == "SSL_Retrain":
            ssl_model = ResNetSimCLR(base_model="resnet18", out_dim=self.starting_feats)

            if self.modeltype == "SSL_Full":
                model_loc = self.config["model_loc"]
                checkpoint = torch.load(model_loc)
                ssl_model.load_state_dict(checkpoint["state_dict"])
                for param in ssl_model.parameters():
                    param.requires_grad = False

            layer_sizes = [self.starting_feats, self.starting_feats//2, self.starting_feats//4, self.starting_feats//8,
                           self.starting_feats//16, self.starting_feats//32, self.starting_feats//64]
            model = nn.Sequential(*[ssl_model,
                nn.Linear(layer_sizes[0], layer_sizes[1]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[1]), nn.ReLU(),
                nn.Linear(layer_sizes[1], layer_sizes[2]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[2]), nn.ReLU(),
                nn.Linear(layer_sizes[2], layer_sizes[3]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[3]), nn.ReLU(),
                nn.Linear(layer_sizes[3], layer_sizes[4]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[4]), nn.ReLU(),
                nn.Linear(layer_sizes[4], layer_sizes[5]), nn.Dropout(p=self.config["drop"]), nn.LayerNorm(layer_sizes[5]), nn.ReLU(),
                nn.Linear(layer_sizes[5], layer_sizes[6]), nn.Dropout(p=self.config["drop"]),nn.LayerNorm(layer_sizes[6]), nn.ReLU(),
                nn.Linear(layer_sizes[6], self.classes)])
        else:
            print("Model type invalid")
            model = None

        return model

    def forward(self, X):
        pred = self.model(X)
        if self.activation:
            pred = self.activation(pred)
        return pred


class DeepSurvHybrid(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurvHybrid, self).__init__()
        # parses parameters of network from configuration
        # self.drop = config['drop']
        # self.norm = config['norm']
        # self.dims = config['dims']
        self.activation = config['activation']
        self.modeltype = config['modeltype']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''

        self.seg_model = UNet3D(in_channels=1, out_channels=1, f_maps=16, num_levels=3)
        self.encoder = resnet3d.generate_model(model_depth=10,n_classes=1,n_input_channels=1,shortcut_type='B',conv1_t_size=7,conv1_t_stride=1,no_max_pool=False,widen_factor=1.0)

        self.encoder.fc = nn.Sequential(*[
            nn.Linear(512, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64), nn.LayerNorm(64),
            nn.ReLU(), nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, 1)])

        if self.activation == "wavelet":
            # model = replace_relu_with_wavelet_unet(model)
            model = replace_relu_with_wavelet(model)
            # self.mlp = replace_relu_with_wavelet_fc_unet(self.mlp)
            model.fc = replace_relu_with_wavelet_fc(model.fc) # also works for new resnet
        return model

    def forward(self, X):
        pred = self.seg_model(X)
        pred = self.encoder(pred)
        return pred
    def get_seg(self, X):
        pred = self.seg_model(X)
        return pred


class DeepSurvBranched(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurvBranched, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self.modeltype = config['modeltype']
        self.branch1_file = config['branch1_file']
        self.branch2_file = config['branch2_file']
        # builds network
        self.branch1, self.branch2, self.mlp = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        if self.modeltype=="2D":
            model = torchvision.models.resnet18()
            model.fc = nn.Sequential(torch.nn.Linear(512,256, bias=True),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     torch.nn.Linear(256,128, bias=True),
                                     nn.ReLU(),
                                     torch.nn.Linear(128, 64, bias=True),
                                     nn.ReLU(),
                                     torch.nn.Linear(64, 1, bias=True),
                                     nn.Tanh())
            model.conv1 = torch.nn.Conv2d(1,64,(7,7),(2,2), (3,3), bias=False)
        elif self.modeltype=="3D":
            # Create empty branch models
            branch1 = nn.Sequential(*[pytorchvideo.models.create_res_basic_stem(in_channels=1, out_channels=64, conv_kernel_size=(1, 7, 7), conv_padding = (0, 3, 3)),
                                    pytorchvideo.models.resnet.create_res_stage(depth=3,dim_out=256, dim_in=64, dim_inner=64,
                                                                                conv_a_kernel_size=(1,1,1),
                                                                                conv_a_stride=(1,1,1),
                                                                                conv_a_padding=(0,0,0),
                                                                                conv_b_kernel_size=(1,3,3),
                                                                                conv_b_stride=(1,1,1),
                                                                                bottleneck=pytorchvideo.models.resnet.create_bottleneck_block)])
            branch2 = nn.Sequential(*[pytorchvideo.models.create_res_basic_stem(in_channels=1, out_channels=64, conv_kernel_size=(1, 7, 7), conv_padding = (0, 3, 3)),
                                    pytorchvideo.models.resnet.create_res_stage(depth=3,dim_out=256, dim_in=64, dim_inner=64,
                                                                                conv_a_kernel_size=(1,1,1),
                                                                                conv_a_stride=(1,1,1),
                                                                                conv_a_padding=(0,0,0),
                                                                                conv_b_kernel_size=(1,3,3),
                                                                                conv_b_stride=(1,1,1),
                                                                            bottleneck=pytorchvideo.models.resnet.create_bottleneck_block)])
            branch1.fc = nn.Sequential(
                *[nn.AvgPool3d(kernel_size=3, stride=3), nn.Flatten(start_dim=1), nn.Linear(12288, 3072),
                  nn.LayerNorm(3072), nn.ReLU(), nn.Linear(3072, 1024), nn.LayerNorm(1024), nn.ReLU(),
                  nn.Linear(1024, 512), nn.LayerNorm(512),
                  nn.ReLU(), nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()])
            branch2.fc = nn.Sequential(
                *[nn.AvgPool3d(kernel_size=3, stride=3), nn.Flatten(start_dim=1), nn.Linear(12288, 3072),
                  nn.LayerNorm(3072), nn.ReLU(), nn.Linear(3072, 1024), nn.LayerNorm(1024), nn.ReLU(),
                  nn.Linear(1024, 512), nn.LayerNorm(512),
                  nn.ReLU(), nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()])

            mlp = nn.Sequential(
                *[nn.AvgPool3d(kernel_size=3, stride=3), nn.Flatten(start_dim=1), nn.Linear(24576, 3072),
                  nn.LayerNorm(3072), nn.ReLU(), nn.Linear(3072, 1024), nn.LayerNorm(1024), nn.ReLU(),
                  nn.Linear(1024, 512), nn.LayerNorm(512),
                  nn.ReLU(), nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()])
            if self.activation == 'wavelet':
                branch1 = replace_relu_with_wavelet(branch1)
                # branch1.fc = nn.Sequential(*[nn.AvgPool3d(kernel_size=3,stride=3),nn.Flatten(start_dim=1),nn.Linear(12288,3072),
                #                        nn.LayerNorm(3072),WN(3072),nn.Linear(3072,1024),nn.LayerNorm(1024),WN(1024),nn.Linear(1024,512),
                #                        nn.LayerNorm(512),WN(512),nn.Linear(512,128), nn.LayerNorm(128),WN(128),nn.Linear(128,1),
                #                        nn.Tanh()])
                branch2 = replace_relu_with_wavelet(branch2)
                # branch2.fc = nn.Sequential(*[nn.AvgPool3d(kernel_size=3,stride=3),nn.Flatten(start_dim=1),nn.Linear(12288,3072),
                #                        nn.LayerNorm(3072),WN(3072),nn.Linear(3072,1024),nn.LayerNorm(1024),WN(1024),nn.Linear(1024,512),
                #                        nn.LayerNorm(512),WN(512),nn.Linear(512,128), nn.LayerNorm(128),WN(128),nn.Linear(128,1),
                #                        nn.Tanh()])
                # mlp = nn.Sequential(*[nn.AvgPool3d(kernel_size=3,stride=3),nn.Flatten(start_dim=1),nn.Linear(24576,3072),
                #                        nn.LayerNorm(3072),WN(3072),nn.Linear(3072,1024),nn.LayerNorm(1024),WN(1024),nn.Linear(1024,512),
                #                        nn.LayerNorm(512),WN(512),nn.Linear(512,128), nn.LayerNorm(128),WN(128),nn.Linear(128,1),
                #                        nn.Tanh()])
                # mlp = nn.Sequential(*[nn.AvgPool3d(kernel_size=3,stride=3),nn.Flatten(start_dim=1),nn.Linear(24576,3072),
                #                            nn.LayerNorm(3072),nn.ReLU(),nn.Linear(3072,1024),nn.LayerNorm(1024),nn.ReLU(),nn.Linear(1024,512), nn.LayerNorm(512),
                #                            nn.ReLU(),nn.Linear(512,128),nn.LayerNorm(128), nn.ReLU(),nn.Linear(128,1),nn.Tanh()])
                branch1 = replace_relu_with_wavelet_fc(branch1)
                branch2 = replace_relu_with_wavelet_fc(branch2)
                mlp = replace_relu_with_wavelet_fc(mlp)

            # Load model and freeze its convolutional layers
            temp1 = torch.load(self.branch1_file)['model']
            temp1 = {key.replace("model.", ""): value for key, value in temp1.items() if key.startswith("model.")}
            branch1.load_state_dict(temp1)
            branch1.fc = nn.Sequential()
            # for param in branch1.parameters():
            #     param.requires_grad = False

            temp2 = torch.load(self.branch2_file)['model']
            temp2 = {key.replace("model.", ""): value for key, value in temp2.items() if key.startswith("model.")}
            branch2.load_state_dict(temp2)
            branch2.fc = nn.Sequential()
            # for param in branch2.parameters():
            #     param.requires_grad = False

        return branch1, branch2, mlp

    # def forward(self, X1,X2):
    #     out1 = self.branch1(X1)
    #     out2 = self.branch2(X2)
    #     out = torch.cat((out1,out2), dim=1)
    #
    #     return self.mlp(out)
    def forward(self, X):
        out1 = self.branch1(X)
        out2 = self.branch2(X)
        out = torch.cat((out1,out2), dim=1)

        return self.mlp(out)

class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        reg_loss = 1 / torch.std(risk_pred)
        return neg_log_loss + l2_loss


class WN(nn.Module):

    def __init__(self, in_features, n=None, m=None):
        super(WN, self).__init__()
        self.in_features = in_features
        if n == None:
            self.shift = nn.parameter.Parameter(torch.tensor(0.0))
        else:
            self.shift = nn.parameter.Parameter(torch.tensor(n))

        if m == None:
            self.scale = nn.parameter.Parameter(torch.tensor(0.0))
        else:
            self.scale = nn.parameter.Parameter(torch.tensor(m))

        self.shift.requiresGrad = True
        self.scale.requiresGrad = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return sai(input, self.shift, self.scale)


class ClassSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config, numclasses):
        super(ClassSurv, self).__init__()
        # parses parameters of network from configuration
        self.activation = config['activation']
        self.modeltype = config['modeltype']
        self.numclasses = numclasses
        # builds network
        self.model = self._build_network()


    def _build_network(self):
        ''' Performs building networks according to parameters'''
        model = resnet3d.generate_model(model_depth=10, n_classes=1, n_input_channels=1, shortcut_type='B',
                                        conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, widen_factor=1.0)
        # model.fc = nn.Sequential(*[
        #     nn.Linear(512, 256), nn.LayerNorm(256),
        #     nn.ReLU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64), nn.LayerNorm(64),
        #     nn.ReLU(), nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
        #     nn.Linear(32, self.numclasses), nn.Softmax()])
        model.fc = nn.Linear(in_features=512, out_features=self.numclasses)
        return model

    def forward(self, X):
        return self.model(X)

class CoxLoss(nn.Module):
    def __init__(self, config):
        super(CoxLoss, self).__init__()
        self.L2_reg = Regularization(order=2, weight_decay=config['l2_reg'])
        self.L1_reg = Regularization(order=1, weight_decay=config['l1_reg'])

    def forward(self, risk_pred, y, e, model = None):
        time_value = torch.squeeze(y)
        event = torch.squeeze(e).bool()
        score = torch.squeeze(risk_pred)

        ix = torch.where(event)

        sel_mat = (time_value[ix] <= time_value.view(-1, 1)).float()

        p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score).view(-1,1), dim=0))

        loss = -torch.mean(p_lik)
        l2_loss = self.L2_reg(model)
        l1_loss = self.L1_reg(model)

        loss = loss + l2_loss + l1_loss

        return loss

class PLLL(nn.Module):
    def forward(self, logits, fail_indicator, ties):
        '''
        fail_indicator: 1 if the sample fails, 0 if the sample is censored.
        logits: raw output from model
        ties: 'noties' or 'efron' or 'breslow'
        '''
        logL = 0
        # pre-calculate cumsum
        cumsum_y_pred = torch.cumsum(logits, 0)
        hazard_ratio = torch.exp(logits)
        cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
        if ties == 'noties':
            log_risk = torch.log(cumsum_hazard_ratio)
            likelihood = logits - log_risk
            # dimension for E: np.array -> [None, 1]
            uncensored_likelihood = likelihood * fail_indicator
            logL = -torch.sum(uncensored_likelihood)
        else:
            raise NotImplementedError()
        # negative average log-likelihood
        observations = torch.sum(fail_indicator, 0)
        return 1.0*logL / observations