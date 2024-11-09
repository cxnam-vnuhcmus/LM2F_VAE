import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.channel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Spatial Attention
        self.spatial = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_avg_pool(x).squeeze(-1)
        max_out = self.channel_max_pool(x).squeeze(-1)
        channel_att = self.mlp(avg_out) + self.mlp(max_out)
        channel_att = torch.sigmoid(channel_att).unsqueeze(-1)
        x = x * channel_att
        
        # Spatial Attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.cat([max_out, avg_out], dim=1)
        spatial_att = torch.sigmoid(self.spatial(spatial_att))
        x = x * spatial_att
        
        return x
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False,act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        if act=='ReLU':
            self.act = nn.ReLU()
        elif act=='Tanh':
            self.act =nn.Tanh()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()

        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class AdaINLayer(nn.Module):
    def __init__(self, input_nc, modulation_nc):
        super().__init__()

        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):

        # Part 1. generate parameter-free normalized activations
        normalized = self.InstanceNorm2d(input)

        # Part 2. produce scaling and bias conditioned on feature
        modulation_input = modulation_input.view(modulation_input.size(0), -1)
        actv = self.mlp_shared(modulation_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        
        out = normalized * (1 + gamma) + beta
        return out

class AdaIN(torch.nn.Module):

    def __init__(self, input_channel, modulation_channel,kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel)

    def forward(self, x, modulation):

        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x




class SPADELayer(torch.nn.Module):
    def __init__(self, input_channel, modulation_channel, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADELayer, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(input_channel)

        self.conv1 = torch.nn.Conv2d(modulation_channel, hidden_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        self.gamma = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.beta = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input, modulation):
        norm = self.instance_norm(input)

        conv_out = self.conv1(modulation)

        gamma = self.gamma(conv_out)
        beta = self.beta(conv_out)
        
        return norm + norm * gamma + beta


class SPADE(torch.nn.Module):
    def __init__(self, num_channel, num_channel_modulation, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADE, self).__init__()
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.spade_layer_1 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)
        self.spade_layer_2 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)

    def forward(self, input, modulations):
        input = self.spade_layer_1(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_1(input)
        input = self.spade_layer_2(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_2(input)
        return input


def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b, c, h, w = flow.shape
    flow_norm = 2 * torch.cat([flow[:, :1, ...] / (w - 1), flow[:, 1:, ...] / (h - 1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0, 2, 3, 1)
    return deformation


def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """
    b, c, h, w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed


def warping(source_image, deformation):
    r"""warp the input image according to the deformation
    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation)
