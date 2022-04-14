import torch
import torch.nn as nn
from torch import sigmoid

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting):
        super(DoubleConv, self).__init__()

        # group norm fixed/dynamic group numbers
        group, channel, no_affine = norm_setting
        affine = not no_affine

        if group == 0 and channel > 0:
            # dynamically divide groups based on channel number
            n_group = max(1, int(out_ch // channel))
        else:
            # fixed group number
            n_group = group

        norms = nn.ModuleDict(
            {
                "batch": nn.BatchNorm2d(
                    out_ch, momentum=0.005, affine=True, track_running_stats=True
                ),
                "group": nn.GroupNorm(n_group, out_ch, affine=affine),
                "instance": nn.InstanceNorm2d(out_ch, affine=affine),
            }
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            norms[norm],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
            norms[norm],
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting):
        super(ResBlock, self).__init__()

        # group norm fixed/dynamic group numbers
        group, channel, no_affine = norm_setting

        if group == 0 and channel > 0:
            # dynamically divide groups based on channel number
            n_group = max(1, int(out_ch // channel))
        else:
            # fixed group number
            n_group = group

        norms = nn.ModuleDict(
            {
                "batch": nn.BatchNorm2d(out_ch, momentum=0.005),
                "group": nn.GroupNorm(n_group, out_ch),
                "instance": nn.InstanceNorm2d(out_ch, affine=False),
            }
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,
                      stride=1, bias=True), norms[norm]
        )

        self.double_conv = DoubleConv(in_ch, out_ch, norm, norm_setting)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.residual(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return out

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting):
        super(type(self), self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, norm, norm_setting)

    def forward(self, x):
        x = self.conv(x)
        return x
        
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting, conv_type, down_type):
        super(type(self), self).__init__()

        downs = nn.ModuleDict(
            {
                "maxpool": nn.MaxPool2d(2),
                "avgpool": nn.AvgPool2d(2),
                "stride": nn.Conv2d(in_ch, in_ch, 3, 2, padding=1),
            }
        )

        convs = nn.ModuleDict(
            {
                "unet": DoubleConv(in_ch, out_ch, norm, norm_setting),
                "resnet": ResBlock(in_ch, out_ch, norm, norm_setting),
            }
        )

        self.down_cov = nn.Sequential(downs[down_type], convs[conv_type])

    def forward(self, x):
        x = self.down_cov(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting, up_type):
        super(type(self), self).__init__()

        if up_type == "transconv":
            self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        elif up_type == "upscale":
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            )

        else:
            raise ValueError(
                f"unknown up_type {up_type}, acceptable transconv, upscale"
            )

        self.conv = DoubleConv(in_ch, out_ch, norm, norm_setting)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, norm, norm_setting, out_ch, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()
        
        # group norm fixed/dynamic group numbers
        group, channel, no_affine = norm_setting
        affine = not no_affine

        if group == 0 and channel > 0:
            # dynamically divide groups based on channel number
            n_group = max(1, int(out_ch // channel))
        else:
            # fixed group number
            n_group = group

        norms = nn.ModuleDict(
            {
                "batch": nn.BatchNorm2d(n_coefficients, momentum=0.005, affine=True, track_running_stats=True),
                "group": nn.GroupNorm(n_group, n_coefficients, affine=affine),
                "instance": nn.InstanceNorm2d(n_coefficients, affine=affine),
            }
        )

        norms_f = nn.ModuleDict(
            {
                "batch": nn.BatchNorm2d(1, momentum=0.005, affine=True, track_running_stats=True),
                "group": nn.GroupNorm(1, 1, affine=affine),
                "instance": nn.InstanceNorm2d(1, affine=affine),
            }
        )


        self.W_gate = nn.Sequential(
            nn.Conv2d(out_ch, n_coefficients, 1, 1, padding=0),
            norms[norm],
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(out_ch, n_coefficients, 1, 1, padding=0),
            norms[norm],
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, 1, 1, padding=0),
            norms_f[norm],
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection*psi
        return out

class Up_att(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting, up_type):
        super(type(self), self).__init__()

        if up_type == "transconv":
            self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        elif up_type == "upscale":
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            )

        else:
            raise ValueError(
                f"unknown up_type {up_type}, acceptable transconv, upscale"
            )
            
        
        self.conv = DoubleConv(in_ch, out_ch, norm, norm_setting)
        self.attention = AttentionBlock(norm, norm_setting, out_ch, out_ch//2)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x2 = self.attention(x1,x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet_module(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        hidden,
        norm,
        norm_setting,
        conv_type,
        down_type,
        up_type,
        att,
        deeper,
    ):
        super(UNet_module, self).__init__()
        
        self.deeper = deeper
        self.att = att

        self.inc = InConv(n_channels, hidden, norm, norm_setting)

        self.down1 = Down(hidden, hidden * 2, norm,
                          norm_setting, conv_type, down_type)

        if deeper:
            self.down2 = Down(
                hidden * 2, hidden * 4, norm, norm_setting, conv_type, down_type
            )
            
            if att:
                self.up2 = Up_att(hidden * 4, hidden * 2, norm, norm_setting, up_type)
            else:
                self.up2 = Up(hidden * 4, hidden * 2, norm, norm_setting, up_type)

        if att:
            self.up1 = Up_att(hidden * 2, hidden , norm, norm_setting, up_type)
        else:
            self.up1 = Up(hidden * 2, hidden , norm, norm_setting, up_type)
            
        self.outc = OutConv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)

        if self.deeper:
            x3 = self.down2(x2)
            x = self.up2(x3, x2)
            x = self.up1(x, x1)
        else:
            x = self.up1(x2, x1)

        x = self.outc(x)
        x = sigmoid(x)
        return x
                            
class WrappedModel(nn.Module):
    def __init__(self, network):
        super(type(self), self).__init__()
        self.module = network

    def forward(self, *x):
        return self.module(*x)


