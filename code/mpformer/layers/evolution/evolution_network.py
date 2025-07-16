import torch.nn.functional as F
from mpformer.layers.evolution.module import *
from mpformer.layers.smt.smtseg import SMTWrapper
from mpformer.layers.adapters.bottleneck_adapter import BottleneckAdapter

class Evolution_Network(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=64, bilinear=True, net="smt"):
        """
        Evolution Network with dynamic architecture based on `net` parameter.
        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            base_c (int): Base number of channels for UNet.
            bilinear (bool): Whether to use bilinear upsampling.
            net (str): Architecture type, either "unet", "mvit", or "smt".
        """
        super(Evolution_Network, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.net_type = net.lower()  # Convert to lowercase for consistency

        if self.net_type == "unet":
            # UNet-based architecture
            self.inc = DoubleConv(n_channels, base_c)
            self.down1 = Down(base_c * 1, base_c * 2)
            self.down2 = Down(base_c * 2, base_c * 4)
            self.down3 = Down(base_c * 4, base_c * 8)
            factor = 2 if bilinear else 1
            self.down4 = Down(base_c * 8, base_c * 16 // factor)

            self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
            self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
            self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
            self.up4 = Up(base_c * 2, base_c * 1, bilinear)
            self.outc = OutConv(base_c * 1, n_classes)

            self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
            self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
            self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
            self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
            self.outc_v = OutConv(base_c * 1, n_classes * 2)
            
        elif self.net_type == "smt":
            # SMT-based architecture
            self.inc = DoubleConv(n_channels, base_c)
            self.adapter = BottleneckAdapter(in_channels=base_c, bottleneck_dim=base_c // 4)
            self.smt = SMTWrapper(num_classes=32, pretrained=None)
            self.outc = OutConv(base_c * 1, n_classes)
            self.outc_v = OutConv(base_c * 1, n_classes * 2)

        else:
            raise ValueError(f"Unsupported net type: {net}. Use 'unet' or 'mvit'.")

        # Shared parameters
        self.gamma = nn.Parameter(torch.zeros(1, n_classes, 1, 1), requires_grad=True)

    def forward(self, x):
        if self.net_type == "unet":
            # Forward pass for UNet-based architecture
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x) * self.gamma

            v = self.up1_v(x5, x4)
            v = self.up2_v(v, x3)
            v = self.up3_v(v, x2)
            v = self.up4_v(v, x1)
            v = self.outc_v(v)
        
        elif self.net_type == "smt":
            # Forward pass for SMT-based architecture
            x1 = self.inc(x)                 # DoubleConv
            x2 = self.adapter(x1)            # Bottleneck Adapter
            x3 = self.smt(x2)                # SMTWrapper
            x = self.outc(x3[0]) * self.gamma
            v = self.outc_v(x3[0])

        return x, v
