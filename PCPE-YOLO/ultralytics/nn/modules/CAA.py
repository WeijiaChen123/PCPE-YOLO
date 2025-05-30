import torch.nn as nn
import torch


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""


    default_act = nn.SiLU()  # default activation
    
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
    
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()


    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x


class Bottleneck(nn.Module):
    """Standard bottleneck."""


    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2


    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_CAA(nn.Module):
    """CSP Bottleneck with 2 convolutions followed by CAA attention."""


    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with CAA attention."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.caa = CAA(c2)


    def forward(self, x):
        """Forward pass through C2f layer followed by CAA attention."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.caa(out)


    def forward_split(self, x):
        """Forward pass using split() instead of chunk() followed by CAA attention."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.caa(out)
