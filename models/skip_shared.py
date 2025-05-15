import torch
import torch.nn as nn
from .common import *

#from your_project.models import conv, bn, act  # wherever your conv/bn/act helpers live

class SkipEncoder(nn.Module):
    def __init__(
        self,
        num_input_channels,
        num_channels_down,
        num_channels_skip,
        filter_size_down=3,
        filter_skip_size=1,
        need_bias=True,
        pad='zero',
        downsample_mode='stride',
        act_fun='LeakyReLU',
    ):
        super().__init__()
        assert len(num_channels_down) == len(num_channels_skip)
        self.n_scales = len(num_channels_down)

        # normalize to lists
        if isinstance(filter_size_down, int):
            filter_size_down = [filter_size_down]*self.n_scales
        if isinstance(downsample_mode, str):
            downsample_mode = [downsample_mode]*self.n_scales

        self.skip_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        in_ch = num_input_channels
        for i in range(self.n_scales):
            # skip branch at this scale
            if num_channels_skip[i] != 0:
                self.skip_convs.append(nn.Sequential(
                    conv(in_ch, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad),
                    bn(num_channels_skip[i]),
                    act(act_fun)
                ))
            else:
                self.skip_convs.append(None)

            # deeper (down) branch at this scale
            down_layers = [
                conv(in_ch,
                     num_channels_down[i],
                     filter_size_down[i],
                     2,  # stride=2 downsampling
                     bias=need_bias,
                     pad=pad,
                     downsample_mode=downsample_mode[i]),
                bn(num_channels_down[i]),
                act(act_fun),
                conv(num_channels_down[i],
                     num_channels_down[i],
                     filter_size_down[i],
                     bias=need_bias,
                     pad=pad),
                bn(num_channels_down[i]),
                act(act_fun)
            ]
            self.down_convs.append(nn.Sequential(*down_layers))
            in_ch = num_channels_down[i]

    def forward(self, x):
        """
        Returns:
          encoded: torch.Tensor, the deepest feature map
          skips:   List[torch.Tensor or None], the skip features at each scale
        """
        skips = []
        out = x
        for i in range(self.n_scales):
            if self.skip_convs[i] is not None:
                skips.append(self.skip_convs[i](out))
            else:
                skips.append(None)
            out = self.down_convs[i](out)
        return out, skips


class SkipDecoder(nn.Module):
    def __init__(
        self,
        num_channels_down,
        num_channels_skip,
        num_channels_up,
        filter_size_up=3,
        need1x1_up=True,
        need_bias=True,
        pad='zero',
        upsample_mode='nearest',
        act_fun='LeakyReLU',
        num_output_channels=1,
        need_sigmoid=True,
    ):
        super().__init__()
        assert len(num_channels_down) == len(num_channels_skip) == len(num_channels_up)
        self.n_scales = len(num_channels_down)
        last = self.n_scales - 1

        # normalize lists
        if isinstance(filter_size_up, int):
            filter_size_up = [filter_size_up]*self.n_scales
        if isinstance(upsample_mode, str):
            upsample_mode = [upsample_mode]*self.n_scales

        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode=upsample_mode[i])
            for i in range(self.n_scales)
        ])
        self.up_convs = nn.ModuleList()
        self.conv1x1s = nn.ModuleList()

        # build one block per scale (to be used in reverse order)
        for i in reversed(range(self.n_scales)):
            skip_ch = num_channels_skip[i]
            deeper_ch = (num_channels_up[i+1] if i < last
                         else num_channels_down[i])
            in_ch = skip_ch + deeper_ch
            out_ch = num_channels_up[i]

            # main 3×3 up-conv
            self.up_convs.append(nn.Sequential(
                conv(in_ch, out_ch, filter_size_up[i], bias=need_bias, pad=pad),
                bn(out_ch),
                act(act_fun),
            ))
            # optional 1×1 refinement
            if need1x1_up:
                self.conv1x1s.append(nn.Sequential(
                    conv(out_ch, out_ch, 1, bias=need_bias, pad=pad),
                    bn(out_ch),
                    act(act_fun),
                ))
            else:
                self.conv1x1s.append(None)

        # final projection to output channels
        self.final_conv = conv(num_channels_up[0], num_output_channels, 1,
                               bias=need_bias, pad=pad)
        self.need_sigmoid = need_sigmoid
        if need_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, encoded, skips):
        out = encoded
        idx = 0
        # walk scales in reverse
        for i in reversed(range(self.n_scales)):
            out = self.upsamples[i](out)
            if skips[i] is not None:
                out = torch.cat([skips[i], out], dim=1)
            out = self.up_convs[idx](out)
            if self.conv1x1s[idx] is not None:
                out = self.conv1x1s[idx](out)
            idx += 1

        out = self.final_conv(out)
        if self.need_sigmoid:
            out = self.sigmoid(out)
        return out


class MultiDecoderSkipNet(nn.Module):
    def __init__(
        self,
        num_input_channels=2,
        num_output_channels=1,
        num_channels_down=[16,32,64,128,128],
        num_channels_up=[16,32,64,128,128],
        num_channels_skip=[4,4,4,4,4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad='zero',
        upsample_mode='nearest',
        downsample_mode='stride',
        act_fun='LeakyReLU',
        need1x1_up=True,
        num_decoders=2
    ):
        super().__init__()
        # shared encoder
        self.encoder = SkipEncoder(
            num_input_channels,
            num_channels_down,
            num_channels_skip,
            filter_size_down,
            filter_skip_size,
            need_bias,
            pad,
            downsample_mode,
            act_fun
        )
        # multiple decoder heads
        self.decoders = nn.ModuleList([
            SkipDecoder(
                num_channels_down,
                num_channels_skip,
                num_channels_up,
                filter_size_up,
                need1x1_up,
                need_bias,
                pad,
                upsample_mode,
                act_fun,
                num_output_channels,
                need_sigmoid
            )
            for _ in range(num_decoders)
        ])

    def forward(self, x):
        encoded, skips = self.encoder(x)
        # each head gets the same features+skips
        return [dec(encoded, skips) for dec in self.decoders]
