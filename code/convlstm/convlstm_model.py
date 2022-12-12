import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim,  kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=4 * hidden_dim, kernel_size=kernel_size,
                              padding=pad)

    def initialize(self, inputs):
        N, _, H, W = inputs.size()
        self.hidden_state = torch.zeros(N, self.hidden_dim, H, W).cuda()
        self.cell_state = torch.zeros(N, self.hidden_dim, H, W).cuda()
        self.memory_state = torch.zeros(N, self.hidden_dim, H, W).cuda()

    def forward(self, inputs, first_step=False):
        if first_step:
            self.initialize(inputs)

        combined = torch.cat([inputs, self.hidden_state], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        self.cell_state = f * self.cell_state + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)


        return self.hidden_state


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,  kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, 3, stride=1, padding=1),
            torch.nn.BatchNorm3d(1),
        )

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            layers.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                         kernel_size=kernel_size))
        self.layers = nn.ModuleList(layers)

        self.conv_output = nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=1)

    def forward(self, input_x, input_frames=7, future_frames=1, output_frames=7,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        x = self.conv1(input_x)
        x1 = x.permute(0, 2, 3, 4, 1)
        input_x = x1.permute(0, 1, 4, 2, 3).contiguous()
        print(x1.shape)

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames:
                input_ = input_x[:, t]
            elif not teacher_forcing:
                input_ = outputs[t - 1]
            first_step = (t == 0)
            input_ = input_.float()

            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)

        outputs = [x for x in outputs if x is not None]

        if train:
            assert len(outputs) == output_frames
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1)[:, :, 0]
        outputs1 = outputs[:,-future_frames:,:,:]

        return outputs1
