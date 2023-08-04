"""
The implementation of DQN

Reference:
Paper: Human-level control through deep reinforcement learning
Sub_sub_title: Model Architecture

"""
import torch
import torch.nn as nn

k_frames = 4
game_frame_height = 84
game_frame_width = 84
game_num_actions = 4


class DQN(nn.Module):
    def __init__(self, num_actions=game_num_actions, input_channels=k_frames, height=game_frame_height,
                 width=game_frame_width):
        super(DQN, self).__init__()
        self.first_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU())

        self.second_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.ReLU())

        self.third_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.ReLU())
        """
        # Solution 1
        # conv_out_size = self._get_conv_output_size((input_channels, height, width))

        # self.final_hidden_layer = nn.Sequential(
        #     nn.Linear(conv_out_size, 512),
        #     nn.ReLU()
        # )
        """
        # Solution 2
        self.final_hidden_layer = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.second_hidden_layer(x)
        x = self.third_hidden_layer(x)

        x = torch.flatten(x, start_dim=1)

        x = self.final_hidden_layer(x)
        x = self.output_layer(x)
        return x

    def _get_conv_output_size(self, shape):
        # Function to calculate the output size of the convolutional layers
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.first_hidden_layer(x)
            x = self.second_hidden_layer(x)
            x = self.third_hidden_layer(x)
            print(x.view(1, -1).size())
            print(64 * 7 * 7)
            return x.view(1, -1).size(1)


if __name__ == "__main__":
    demo_model = DQN()
    sample_input = torch.randn(1, k_frames, game_frame_height, game_frame_width)
    output = demo_model(sample_input)
    print(output.size())
