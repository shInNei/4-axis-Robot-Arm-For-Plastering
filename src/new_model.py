import torch
import torch.nn as nn

class RobotArmMovement(nn.Module):
    def __init__(self, pos_dim = 12, hidden_dim=256):
        super(RobotArmMovement, self).__init__()
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim

        # CNN for image feature extraction
        self.cnn = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),

          nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),

          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.AdaptiveAvgPool2d((8, 6)),
        )

        # Distance feature extraction
        self.distance_fc = nn.Sequential(
            nn.Linear(self.pos_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Feature projection to hidden_dim
        self.feature_fc = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(64 * 8 * 6 + 256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.pos_dim),
        )

        # LSTM for temporal modeling
        # self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

        # self.output_fc = nn.Linear(hidden_dim, pos_dim)


    def create_mask(self, joint_angles_tensor, padding_value=-1.0):
        # Create a mask: 1 where joint angles are not equal to padding value, 0 where they are
        mask = (joint_angles_tensor != padding_value).float().prod(dim=-1)  # Shape: (batch, max_seq_len)
        return mask

    # def forward(self, image, pos, padding_mask=None):
    #     # Extract features from image and distance
    #     cnn_features = self.cnn(image)
    #     cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Shape: (batch, 128 * 16 * 9)

    #     distance_features = self.distance_fc(pos)  # Shape: (batch, 256)

    #     combined_features = torch.cat((cnn_features, distance_features), dim=1)  # Shape: (batch, 128 * 16 * 9 + 256)
    #     features = self.feature_fc(combined_features)  # Shape: (batch, hidden_dim)

    #     # Repeat features for each timestep to create a sequence
    #     features = features.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # Shape: (batch, max_seq_len, hidden_dim)
    def forward(self, image, pos):
      cnn_features = self.cnn(image)
      cnn_features = cnn_features.view(cnn_features.size(0), -1)

      distance_features = self.distance_fc(pos)

      combined_features = torch.cat((cnn_features, distance_features), dim=1)

      output = self.feature_fc(combined_features)
      # output = self.clip_output(output)
      return output