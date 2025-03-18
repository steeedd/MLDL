from torch import nn

# Define the Custom Neural Network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Define the Layers of the Neural Network

        # Input Shape BEFORE BLOCK #1:
        # [B, 3, 224, 224]

        # BLOCK #1:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # Output Shape AFTER conv1 (+ relu1): [B, 64, 224, 224] BECAUSE of out_channels=64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # MaxPool2d -> Reduce the Spatial Size (Height and Width)
                                                                        # of the Image

        # Input Shape BEFORE BLOCK #2:
        # [B, 64, 112, 112] BECAUSE of stride=2 (Height/2, Width/2)

        # BLOCK #2:
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Output Shape AFTER conv2 (+ relu2): [B, 128, 112, 112] BECAUSE of out_channels=128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Shape BEFORE BLOCK #3:
        # [B, 128, 56, 56] BECAUSE of stride=2 (Height/2, Width/2)

        # BLOCK #3:
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        # Output Shape AFTER conv3 (+ relu3): [B, 256, 56, 56] BECAUSE of out_channels=256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Shape BEFORE BLOCK #4:
        # [B, 256, 28, 28] BECAUSE of stride=2 (Height/2, Width/2)

        # BLOCK #4:
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        # Output Shape AFTER conv4 (+ relu4): [B, 256, 28, 28] BECAUSE of out_channels=256
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Shape BEFORE FINAL BLOCK:
        # [B, 256, 14, 14] BECAUSE of stride=2 (Height/2, Width/2)

        # FINAL BLOCK:
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Output Shape AFTER global_pool: [B, 256, 1, 1]
        self.flatten = nn.Flatten()
        # Output Shape AFTER flatten: [B, 256]
        self.fc1 = nn.Linear(in_features=256, out_features=200)


    def forward(self, x):
        # Apply ALL Defined in __init__
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = self.flatten(x)         # [B, 256]
        x = self.fc1(x)             # [B, 200]
        return x