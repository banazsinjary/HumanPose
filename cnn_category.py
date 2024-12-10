import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# Define CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 224, 224)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.num_classes = num_classes

        # Automatically initialize fully connected layers
        self._initialize_fc_layers(input_shape)

    def _compute_flattened_size(self, x):
        """Helper to compute flattened size for dynamically sized input."""
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.view(x.size(0), -1).size(1)

    def _initialize_fc_layers(self, input_shape):
        """Initialize fully connected layers dynamically."""
        dummy_input = torch.zeros(1, *input_shape)
        self.flattened_size = self._compute_flattened_size(dummy_input)
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    
class ClassificationCNN(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(ClassificationCNN, self).__init__()
        # Use the pre-trained feature extractor
        self.feature_extractor = pretrained_model.cnn  # Keep the CNN part
        
        # Freeze pre-trained layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # # # Unfreeze the last layer for fine-tuning
        # for param in self.feature_extractor.conv3.parameters():
        #     param.requires_grad = True
        
        # project to smaller # of features if necessary (for resnet)
        self.feature_projection = nn.Linear(2052, 256 + 4) 
        
        # New classification head
        self.fc1 = nn.Linear(256 + 4, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 96)
        self.bn2 = nn.BatchNorm1d(96)
        
        # Additional fully connected layer
        self.fc3 = nn.Linear(96, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Output layer
        self.output = nn.Linear(64, num_classes)
    
    def forward(self, x, l_and_s, additional_features=None):
        # Pass through the pre-trained feature extractor
        x = self.feature_extractor(x)
        
        # Concatenate additional features if they exist
        if additional_features is not None:
            x = torch.cat((x, l_and_s, additional_features), dim=1)
        else:
            x = torch.cat((x, l_and_s), dim=1)
        
        x = self.feature_projection(x)

        # Pass through the classification head
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Output layer
        x = self.output(x)
        
        return x
    
class PoseFeatureExtraction(nn.Module):
    def __init__(self):
        super(PoseFeatureExtraction, self).__init__()
        
        # Load pretrained ResNet model
        pose_model = resnet50(pretrained=True)
        
        # Remove the final classification layers to use as a feature extractor
        self.feature_extractor = nn.Sequential(*list(pose_model.children())[:-2])  # Outputs (batch, 2048, h', w')
        
        # Add Global Average Pooling to reduce spatial dimensions to (batch, 2048)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)  # (batch, 2048, h', w')
        x = self.global_avg_pool(x)    # (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)      # Flatten to (batch, 2048)
        return x
    
class ClassCNN_resnet(nn.Module):
    def __init__(self, pretrained_model, num_classes, dropout_rate=.5):
        super(ClassCNN_resnet, self).__init__()
        
        # Use PyTorch Pose for feature extraction
        self.cnn = PoseFeatureExtraction()
        
        # Update the input size to match Pose feature output (2048) + 3
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1075, 512)
        self.bn2 = nn.BatchNorm1d(512)

        #self.fc3 = nn.Linear(563, 256)
        #self.bn3 = nn.BatchNorm1d(256)

        #self.fc4 = nn.Linear(256, 128)
        #self.bn4 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(dropout_rate)

        self.cat_output = nn.Linear(512, num_classes)

    def forward(self, image, loc_and_scale):
        # Extract features using the pose model
        image_features = self.cnn(image)  # Shape: (batch, 2048)
        
        # Concatenate image features with location and scale data
        #combined_features = torch.cat((image_features, loc_and_scale), dim=1)  # Shape: (batch, 2048 + 3)
        #print(image_features.shape)
        
        # Pass through fully connected layers
        x = F.relu(self.bn1(self.fc1(image_features)))
        x = self.dropout(x)
        
        # Concatenate image features with location and scale data
        combined_features = torch.cat((x, loc_and_scale), dim=1)  # Shape: (batch, 2048 + 3)
        
        x = F.relu(self.bn2(self.fc2(combined_features)))
        x = self.dropout(x)
        
       
        #x = F.relu(self.bn3(self.fc3(combined_features)))
        #x = self.dropout(x)
        #x = F.relu(self.bn4(self.fc4(x)))
        #x = self.dropout(x)
        
        classes = self.cat_output(x)
        
        return classes