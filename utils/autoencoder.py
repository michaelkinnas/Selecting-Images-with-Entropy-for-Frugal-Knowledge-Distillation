from torch.nn import Conv2d, ReLU, MaxPool2d, Sequential, ConvTranspose2d, Sigmoid, Module, MSELoss
from torch import optim
from tqdm import tqdm

class Autoencoder(Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential(
            Conv2d(3, 16, 3, padding=1),   # 32x32x3 -> 32x32x16
            ReLU(True),
            MaxPool2d(2, 2),               # 32x32x16 -> 16x16x16
            Conv2d(16, 8, 3, padding=1),   # 16x16x16 -> 16x16x8
            ReLU(True),
            MaxPool2d(2, 2),               # 16x16x8 -> 8x8x8
            Conv2d(8, 8, 3, padding=1),    # 8x8x8 -> 8x8x8
            ReLU(True),
            MaxPool2d(2, 2)                # 8x8x8 -> 4x4x8
        )
        self.decoder = Sequential(
            ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1),    # 4x4x8 -> 8x8x8
            ReLU(True),
            ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),   # 8x8x8 -> 16x16x16
            ReLU(True),
            ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # 16x16x16 -> 32x32x3
            Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def train_autoencoder(autoencoder, trainloader, device):
    '''
    Select top N images using compressed feature vector entropy criterion
    '''
    autoencoder.to(device)
    # Define loss function and optimizer
    mse_criterion = MSELoss()
    adam_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in tqdm(trainloader, total=len(trainloader), desc=f"Training autoencoder, epoch {epoch+1} of {num_epochs}"):
            inputs, _ = data
            inputs = inputs.to(device)
            adam_optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = mse_criterion(outputs, inputs)
            loss.backward()
            adam_optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss / len(trainloader)))
