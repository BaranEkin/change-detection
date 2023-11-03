import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models.resnet import ResNet, BasicBlock


class ChangeDetNet(pl.LightningModule):
    def __init__(self, train_set, val_set, batch_size, num_classes=2, lr=1e-3):
        super(ChangeDetNet, self).__init__()

        # self.backbone = UNet(num_classes=num_classes)
        self.backbone = ConvBack(num_classes=num_classes)
        self.lr = lr
        self.train_set = train_set
        self.val_set = val_set
        self.bs = batch_size

    def forward(self, x1, x2):
        output1 = self.backbone(x1)
        output2 = self.backbone(x2)
        return output1, output2

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.bs, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_set, batch_size=self.bs, shuffle=False, num_workers=0)

    def training_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        out1, out2 = self.forward(x1, x2)
        target1 = torch.argmax(out1, dim=1)
        target2 = torch.argmax(out2, dim=1)

        clustering_loss = 0.5 * nn.CrossEntropyLoss()(out1, target1) + 0.5 * nn.CrossEntropyLoss()(out2, target2)

        pos_loss = nn.L1Loss(reduction="mean")(out1, out2)
        neg_loss = torch.clamp(2.0 - nn.L1Loss(reduction="mean")(out1, out2), min=0.0)
        # contrastive_loss = torch.mean(torch.abs(pair_label * pos_loss + (1 - pair_label) * neg_loss))

        loss = clustering_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        out1, out2 = self.forward(x1, x2)
        target1 = torch.argmax(out1, dim=1)
        target2 = torch.argmax(out2, dim=1)

        clustering_loss = 0.5 * nn.CrossEntropyLoss()(out1, target1) + 0.5 * nn.CrossEntropyLoss()(out2, target2)

        pos_loss = nn.L1Loss(reduction="mean")(out1, out2)
        neg_loss = torch.clamp(2.0 - nn.L1Loss(reduction="mean")(out1, out2), min=0.0)
        contrastive_loss = torch.mean(torch.abs(pair_label * pos_loss + (1 - pair_label) * neg_loss))

        loss = clustering_loss

        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class CustomResNet34(ResNet):
    def __init__(self):
        super(CustomResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.encoder = CustomResNet34()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, num_classes, kernel_size=4, stride=2, padding=1),
            nn.Softmax()
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class ConvBack(nn.Module):
    def __init__(self, num_classes):
        super(ConvBack, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class PatchChangeDet(pl.LightningModule):
    def __init__(self, train_set, val_set, batch_size, lr=1e-4, feature_size=128, dropout_p=0.3):
        super(PatchChangeDet, self).__init__()
        self.feature_size = feature_size
        self.dropout_p = dropout_p
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(512, self.feature_size),
            nn.Dropout(self.dropout_p),
            # nn.Sigmoid()
            nn.Softmax()
        )

        self.lr = lr
        self.train_set = train_set
        self.val_set = val_set
        self.bs = batch_size
        self.latest_inp = None
        self.latest_outs = None
        self.norm_constant = torch.sqrt(torch.tensor(2, dtype=torch.float32, requires_grad=False)).to(self.device)

    def forward(self, x1, x2):
        output1 = self.layers(x1)
        output2 = self.layers(x2)
        assert not torch.isnan(output1).any()
        assert not torch.isnan(output2).any()
        self.latest_inp = (x1, x2)
        self.latest_outs = (output1, output2)
        # Outputs the difference/change
        return euclidean_distance(output1, output2) / self.norm_constant

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.bs, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_set, batch_size=self.bs, shuffle=False, num_workers=0)

    def training_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        pair_label = torch.squeeze(pair_label).to(torch.float32)
        out = self.forward(x1, x2)
        loss = nn.BCELoss()(out, pair_label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        pair_label = torch.squeeze(pair_label).to(torch.float32)
        out = self.forward(x1, x2)
        loss = nn.BCELoss()(out, pair_label)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            },
        )


class PatchChangeDet128(pl.LightningModule):
    def __init__(self, train_set, val_set, batch_size, lr=1e-4, feature_size=64, dropout_p=0.2):
        super(PatchChangeDet128, self).__init__()
        self.feature_size = feature_size
        self.dropout_p = dropout_p
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(256, self.feature_size),
            nn.Dropout(self.dropout_p),
            # nn.Sigmoid()
            nn.Softmax()
        )

        self.lr = lr
        self.train_set = train_set
        self.val_set = val_set
        self.bs = batch_size

        self.norm_constant = torch.sqrt(torch.tensor(2, dtype=torch.float32, requires_grad=False)).to(self.device)

    def forward(self, x1, x2):
        output1 = self.layers(x1)
        output2 = self.layers(x2)
        assert not torch.isnan(output1).any()
        assert not torch.isnan(output2).any()
        # Outputs the difference/change
        return euclidean_distance(output1, output2) / self.norm_constant

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.bs, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_set, batch_size=self.bs, shuffle=False, num_workers=0)

    def training_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        pair_label = torch.squeeze(pair_label).to(torch.float32)
        out = self.forward(x1, x2)
        loss = nn.BCELoss()(out, pair_label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        pair_label = torch.squeeze(pair_label).to(torch.float32)
        out = self.forward(x1, x2)
        loss = nn.BCELoss()(out, pair_label)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.2)
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            },
        )


class PatchChangeDet128XL(pl.LightningModule):
    def __init__(self, train_set, val_set, batch_size, lr=1e-4, feature_size=256, dropout_p=0.3):
        super(PatchChangeDet128XL, self).__init__()
        self.feature_size = feature_size
        self.dropout_p = dropout_p
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=5),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 1024, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(1024, self.feature_size),
            nn.Dropout(self.dropout_p),
            # nn.Sigmoid()
            nn.Softmax()
        )

        self.lr = lr
        self.train_set = train_set
        self.val_set = val_set
        self.bs = batch_size

        self.norm_constant = torch.sqrt(torch.tensor(2, dtype=torch.float32, requires_grad=False)).to(self.device)

    def forward(self, x1, x2):
        output1 = self.layers(x1)
        output2 = self.layers(x2)
        assert not torch.isnan(output1).any()
        assert not torch.isnan(output2).any()
        # Outputs the difference/change
        return euclidean_distance(output1, output2) / self.norm_constant

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.bs, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_set, batch_size=self.bs, shuffle=False, num_workers=0)

    def training_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        pair_label = torch.squeeze(pair_label).to(torch.float32)
        out = self.forward(x1, x2)
        loss = nn.BCELoss()(out, pair_label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, pair_label = batch
        pair_label = torch.squeeze(pair_label).to(torch.float32)
        out = self.forward(x1, x2)
        loss = nn.BCELoss()(out, pair_label)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            },
        )


def euclidean_distance(t1, t2):
    squared_diff = (t1 - t2).pow(2)
    sum_squared_diff = squared_diff.sum()
    distance = torch.sqrt(sum_squared_diff)
    return distance
