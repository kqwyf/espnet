from numpy.lib.function_base import vectorize
from egs2.lrs2.enh1.local.create_mixture import main
import torch
import torch.nn as nn

from loss import bi_modal_loss


class VGGM_V(nn.Module):

    def __init__(self, extract_emb=False):
        super(VGGM_V, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3,96,(5, 7, 7),(1, 2, 2), (2, 0, 0)),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3),(1, 2, 2),(0, 0, 0),ceil_mode=True),
            nn.Conv3d(96,256,(1, 5, 5),(1, 2, 2),(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3),(1, 2, 2),(0, 0, 0),ceil_mode=True),
            nn.Conv3d(256,512,(1, 3, 3),(1, 1, 1),(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512,512,(1, 3, 3),(1, 1, 1),(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512,512,(1, 3, 3),(1, 1, 1),(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3),(1, 2, 2),(0, 0, 0),ceil_mode=True),
            nn.Conv3d(512,512,(1, 6, 6),(1, 1, 1),(0, 0, 0)),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,128),
        )
        self.extract_emb = extract_emb

    def forward(self, x):
        """
            x.shape: batch, channel, len, w, h
        """

        bs, input_channel, length, w, h = x.shape 
        x = torch.nn.functional.interpolate(x, (length, 224, 224))

        x = self.features(x)
        x = x.view(bs, 512, length)
        x = x.transpose(1, 2)
        if self.extract_emb:
            return x
        x = self.mlp(x)
        return x

class VGGM_A(nn.Module):

    def __init__(self):
        super(VGGM_A, self).__init__()
        self.features = nn.Sequential( #  1, 257, 401
            nn.Conv2d(1,64,(5, 3),(1, 1)),  # 64, 255 399 
            nn.ReLU(),
            nn.MaxPool2d((5, 3),(3, 2),(0, 0),ceil_mode=True), 
            nn.Conv2d(64,192,(7, 5),(3, 1),(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(0, 1),ceil_mode=True),
            nn.Conv2d(192,384,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 1),(0, 1),ceil_mode=True),
            nn.Conv2d(512,512,(6, 1),(1, 1),(0, 0)),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,128),
        )

    def forward(self, x):

        # turn wav to magnitude specturm
        feats = torch.stft(x, 512, 160, 512)
        mag = (feats[..., 0].pow(2) + feats[..., 1].pow(2)).sqrt()
        mag = mag.unsqueeze(1)
        bs = mag.shape[0]
        length = mag.shape[-1] // 4

        x = self.features(mag)

        x = x.view(bs, 512, length)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        return x

class EmbeddingModel(nn.Module):

    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.model_v = VGGM_V()
        self.model_a = VGGM_A()

    def loss(self, a_emb, v_emb):
        return bi_modal_loss(a_emb, v_emb)
    
    def forward(self, audio, visual):
        a_emb = self.model_a(audio)
        v_emb = self.model_v(visual)


        loss = self.loss(a_emb,v_emb) 

        return a_emb, v_emb, loss

if __name__ == "__main__":
    
    video = torch.ones(2, 3, 100, 224, 224)
    print(video.shape)
    net = VGGM_V()
    print(net(video).shape)
    print('======')
    audio = torch.ones(2, 16000 * 4)
    output = VGGM_A()(audio)

    print(output.shape)


