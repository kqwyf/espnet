from numpy.lib.function_base import vectorize
from egs2.lrs2.enh1.local.create_mixture import main
import torch
import torch.nn as nn

from loss import bi_modal_loss


class VGGM_V(nn.Module):

    def __init__(self, extract_emb=False):
        super(VGGM_V, self).__init__()

        self.front_3d = nn.Sequential(
            nn.Conv3d(3,96,(5, 7, 7),(1, 2, 2), (2, 0, 0)),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3),(1, 2, 2),(0, 0, 0),ceil_mode=True),
        )
        self.features = nn.Sequential(
            nn.Conv3d(96,256,(1, 5, 5),(1, 2, 2),(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3),(1, 2, 2),(0, 0, 0),ceil_mode=True),
            nn.Conv3d(256,512,(1, 3, 3),(1, 1, 1),(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512,512,(1, 3, 3),(1, 1, 1),(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512,512,(3, 3, 3),(1, 1, 1),(1, 1, 1)), # time-spatial conv
            nn.ReLU(),
        )

        self.pooling_mlp = nn.Sequential( 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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

        x = self.front_3d(x)
        x = self.features(x) # batch, channel, len, w', h'
        x = x.view(bs, 512, length, -1) # batch, channel, len, S (S=w' x h')
        w = x.permute(0, 2, 3, 1) # batch, len, S, channel
        w = self.pooling_mlp(w).squeeze(3)
        w = torch.softmax(w, dim=2)
        w = w.view(bs, 1, length, -1)
        x = (x * w).sum(-1)
        x = x.view(bs, 512, length)
        x = x.transpose(1, 2)
        if self.extract_emb:
            return x
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
    
    video = torch.rand(2, 3, 100, 224, 224)
    print(video.shape)
    net = VGGM_V()
    print(net(video).shape)
    print('======')


    print(output.shape)


