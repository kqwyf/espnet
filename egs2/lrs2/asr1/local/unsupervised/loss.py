import torch



eps = 1.e-8
cos = torch.nn.CosineSimilarity()

def similarity(x1,x2):
    tmp = (x1 - x2 + eps).norm(dim=-1)
    return 1/(tmp + 0.05)

def AV_loss(audio, visual):
    """
    audio: B, L, N
    visual: B, L, N
    """
    B = len(audio)
    sigma_1 = 0
    for j in range(B):
        sigma_2 = 0
        for k in range(B):
            sigma_2 += similarity(audio[j], visual[k])
        sigma_1 += torch.log(similarity(audio[j], visual[j]) / sigma_2)
    loss = - (1/B) * sigma_1
    return loss


def AAV_loss(audio, visual):
    """
    audio: B, L, N
    visual: B, L, N
    """
    B = len(audio)
    sigma_1 = 0
    for j in range(B):
        sigma_2 = 0
        for k in range(B):
            if k == j:
                continue
            sigma_2 += similarity(audio[k], audio[j])
        sigma_1 += torch.log(similarity(audio[j], visual[j]) / (similarity(audio[j], visual[j]) + sigma_2))
    loss = - (1/B) * sigma_1
    return loss


def in_modal_sim(seq):
    """
        seq: B, L, N
    """
    B, L, N = seq.shape

    # build a similarity weight matrix
    weight_kernel = torch.ones((3, 3))  / 3
    weight_kernel = weight_kernel.to(seq.device)
    weight_kernel = weight_kernel.view(1, 1, 3, 3)
    similarity_weight = torch.eye(L).to(seq.device)
    similarity_weight = similarity_weight.view(1, 1, L, L)
    similarity_weight = torch.conv2d(similarity_weight, weight_kernel, padding=1)
    similarity_weight = similarity_weight.squeeze()
    similarity_weight = (1 - similarity_weight)**2
    similarity_weight[0,0] = 0
    similarity_weight[L-1, L-1] = 0
    # print(similarity_weight)

    seq_row = seq.unsqueeze(1)
    seq_col = seq.unsqueeze(2)
    similarity_score = similarity(seq_row, seq_col)

    similarity_score = similarity_score * similarity_weight

    # print(similarity_score)

    return similarity_score.mean()
    # return similarity_score.shape

def cross_modal_sim(x1, x2):
    """
        x1: B, L, N
        x2: B, L, N
    """
    similarity_score = similarity(x1=x1, x2=x2).mean()
    return similarity_score

    
def bi_modal_loss(x1, x2):

    cross_modal = cross_modal_sim(x1, x2)
    in_modal_1 = in_modal_sim(x1)
    in_modal_2 = in_modal_sim(x2)


    loss = -10 * torch.log(cross_modal) + torch.log(in_modal_1) + torch.log(in_modal_2)

    return loss

if __name__ == "__main__":
    # video = torch.ones(6, 100, 128) + 10
    # audio = torch.ones(6, 100, 128)
    # print(AV_loss(visual=video, audio=audio))
    # print(AV_loss(visual=audio, audio=video))
    # print(AAV_loss(visual=audio, audio=video))
    # print(AAV_loss(visual=video, audio=audio))

    x1 = torch.rand(6, 10, 128)
    x2 = torch.rand(6, 10, 128)

    # print(similarity(x1=x1,x2=x2))
    print(in_modal_sim(x1))
    print(cross_modal_sim(x1, x2))
    print(bi_modal_loss(x1=x1, x2=x2))
    # AV_loss(viusal=video, audio=audio)
