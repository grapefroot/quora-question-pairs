import torch


class SentenceClf(torch.nn.Module):
    def __init__(self, emb_model):
        super(SentenceClf, self).__init__()
        self.emb_model = emb_model
        self.emb_size = 768
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size * 2, 512),
            torch.nn.Dropout(),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 2)
        ).cuda()

    # self.attn_block = attn_block.cuda()

    def forward(self, enc_1, mask_1, enc_2, mask_2):
        # average, concatenate, process with mlp

        with torch.no_grad():
            hidden_1 = self.emb_model(enc_1, attention_mask=mask_1)[2][-2]
            hidden_2 = self.emb_model(enc_2, attention_mask=mask_2)[2][-2]

            hidden_1_count = mask_1.sum(axis=1, keepdims=True)
            hidden_2_count = mask_2.sum(axis=1, keepdims=True)

            first = (hidden_1).sum(axis=1) / hidden_1_count
            second = (hidden_2).sum(axis=1) / hidden_2_count

        #             first = (hidden_1 * mask_1.unsqueeze(2)).sum(axis=1) / hidden_1_count
        #             second = (hidden_2 * mask_2.unsqueeze(2)).sum(axis=1) / hidden_2_count
        #         first, second = self.attn_block(hidden_1, mask_1, hidden_2, mask_2)

        # input: batch_size x word_size x embed_dim
        mlp_input = torch.cat(
            (first, second),
            axis=1
        )

        return self.clf(mlp_input)
