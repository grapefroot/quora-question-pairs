import torch


def masked_softmax(input, masks, dim, epsilon=1e-12):
  masked_vec = input * masks
  max_vec = torch.max(masked_vec, dim=dim, keepdim=True).values
  exps = torch.exp(masked_vec - max_vec)
  masked_exps = exps * masks
  masked_sums = masked_exps.sum(dim, keepdim=True)
  return masked_exps / masked_sums


class VectorAttention(torch.nn.Module):
  def __init__(self, dim):
    super(VectorAttention, self).__init__()
    self.dim = dim
    self.param = torch.nn.Parameter(data=(torch.ones(self.dim, 1) / self.dim))

  def forward_one(self, batch_vectors, batch_masks):
    # batch_masks: batch_size x seq_len x 1
    coef = masked_softmax((batch_vectors @ self.param),
                          batch_masks.unsqueeze(2), dim=1)
    return (batch_vectors * coef).sum(dim=1)

  def forward(self, batch_1, mask_1, batch_2, mask_2):
    return self.forward_one(batch_1, mask_1), self.forward_one(batch_2, mask_2)


class NNAttention(torch.nn.Module):
  def __init__(self, dim):
    super(NNAttention, self).__init__()
    self.dim = dim
    self.nn = torch.nn.Sequential(torch.nn.Linear(dim, 1), torch.nn.Tanh())

  def forward_one(self, batch_vectors, batch_masks):
    coef = masked_softmax(self.nn(batch_vectors), batch_masks.unsqueeze(2),
                          dim=1)
    return (batch_vectors * coef).sum(dim=1)

  def forward(self, batch_1, mask_1, batch_2, mask_2):
    return self.forward_one(batch_1, mask_1), self.forward_one(batch_2, mask_2)


class Seq2SeqAttention(torch.nn.Module):
  def __init__(self, dim, hidden_dim=50):
    super(Seq2SeqAttention, self).__init__()
    self.dim = dim  # embedding dim
    self.hidden_dim = hidden_dim  # attention dim
    self.mapping = torch.nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, batch_1, mask_1, batch_2, mask_2):
    mapped_1 = self.mapping(batch_1)
    mapped_2 = self.mapping(batch_2)

    coef_matrix = torch.bmm(mapped_1, mapped_2.permute(0, 2, 1))
    coef_matrix = coef_matrix * mask_1.unsqueeze(2) * mask_2.unsqueeze(1)

    coef_1 = coef_matrix.sum(dim=2)
    coef_2 = coef_matrix.sum(dim=1)

    print(mask_1.shape)
    print(coef_1.shape)
    coef_1 = masked_softmax(coef_1, mask_1, dim=1).unsqueeze(2)
    coef_2 = masked_softmax(coef_2, mask_2, dim=1).unsqueeze(2)

    res_1 = (batch_1 * coef_1).sum(dim=1)
    res_2 = (batch_2 * coef_2).sum(dim=1)

    return res_1, res_2
