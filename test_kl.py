import torch
from torch.distributions import Categorical, kl_divergence

logits_p = torch.tensor([-1.0, 2.0, float('-inf')])
logits_q = torch.tensor([-0.5, -1000.0, float('-inf')])

p = Categorical(logits=logits_p)
q = Categorical(logits=logits_q)
print(kl_divergence(p, q))
