import torch


def dense_ot(cost, gamma, niters):
    K = torch.exp(-cost / gamma)

    a = torch.full([cost.shape[0]], 1. / cost.shape[0], requires_grad=False, device=cost.device)
    b = torch.full([cost.shape[1]], 1. / cost.shape[1], requires_grad=False, device=cost.device)

    lefts = [torch.ones_like(a).to(a)]
    rights = []

    for i in range(niters):
        rights += [b / torch.matmul(lefts[i - 1], K)]
        lefts += [a / torch.matmul(K, rights[i])]

    transport_plan = lefts[-1].view(-1, 1) * K * rights[-1].view(1, -1)
    # wasserstein_distance = torch.sum(cost * transport_plan)

    mined_view_id = torch.max(transport_plan, 1)[1]
    return mined_view_id
