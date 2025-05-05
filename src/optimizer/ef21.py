import torch
import torch.distributed as dist


class EF21(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        lambda_=0.1,
        k=1,
    ):
        super(EF21, self).__init__(
            params,
            defaults={
                "lr": lr,
                "lambda": lambda_,
                "k": k,  # k to define top-k
            },
        )

        # initialize memory
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {'g': torch.zeros_like(p)}
    
    # AI generated function
    def _top_k_compression(self, tensor, k):
        _, indices = torch.topk(tensor.abs().flatten(), k)
        compressed = torch.zeros_like(tensor).flatten()
        compressed[indices] = tensor.flatten()[indices]
        return compressed.reshape(tensor.shape)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if dist.get_rank() != 0:
                    p.grad.data += group["lambda"] * (2 * p.data) / (1 + p.data ** 2) ** 2  # add regularization term from equation (19) of the paper

                    c = self._top_k_compression(p.grad.data - self.state[p]['g'], group["k"])  # line 5
                    self.state[p]['g'] += c  # line 6
                else:
                    c = torch.zeros_like(p)
                dist.reduce(c, dst=0)  # send c to the master

                if dist.get_rank() == 0:
                    self.state[p]['g'] += c / (dist.get_world_size() - 1)
                    p.data -= group['lr'] * self.state[p]['g']
                dist.broadcast(p, src=0)  # broadcast new parameter to all nodes
