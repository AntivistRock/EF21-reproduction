import torch
import torch.distributed as dist


class EF(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        lambda_=0.1,
        k=1,
    ):
        super(EF, self).__init__(
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
                self.state[p] = {'e': torch.zeros_like(p)}
    
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

                    w = self._top_k_compression(self.state[p]['e'] + group['lr'] * p.grad.data, group["k"])  # line 5
                    self.state[p]['e'] += group['lr'] * p.grad.data - w  # save error
                else:
                    w = torch.zeros_like(p)
                dist.reduce(w, dst=0)  # send w to the master

                if dist.get_rank() == 0:
                    p.data -= w / (dist.get_world_size() - 1)
                dist.broadcast(p, src=0)  # broadcast new parameter to all nodes
