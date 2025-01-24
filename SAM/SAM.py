import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        """Sharpness-Aware Minimization (SAM) Optimizer.

        Args:
            params: Model parameters to optimize.
            base_optimizer_cls: The base optimizer class (e.g., torch.optim.SGD).
            rho: Neighborhood size (sharpness constraint).
            kwargs: Additional arguments for the base optimizer.
        """
        if rho <= 0:
            raise ValueError("Invalid value for rho, should be > 0")
        
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.rho = rho
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)  

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """Perform the first step: perturb the weights."""
        grad_norm = self._grad_norm()  
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for param in group['params']:
                if param.grad is not None:
                    self.state[param]['original_param'] = param.data.clone()
                    param.add_(param.grad * scale)  

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """Perform the second step: restore weights and apply SAM update."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.data = self.state[param]['original_param']
                    self.state[param]['original_param'] = None  

        self.base_optimizer.step()  

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """Perform both steps of the SAM update."""
        if closure is None:
            raise ValueError("A closure that reevaluates the model and returns the loss is required.")
        
        closure()
        self.first_step()

        closure()
        self.second_step()

    def _grad_norm(self):
        """Compute the norm of gradients."""
        norm = torch.norm(
            torch.stack([
                param.grad.norm(p=2)
                for group in self.param_groups
                for param in group['params']
                if param.grad is not None
            ]),
            p=2
        )
        return norm
