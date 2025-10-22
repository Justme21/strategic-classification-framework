import torch

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility

ZERO_THRESHOLD = 1e-7

class GradientAscentBestResponse(BaseBestResponse):
    """Use Stochastic Gradient Descent on the Agent objective to determine the best response z, for a given x, and a given function"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, strategic_columns=None, lr=1e-2, max_iterations=100, **kwargs):
        #BaseBestResponse.__init__(self, utility, cost)
        self._cost = cost
        self._utility = utility

        self._strategic_columns = strategic_columns

        self.max_iterations = max_iterations
        self.lr= lr
        self.opt = torch.optim.Adam

    def objective(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel) -> torch.Tensor:
        return self._utility(Z, model) - self._cost(X, Z)

    def __call__(self, X:torch.Tensor, model:BaseModel, debug=False, animate_rate=None) ->torch.Tensor:
        Z = X.detach().clone().requires_grad_()
        opt = self.opt([Z], self.lr)

        strat_mask = torch.zeros_like(Z, device=Z.device)
        strat_mask[:, self._strategic_columns] = 1.0

        if animate_rate is not None:
            assert isinstance(animate_rate, int)
            Z_store = X.detach().clone().unsqueeze(0)

        for t in range(self.max_iterations):
            opt.zero_grad()
            # To maximise the objective, we do gradient descent on the negative of the loss
            obj = self.objective(Z, X, model)

            l = -obj.mean()

            l.backward(inputs=[Z])

            # Only optimise the strategic features
            with torch.no_grad():
                if Z.grad is not None:
                    Z.grad *= strat_mask

            opt.step()

            # Save frames if animating
            if animate_rate is not None and t%animate_rate==0:
                Z_store = torch.cat([Z_store, Z.detach().clone().unsqueeze(0)], dim=0)

            # Check for Convergence
            #grad = Z.grad.detach()
            #grad_norm = grad.norm(dim=1).mean().item()
            #if grad_norm < ZERO_THRESHOLD:   # e.g. 1e-4
            #    break

        # To do the where evaluation we need a X_store the same size as Z
        if animate_rate is not None:
            if t%animate_rate !=0:
                # Store the converged Z value
                Z_store = torch.cat([Z_store, Z.detach().clone().unsqueeze(0)], dim=0) 
            X_store = X.detach().clone().unsqueeze(0).repeat([Z_store.shape[0]]+[1 for _ in range(len(X.shape))])
        

        with torch.no_grad():
            pred_X = model.predict(X)
            pred_Z = model.predict(Z)
            cond1 = pred_Z>pred_X

            cost = self._cost(X,Z)
            cond2 = cost<2
            cond = cond1*cond2

            cond = cond.unsqueeze(1)
            if animate_rate is not None:
                cond = cond.unsqueeze(0)
                X_opt = torch.where(cond, Z_store, X_store).clone()
            else:
                X_opt = torch.where(cond, Z, X).clone()

        return X_opt