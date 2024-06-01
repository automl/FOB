from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value,
                        _stack_if_compiling, _dispatch_sqrt, _default_to_fused_or_foreach)


import torch.nn as nn
def group_parameters_for_cpr_optimizer(model, bias_weight_decay=False,
                                   normalization_weight_decay=False): # TODO I SHOULD USE THIS AS WELL
    """Set weight_decay=0.0 for parameters in model.no_weight_decay, for parameters with
    attribute _no_weight_decay==True, for bias parameters if bias_weight_decay==False, for
    normalization parameters if normalization_weight_decay==False
    """
    # Get the weight decay from the config, or from the default value of the optimizer constructor
    # if it's not specified in the config.
    weight_decay = 1.0

    # If none of the parameters have weight decay anyway, and there are no parameters with special
    # optimization params
    skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
    skip_keywords = (model.no_weight_decay_keywords() if hasattr(model, 'no_weight_decay_keywords')
                     else set())

    # Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    special = set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.Embedding, )
    if not normalization_weight_decay:
        blacklist_weight_modules += (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                     nn.GroupNorm, nn.SyncBatchNorm,
                                     nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                     nn.LayerNorm, nn.LocalResponseNorm)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # In case of parameter sharing, some parameters show up here but are not in
            # param_dict.keys()
            if not p.requires_grad or fpn not in param_dict:
                continue  # frozen weights
            if hasattr(p, '_optim'):
                special.add(fpn)
            elif fpn in skip or any(skip_keyword in fpn for skip_keyword in skip_keywords):
                no_decay.add(fpn)
            elif getattr(p, '_no_weight_decay', False):
                no_decay.add(fpn)
            elif not bias_weight_decay and pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    decay |= (param_dict.keys() - no_decay - special)
    # validate that we considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - special - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)}  were not separated into either decay/no_decay set!"

    if weight_decay == 0.0 or not no_decay:
        param_groups = [{"params": [param_dict[pn] for pn in sorted(list(no_decay | decay))],
                         "weight_decay": weight_decay}]
    else:
        # We need sorted(list()) so that the order is deterministic. Otherwise when we resume
        # the order could change and resume will fail. [H/t Albert]
        param_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
    # Add parameters with special hyperparameters
    # Unique dicts
    hps = [dict(s) for s in set(frozenset(param_dict[pn]._optim.items()) for pn in special)]
    for hp in hps:
        params = [param_dict[pn] for pn in sorted(list(special)) if param_dict[pn]._optim == hp]
        param_groups.append({"params": params, **hp})

    return param_groups


__all__ = ['AdamCPRfast', 'adamcpr']


class AdamCPRfast(Optimizer):
    def __init__(self,
                 params,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 kappa_init_param: float = 1000,
                 kappa_init_method: str = 'warm_start',
                 reg_function: str = 'l2',
                 kappa_update: float = 1.0,
                 amsgrad: bool = False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(lr, Tensor) and foreach and not capturable:
            raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        if not 0.0 <= kappa_update:
            raise ValueError(f"Invalid kappa_update value: {kappa_update}")
        if not (0.0 == weight_decay or 1.0 == weight_decay):
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self.reg_function = reg_function
        self.kappa_init_method = kappa_init_method

        if self.kappa_init_method not in ['warm_start', 'uniform', 'dependent', 'inflection_point']:
            raise ValueError(f"Invalid kappa_init_method: {kappa_init_method}")
        if self.kappa_init_method == "warm_start":
            self.warm_start = kappa_init_param
        elif self.kappa_init_method == 'inflection_point':
            self.warm_start = int(kappa_init_param // 10)
        else:
            self.warm_start = 0
            self.kappa_init_param = kappa_init_param

        self.kappa_update = kappa_update

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, kappa_update=kappa_update,
                        amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable)
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            state_values = list(self.state.values())
            step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
            if not step_is_tensor:
                for s in state_values:
                    s['step'] = torch.tensor(float(s['step']))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        lagmuls,
        kappas,
        kappa_updates,
        prev_regs,
        prev_reg_gradients,
        prev_reg_second_derivatives,
        state_steps
    ):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable is off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group['capturable']
                        else torch.tensor(0.0)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['lagmul'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['prev_reg'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['prev_reg_gradient'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['prev_reg_second_derivative'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['lagmul'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['kappa_update'] = torch.tensor([self.kappa_update], dtype=torch.float, device=p.device)
                    if self.kappa_init_method == 'uniform':
                        state["kappa"] = torch.tensor([self.kappa_init_param], dtype=torch.float, device=p.device)
                    elif self.kappa_init_method == 'warm_start':
                        state["kappa"] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    elif self.kappa_init_method == 'inflection_point':
                        state["kappa"] = torch.tensor([1000], dtype=torch.float, device=p.device)
                    elif self.kappa_init_method == 'dependent':
                        if self.reg_function == 'std':
                            state["kappa"] = self.kappa_init_param * torch.std(p).detach()
                        elif self.reg_function == 'l2':
                            kappa = self.kappa_init_param * p.square().mean().detach().item()
                            state["kappa"] = torch.tensor([kappa], dtype=torch.float, device=p.device)
                        elif self.reg_function == 'l1':
                            kappa = self.kappa_init_param * p.abs().mean().detach().item()
                            state["kappa"] = torch.tensor([kappa], dtype=torch.float, device=p.device)
                        elif self.reg_function == 'huber':
                            kappa = self.kappa_init_param * torch.where(p.abs() < 1, 0.5 * p.square(), p.abs() - 0.5).mean().detach().item()
                            state["kappa"] = torch.tensor([kappa], dtype=torch.float, device=p.device)


                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                lagmuls.append(state['lagmul'])
                kappas.append(state['kappa'])
                kappa_updates.append(state['kappa_update'])
                prev_regs.append(state['prev_reg'])
                prev_reg_gradients.append(state['prev_reg_gradient'])
                prev_reg_second_derivatives.append(state['prev_reg_second_derivative'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

                # Foreach without capturable does not support a tensor lr
                if group['foreach'] and torch.is_tensor(group['lr']) and not group['capturable']:
                    raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

                state_steps.append(state['step'])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            lagmuls = []
            kappas = []
            kappa_updates = []
            prev_regs = []
            prev_reg_gradients = []
            prev_reg_second_derivatives = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group['betas']

            self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lagmuls,
                kappas,
                kappa_updates,
                prev_regs,
                prev_reg_gradients,
                prev_reg_second_derivatives,
                state_steps)

            adam_cpr(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lagmuls,
                kappas,
                kappa_updates,
                prev_regs,
                prev_reg_gradients,
                prev_reg_second_derivatives,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                warm_start=self.warm_start,
                reg_function=self.reg_function,
                kappa_init_method=self.kappa_init_method,
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def adam_cpr(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         lagmuls: List[Tensor],
         kappas: List[Tensor],
         kappa_updates: List[Tensor],
         prev_regs: List[Tensor],
         prev_reg_gradients: List[Tensor],
         prev_reg_second_derivatives: List[Tensor],
         state_steps: List[Tensor],
         foreach: Optional[bool] = None,
         capturable: bool = False,
         differentiable: bool = False,
         grad_scale: Optional[Tensor] = None,
         found_inf: Optional[Tensor] = None,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: Union[float, Tensor],
         weight_decay: float,
         warm_start: int,
         reg_function: str,
         kappa_init_method: str,
         eps: float,
         maximize: bool):

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         lagmuls,
         kappas,
         kappa_updates,
         prev_regs,
         prev_reg_gradients,
         prev_reg_second_derivatives,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         warm_start=warm_start,
         reg_function=reg_function,
         kappa_init_method=kappa_init_method,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         differentiable=differentiable,
         grad_scale=grad_scale,
         found_inf=found_inf)


@torch.jit.script
def l2_update(param, lagmul, kappa, kappa_update):

    n = param.numel()
    sum_l2norm = param.square().sum()

    param_specific_lagmul_rate = kappa_update / n
    param_specific_kappa = kappa * n

    constraint_value = sum_l2norm - param_specific_kappa

    lagmul.add_(param_specific_lagmul_rate * constraint_value).clip_(min=0.)
    param.addcmul_(param, lagmul, value=-2)


@torch.jit.script
def l1_update(param, lagmul, kappa, kappa_update):

    n = param.numel()
    sum_l1norm = param.abs().sum()

    param_specific_lagmul_rate = kappa_update / n
    param_specific_kappa = kappa * n

    constraint_value = sum_l1norm - param_specific_kappa

    lagmul.add_(param_specific_lagmul_rate * constraint_value).clip_(min=0.)
    param.addcmul_(param.sign(), lagmul, value=-1)


@torch.jit.script
def std_update(param, lagmul, kappa, kappa_update):

    n = param.numel()
    std_dev = param.std()
    constraint_value = std_dev - kappa

    mean = param.mean()
    norm_param = param.sub(mean)
    grad_std_dev = norm_param.mul_(2).sub_(2 * norm_param.mean()).div_(n - 1)
    grad_std_dev.div_(std_dev.mul_(2))

    lagmul.add_(kappa_update * constraint_value).clip_(min=0.)
    param.addcmul_(grad_std_dev, lagmul, value=-1)


@torch.jit.script
def huber_update(param, lagmul, kappa, kappa_update):

    n = param.numel()

    param_abs = param.abs()
    huber_idx = param_abs < 1
    huber_loss = torch.where(huber_idx, 0.5 * param.square(), param_abs - 0.5)
    sum_huber_loss = huber_loss.sum()

    param_specific_lagmul_rate = kappa_update / n
    param_specific_kappa = kappa * n

    constraint_value = sum_huber_loss - param_specific_kappa

    lagmul.add_(param_specific_lagmul_rate * constraint_value).clip_(min=0.)

    grad_huber = torch.where(huber_idx, param, param.sign())
    param.addcmul_(grad_huber, lagmul, value=-1)


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        lagmuls: List[Tensor],
                        kappas: List[Tensor],
                        kappa_updates: List[Tensor],
                        prev_regs: List[Tensor],
                        prev_reg_gradients: List[Tensor],
                        prev_reg_second_derivatives: List[Tensor],
                        state_steps: List[Tensor],
                        grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: Union[float, Tensor],
                        weight_decay: float,
                        warm_start: int,
                        reg_function: str,
                        kappa_init_method: str,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        lagmul = lagmuls[i]
        kappa = kappas[i]
        kappa_update = kappa_updates[i]
        prev_reg = prev_regs[i]
        prev_reg_gradient = prev_reg_gradients[i]
        prev_reg_second_derivative = prev_reg_second_derivatives[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        # update step
        step_t += 1

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)

        if weight_decay == 1.0:
            if step > warm_start:
                if reg_function == 'l2':
                    l2_update(param, lagmul, kappa, kappa_update)
                elif reg_function == 'std':
                    std_update(param, lagmul, kappa, kappa_update)
                elif reg_function == 'l1':
                    l1_update(param, lagmul, kappa, kappa_update)
                elif reg_function == 'huber':
                    huber_update(param, lagmul, kappa, kappa_update)
                else:
                    raise ValueError(f"Unsupported regularization function: {reg_function}")
            elif kappa_init_method == 'warm_start' and step == warm_start:
                if reg_function == 'l2':
                    kappa.add_(param.square().mean())
                elif reg_function == 'std':
                    kappa.add_(param.std())
                elif reg_function == 'l1':
                    kappa.add_(param.abs().mean())
                elif reg_function == 'huber':
                    param_abs = param.abs()
                    huber_loss = torch.where(param_abs < 1, 0.5 * param.square(), param_abs - 0.5)
                    kappa.add_(huber_loss.mean())


            if (kappa_init_method == 'inflection_point') and kappa == 1000:

                current_l2m = param.square().mean()
                current_reg_gradient = current_l2m - prev_reg
                current_reg_second_derivative = current_reg_gradient - prev_reg_gradient

                # Peak detection for gradient
                if kappa_init_method == 'inflection_point' and step > 1 and prev_gradient > current_reg_gradient:
                    kappa.mul_(0).add_(current_l2m)

                # Update previous values for next iteration
                prev_reg = current_l2m
                prev_reg_gradient = current_reg_gradient
                prev_reg_second_derivative = current_reg_second_derivative


        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adam(params: List[Tensor],
                       grads: List[Tensor],
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       max_exp_avg_sqs: List[Tensor],
                       lagmuls: List[Tensor],
                       kappas: List[Tensor],
                       kappa_updates: List[Tensor],
                       prev_regs: List[Tensor],
                       prev_reg_gradients: List[Tensor],
                       prev_reg_second_derivatives: List[Tensor],
                       state_steps: List[Tensor],
                       grad_scale: Optional[Tensor],
                       found_inf: Optional[Tensor],
                       *,
                       amsgrad: bool,
                       beta1: float,
                       beta2: float,
                       lr: Union[float, Tensor],
                       weight_decay: float,
                       warm_start: int,
                       reg_function: str,
                       kappa_init_method: str,
                       eps: float,
                       maximize: bool,
                       capturable: bool,
                       differentiable: bool):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, lagmuls, kappas, kappa_updates, prev_regs, prev_reg_gradients, prev_reg_second_derivatives, state_steps])
    for ((
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_lagmuls,
        device_kappas,
        device_kappa_updates,
        device_prev_regs,
        device_prev_reg_gradients,
        device_prev_reg_second_derivatives,
        device_state_steps,
    ), _) in grouped_tensors.values():

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        # Handle complex parameters
        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
        device_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avg_sqs]
        device_max_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_max_exp_avg_sqs]
        device_lagmuls = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lagmuls]
        device_kappas = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_kappas]
        device_kappa_updates = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_kappa_updates]
        device_prev_regs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_prev_regs]
        device_prev_reg_gradients = [torch.view_as_real(x) if torch.is_complex(x) else x for x in prev_reg_gradients]
        device_prev_reg_second_derivatives = [torch.view_as_real(x) if torch.is_complex(x) else x for x in prev_reg_second_derivatives]
        device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

        # update steps
        torch._foreach_add_(device_state_steps, 1)


        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)  # type: ignore[assignment]

                # Set intermediate to the max. for normalizing running avg. of gradient when amsgrad
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)

        if weight_decay == 1.0:
            if kappa_init_method == 'inflection_point':

                if reg_function == 'l2':
                    square_params = torch._foreach_pow(device_params, 2)

                    square_sum_params, ns = [], []
                    for square_param in square_params:
                        square_sum_params.append(square_param.sum().unsqueeze(0))
                        ns.append(square_param.numel())

                    param_specific_lagmul_rates = torch._foreach_div(device_kappa_updates, ns)
                    param_specific_kappas = torch._foreach_mul(device_kappas, ns)

                    torch._foreach_sub_(square_sum_params, param_specific_kappas)
                    torch._foreach_mul_(square_sum_params, param_specific_lagmul_rates)
                    torch._foreach_add_(device_lagmuls, square_sum_params)
                    for lagmul in device_lagmuls:
                        lagmul.clip_(min=0.)
                    torch._foreach_addcmul_(device_params, device_params, device_lagmuls, -2)
                else:
                    raise ValueError(f"Unsupported regularization function for grad peak init: {reg_function}")

                iteration = warm_start

                if any([device_kappa == 1000 for device_kappa in device_kappas]) and device_state_steps[0] % iteration == 0:

                    square_params = torch._foreach_pow(device_params, 2)

                    current_l2ss = [square_param.sum() for square_param in square_params]
                    if device_state_steps[0] > iteration:
                        current_gradients = torch._foreach_sub(current_l2ss, prev_regs)

                    if device_state_steps[0] > iteration*3:
                        if kappa_init_method == 'inflection_point' and device_state_steps[0] > 1:
                            for i in range(len(device_params)):
                                if prev_reg_gradients[i] > current_gradients[i] and device_kappas[i] == 1000:
                                    current_l2ms = [square_param.mean() for square_param in square_params]
                                    device_kappas[i].copy_(current_l2ms[i])

                    torch._foreach_copy_(prev_regs, current_l2ss)
                    if device_state_steps[0] > iteration:
                        torch._foreach_copy_(prev_reg_gradients, current_gradients)

            else:
                if warm_start < device_state_steps[0]:
                    if reg_function == 'l2':
                        square_params = torch._foreach_pow(device_params, 2)

                        square_sum_params, ns = [], []
                        for square_param in square_params:
                            square_sum_params.append(square_param.sum().unsqueeze(0))
                            ns.append(square_param.numel())

                        param_specific_lagmul_rates = torch._foreach_div(device_kappa_updates, ns)
                        param_specific_kappas = torch._foreach_mul(device_kappas, ns)

                        torch._foreach_sub_(square_sum_params, param_specific_kappas)
                        torch._foreach_mul_(square_sum_params, param_specific_lagmul_rates)
                        torch._foreach_add_(device_lagmuls, square_sum_params)
                        for lagmul in device_lagmuls:
                            lagmul.clip_(min=0.)
                        torch._foreach_addcmul_(device_params, device_params, device_lagmuls, -2)

                    elif reg_function == 'std':
                        abs_params = torch._foreach_abs(device_params)
                        std_params, ns = [], []
                        for device_param in device_params:
                            std_params.append(device_param.str().unsqueeze(0))
                            ns.append(device_param.numel() - 1)

                        mean_params = [device_param.mean() for device_param in device_params]
                        norm_params = torch._foreach_sub(device_params, mean_params)
                        mean_norm_params = [norm_param.mean() * 2 for norm_param in norm_params]

                        torch._foreach_mul_(norm_params, 2)
                        torch._foreach_sub_(norm_params, mean_norm_params)
                        torch._foreach_div_(norm_params, ns)
                        torch._foreach_div_(norm_params, torch._foreach_mul(std_params, 2))



                        torch._foreach_sub_(std_params, device_kappas)
                        torch._foreach_mul_(std_params, device_kappa_updates)
                        torch._foreach_add_(device_lagmuls, std_params)
                        for lagmul in device_lagmuls:
                            lagmul.clip_(min=0.)
                        torch._foreach_addcmul_(device_params, norm_params, device_lagmuls, -1)

                    elif reg_function == 'l1':
                        abs_params = torch._foreach_abs(device_params)
                        abs_sum_params, ns = [], []
                        for abs_param in abs_params:
                            abs_sum_params.append(abs_param.sum().unsqueeze(0))
                            ns.append(abs_param.numel())

                        param_specific_lagmul_rates = torch._foreach_div(device_kappa_updates, ns)
                        param_specific_kappas = torch._foreach_mul(device_kappas, ns)

                        torch._foreach_sub_(abs_sum_params, param_specific_kappas)
                        torch._foreach_mul_(abs_sum_params, param_specific_lagmul_rates)
                        torch._foreach_add_(device_lagmuls, abs_sum_params)
                        for lagmul in device_lagmuls:
                            lagmul.clip_(min=0.)
                        torch._foreach_addcmul_(device_params, device_params, device_lagmuls, -1)

                    elif reg_function == 'huber':

                        abs_params = torch._foreach_abs(device_params)
                        square_params = torch._foreach_pow(device_params, 2)
                        huber_loss_params, huber_loss_grads, ns = [], [], []
                        for param_abs, square_params, device_param in zip(abs_params, square_params, device_params):
                            huber_loss_params.append(torch.where(param_abs < 1, 0.5 * square_param, param_abs - 0.5).sum())
                            huber_loss_grads.append(torch.where(param_abs < 1, device_param, device_param.sign()))
                            ns.append(abs_param.numel())

                        param_specific_lagmul_rates = torch._foreach_div(device_kappa_updates, ns)
                        param_specific_kappas = torch._foreach_mul(device_kappas, ns)

                        torch._foreach_sub_(huber_loss_params, param_specific_kappas)
                        torch._foreach_mul_(huber_loss_params, param_specific_lagmul_rates)
                        torch._foreach_add_(device_lagmuls, huber_loss_params)
                        for lagmul in device_lagmuls:
                            lagmul.clip_(min=0.)
                        torch._foreach_addcmul_(device_params, huber_loss_grads, device_lagmuls, -1)


                    else:
                        raise ValueError(f"Unsupported regularization function: {reg_function}")
                elif kappa_init_method == 'warm_start' and device_state_steps[0] == warm_start:
                    if reg_function == 'l2':
                        square_params = torch._foreach_pow(device_params, 2)
                        new_kappas = [square_param.mean() for square_param in square_params]
                        torch._foreach_add_(device_kappas, new_kappas)

                    elif reg_function == 'std':
                        new_kappas = [device_param.std() for device_param in device_params]
                        torch._foreach_add_(device_kappas, new_kappas)

                    elif reg_function == 'l1':
                        abs_params = torch._foreach_abs(device_params)
                        new_kappas = [abs_param.mean() for abs_param in abs_params]
                        torch._foreach_add_(device_kappas, new_kappas)

                    elif reg_function == 'huber':
                        abs_params = torch._foreach_abs(device_params)
                        square_params = torch._foreach_pow(device_params, 2)
                        new_kappas = [torch.where(param_abs < 1, 0.5 * square_param, param_abs - 0.5).mean() for param_abs, square_params in zip(abs_params, square_params)]
                        torch._foreach_add_(device_kappas, new_kappas)
