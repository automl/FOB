from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value,
                        _stack_if_compiling, _dispatch_sqrt, _default_to_fused_or_foreach)


def group_parameters_for_adam_plus(model, normalization_weight_decay=False):


    whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Conv1d)
    blacklist_weight_modules = (nn.Embedding, )
    if not normalization_weight_decay:
        blacklist_weight_modules += (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                     nn.GroupNorm, nn.SyncBatchNorm,
                                     nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                     nn.LayerNorm, nn.LocalResponseNorm)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    lr_index_master = {}
    lr_index_client = {}

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn # full param name

            # In case of parameter sharing, some parameters show up here but are not in
            # param_dict.keys()
            if not p.requires_grad or fpn not in param_dict:
                continue  # frozen weights
            if hasattr(p, '_optim'):
                continue
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                master_index = len(lr_index_master)
                lr_index_master[fpn] = master_index
                if fpn.replace('weight', 'bias') in param_dict:
                    lr_index_client[fpn.replace('weight', 'bias')] = master_index
            elif isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                lr_index_client[fpn] = -1
                if fpn.replace('weight', 'bias') in param_dict:
                    lr_index_client[fpn.replace('weight', 'bias')] = -1

    for pn in param_dict.keys():
        if pn not in lr_index_master and pn not in lr_index_client:
            print("Parameter not found in lr_index_master or lr_index_client: ", pn)

    assert len(lr_index_master) + len(lr_index_client) == len(param_dict)

    master_params = []
    master_indexes = []
    client_params = []
    client_indexes = []
    for pn, p in lr_index_master.items():
        master_params.append(param_dict[pn])
        master_indexes.append(p)
    for pn, p in lr_index_client.items():
        client_params.append(param_dict[pn])
        client_indexes.append(p)


    param_groups = [
        {"params": master_params, "lr_index": master_indexes, "lr_update": True, "regularized": True},
        {"params": client_params, "lr_index": client_indexes, "lr_update": False, "regularized": False},
    ]
    # # Add parameters with special hyperparameters
    # # Unique dicts
    # hps = [dict(s) for s in set(frozenset(param_dict[pn]._optim.items()) for pn in special)]
    # for hp in hps:
    #     params = [param_dict[pn] for pn in sorted(list(special)) if param_dict[pn]._optim == hp]
    #     param_groups.append({"params": params, **hp})

    return param_groups

__all__ = ['AdamPlus', 'adamplus']


class AdamPlus(Optimizer):
    def __init__(self,
                 params,
                 lr_grad: Union[float, Tensor] = 1e-6, # 6e-7
                 lr_decay: Union[float, Tensor] = 0.1,
                 train_step: Union[float, Tensor] = 20000,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 reg_step_size: int = 100,
                 kappa_update: float = 1.0,
                 amsgrad: bool = False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False):
        if lr_grad < 0.0:
            raise ValueError(f"Invalid learning rate: {lr_grad}")
        if isinstance(lr_grad, Tensor) and foreach and not capturable:
            raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        if kappa_update < 0.0:
            raise ValueError(f"Invalid kappa_update value: {kappa_update}")
        self.lr_grad = lr_grad
        self.reg_step_size = reg_step_size
        self.kappa_update = kappa_update

        self.lr_decay = lr_decay
        self.train_step = train_step

        defaults = dict(lr_grad=lr_grad, lr_decay=lr_decay, train_step=train_step, betas=betas, eps=eps,
                        kappa_update=kappa_update,
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
            group.setdefault('lr_update', False)
            group.setdefault('regularized', False)
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
        lr_index,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        lrs,
        lagmuls,
        kappas,
        lr_decay_factors,
        kappa_updates,
        prev_regs,
        prev_reg_gradients,
        prev_reg_second_derivatives,
        state_steps
    ):
        for p, lr_idx in zip(group['params'], group['lr_index']):
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)
                lr_index.append(lr_idx)
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

                    state['lr'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['lagmul'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['prev_reg'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['prev_reg_gradient'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['prev_reg_second_derivative'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['lagmul'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                    state['kappa_update'] = torch.tensor([self.kappa_update], dtype=torch.float, device=p.device)
                    state["kappa"] = torch.tensor([1000], dtype=torch.float, device=p.device)
                    state["lr_decay_factor"] = torch.tensor([1], dtype=torch.float, device=p.device)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                lrs.append(state['lr'])
                lagmuls.append(state['lagmul'])
                kappas.append(state['kappa'])
                lr_decay_factors.append(state['lr_decay_factor'])
                kappa_updates.append(state['kappa_update'])
                prev_regs.append(state['prev_reg'])
                prev_reg_gradients.append(state['prev_reg_gradient'])
                prev_reg_second_derivatives.append(state['prev_reg_second_derivative'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

                # Foreach without capturable does not support a tensor lr
                # if group['foreach'] and not group['capturable']:
                #     raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

                state_steps.append(state['step'])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        is_initialized = False
        lr_index_dict = {}
        for group in self.param_groups:
            if group['lr_update']:
                for p, lr_index in zip(group['params'], group['lr_index']):
                    state = self.state[p]
                    if "lr" in state:
                        lr = self.state[p]['lr']
                        lr_index_dict[lr_index] = lr
                        is_initialized = True
        if is_initialized:
            mean_lr = sum(lr_index_dict.values()) / len(lr_index_dict)
            for group in self.param_groups:
                if not group['lr_update']:
                    for p, lr_index in zip(group['params'], group['lr_index']):
                        if lr_index == -1:
                            self.state[p]['lr'] = mean_lr
                        else:
                            self.state[p]['lr'].copy_(lr_index_dict[lr_index])


        for group in self.param_groups:
            params_with_grad = []
            grads = []
            lr_indexs = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            lrs = []
            lagmuls = []
            kappas = []
            lr_decay_factors = []
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
                lr_indexs,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lrs,
                lagmuls,
                kappas,
                lr_decay_factors,
                kappa_updates,
                prev_regs,
                prev_reg_gradients,
                prev_reg_second_derivatives,
                state_steps)

            adam_cpr(
                params_with_grad,
                grads,
                lr_indexs,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lrs,
                lagmuls,
                kappas,
                lr_decay_factors,
                kappa_updates,
                prev_regs,
                prev_reg_gradients,
                prev_reg_second_derivatives,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr_update=group['lr_update'],
                regularized=group['regularized'],
                lr_grad=self.lr_grad,
                lr_decay=self.lr_decay,
                train_step=self.train_step,
                reg_step_size=self.reg_step_size,
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
         lr_indexs: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         lrs: List[Tensor],
         lagmuls: List[Tensor],
         kappas: List[Tensor],
         lr_decay_factors: List[Tensor],
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
         lr_update: bool,
         regularized: bool,
         lr_grad: float,
         lr_decay: float,
         train_step: int,
         reg_step_size: int,
         eps: float,
         maximize: bool):

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and not capturable:
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
         lrs,
         lagmuls,
         kappas,
         lr_decay_factors,
         kappa_updates,
         prev_regs,
         prev_reg_gradients,
         prev_reg_second_derivatives,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr_update=lr_update,
         regularized=regularized,
         lr_grad=lr_grad,
         lr_decay=lr_decay,
         train_step=train_step,
         reg_step_size=reg_step_size,
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


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        lrs: List[Tensor],
                        lagmuls: List[Tensor],
                        kappas: List[Tensor],
                        lr_decay_factors: List[Tensor],
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
                        lr_update: bool,
                        regularized: bool,
                        lr_grad: float,
                        lr_decay: float,
                        train_step: int,
                        reg_step_size: int,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool):

    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        lr = lrs[i]
        lagmul = lagmuls[i]
        kappa = kappas[i]
        lr_decay_factor = lr_decay_factors[i]
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

            param.addcdiv_(exp_avg, denom, value=-step_size.item())

        if lr_update:
            if kappa != 1000:

                if regularized:
                    l2_update(param, lagmul, kappa, kappa_update)

                lr.mul_(lr_decay_factor)


            elif step % reg_step_size == 0 and kappa == 1000:

                current_l2m = param.square().mean()
                if step > reg_step_size:
                    current_reg_gradient = current_l2m - prev_reg
                if step > reg_step_size * 2:
                    current_reg_second_derivative = current_reg_gradient - prev_reg_gradient

                # Peak detection for gradient
                if step > reg_step_size * 3 and prev_reg_gradient > current_reg_gradient and prev_reg_second_derivative > 0 and current_reg_second_derivative <= 0: # TODO do for foreach!
                    kappa.mul_(0).add_(current_l2m)
                    remaining_steps = train_step - step
                    lr_decay_factor.copy_(lr_decay ** (1 / remaining_steps))

                # Update previous values for next iteration
                prev_reg.copy_(current_l2m)
                if step > reg_step_size:
                    prev_reg_gradient.copy_(current_reg_gradient)
                if step > reg_step_size * 2:
                    prev_reg_second_derivative.copy_(current_reg_second_derivative)

                lr.add_(lr_grad)
            elif kappa == 1000:
                lr.add_(lr_grad)


        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adam(params: List[Tensor],
                       grads: List[Tensor],
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       max_exp_avg_sqs: List[Tensor],
                       lrs: List[Tensor],
                       lagmuls: List[Tensor],
                       kappas: List[Tensor],
                       lr_decay_factors: List[Tensor],
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
                       lr_update: bool,
                       regularized: bool,
                       lr_grad: float,
                       lr_decay: float,
                       train_step: int,
                       reg_step_size: int,
                       eps: float,
                       maximize: bool,
                       capturable: bool,
                       differentiable: bool):
    if len(params) == 0:
        return

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, lrs, lagmuls, kappas, lr_decay_factors, kappa_updates, prev_regs, prev_reg_gradients, prev_reg_second_derivatives, state_steps])
    for ((
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_lrs,
        device_lagmuls,
        device_kappas,
        device_lr_decay_factors,
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
        device_lrs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lrs]
        device_lagmuls = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lagmuls]
        device_kappas = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_kappas]
        device_lr_decay_factors = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lr_decay_factors]
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
            torch._foreach_div_(bias_correction1, device_lrs)
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

            step_size = _stack_if_compiling([(lr.item() / bc) * -1 for lr, bc in zip(device_lrs, bias_correction1)])

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

        if lr_update:

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

            if any([device_kappa == 1000 for device_kappa in device_kappas]):

                if device_state_steps[0] % reg_step_size == 0:
                    square_params = torch._foreach_pow(device_params, 2)

                    current_l2ss = [square_param.sum() for square_param in square_params]
                    if device_state_steps[0] > reg_step_size:
                        current_gradients = torch._foreach_sub(current_l2ss, device_prev_regs)
                    if device_state_steps[0] > reg_step_size * 2:
                        current_reg_second_derivatives = torch._foreach_sub(current_gradients, device_prev_reg_gradients)

                    if device_state_steps[0] > reg_step_size * 3:
                        for i in range(len(device_params)):
                            if device_prev_reg_gradients[i] > current_gradients[i] and device_kappas[i] == 1000:
                                current_l2ms = [square_param.mean() for square_param in square_params]
                                device_kappas[i].copy_(current_l2ms[i])
                                lr_decay_factors[i].copy_(lr_decay ** (1 / (train_step - device_state_steps[i])))

                    torch._foreach_copy_(device_prev_regs, current_l2ss)
                    if device_state_steps[0] > reg_step_size:
                        torch._foreach_copy_(device_prev_reg_gradients, current_gradients)
                    if device_state_steps[0] > reg_step_size * 2:
                        torch._foreach_copy_(device_prev_reg_second_derivatives, current_reg_second_derivatives)

                for i in range(len(device_params)):
                    if device_kappas[i] == 1000:
                        device_lrs[i].add_(lr_grad)

            if any([device_kappa != 1000 for device_kappa in device_kappas]):
                for i in range(len(device_params)):
                    if device_kappas[i] != 1000:
                        device_lrs[i].mul_(lr_decay_factors[i])
