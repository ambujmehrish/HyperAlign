import os
import torch
import torch.distributed as dist

from model import model_registry
from torch.nn.parallel import DistributedDataParallel as DDP
from .logger import LOGGER
from .build_optimizer import build_optimizer


class DDP_modify(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module,name)


def build_model(args):

    model = model_registry[args.model_cfg.model_type](args.model_cfg)
    checkpoint = {}
    
    ### load ckpt from a pretrained_dir
    print("args.run_cfg.pretrain_dir", args.run_cfg.pretrain_dir)
    if args.run_cfg.pretrain_dir:
        checkpoint = load_from_pretrained_dir(args)
        LOGGER.info("Load from pretrained dir {}".format(args.run_cfg.pretrain_dir))

    ### load ckpt from specific path
    if args.run_cfg.checkpoint:
        checkpoint =  torch.load(args.run_cfg.checkpoint, map_location = 'cpu')

    ### resume training
    if args.run_cfg.resume:
        checkpoint, checkpoint_optim, start_step = load_from_resume(args.run_cfg)
    else:
        checkpoint_optim, start_step = None , 0


    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    
    if checkpoint != {}:

        checkpoint = model.modify_checkpoint(checkpoint)
        if "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]

        missing_keys,unexpected_keys = model.load_state_dict(checkpoint,strict=False)
        LOGGER.info(f"Unexpected keys {unexpected_keys}")
        LOGGER.info(f"missing_keys  {missing_keys}")

        # strict=False is required -- stage B adds `hgnn` params that no stage-A or VAST
        # checkpoint carries, and those SHOULD be missing. But it also silently tolerates a
        # checkpoint whose keys are named differently from this model's, in which case almost
        # nothing loads and the model evaluates from its random init. That failure is invisible:
        # it produces a plausible-looking low score rather than an error. Fail loudly instead.
        _total = len(model.state_dict())
        _new = tuple(args.run_cfg.new_params_name or ())          # legitimately-missing prefixes
        _unexplained = [k for k in missing_keys
                        if not any(k.startswith(p) or f'.{p}.' in k or k.startswith(f'{p}.')
                                   for p in _new)]
        _loaded = _total - len(missing_keys)
        LOGGER.info(f"checkpoint load: {_loaded}/{_total} tensors matched "
                    f"({_loaded/max(1,_total)*100:.1f}%), {len(missing_keys)} missing "
                    f"({len(_unexplained)} not explained by new_params_name={list(_new)}), "
                    f"{len(unexpected_keys)} unexpected")
        if _loaded < 0.5 * _total and not os.environ.get('GRAM_ALLOW_PARTIAL_CKPT'):
            raise SystemExit(
                f"Refusing to run: only {_loaded}/{_total} ({_loaded/max(1,_total)*100:.1f}%) of "
                f"the model's tensors were found in the checkpoint. The checkpoint's key naming "
                f"probably does not match this model_cfg. Evaluating anyway would score a mostly "
                f"randomly-initialised model and look like a real result.\n"
                f"  checkpoint : {args.run_cfg.checkpoint or args.run_cfg.pretrain_dir}\n"
                f"  first missing    : {missing_keys[:5]}\n"
                f"  first unexpected : {unexpected_keys[:5]}\n"
                f"Set GRAM_ALLOW_PARTIAL_CKPT=1 to override if this is intentional.")



    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    model.to(device)
    if  args.run_cfg.use_ddp:
        model = DDP_modify(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        pass

    return model, checkpoint_optim, start_step



def load_from_pretrained_dir(args):


    try: ### huggingface trainer
        checkpoint_dir = args.run_cfg.pretrain_dir
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('checkpoint')]
        checkpoint_ls = [int(i.split('-')[1]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
        
        try:
            checkpoint_name = f'checkpoint-{step}/pytorch_model.bin'
            ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
            checkpoint = torch.load(ckpt_file, map_location = 'cpu')
        except:
            checkpoint_name1 = f'checkpoint-{step}/pytorch_model-00001-of-00002.bin'
            ckpt_file1 = torch.load(os.path.join(checkpoint_dir, checkpoint_name1), map_location = 'cpu')
            checkpoint_name2 = f'checkpoint-{step}/pytorch_model-00002-of-00002.bin'
            ckpt_file2 = torch.load(os.path.join(checkpoint_dir, checkpoint_name2), map_location = 'cpu')
            ckpt_file1.update(ckpt_file2)
            checkpoint = ckpt_file1
        # checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
        LOGGER.info(f'load_from_pretrained: {ckpt_file}')

    except:
        checkpoint_dir = os.path.join(args.run_cfg.pretrain_dir,'ckpt')
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
            
        checkpoint_name = 'model_step_'+str(step)+'.pt'
        ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(ckpt_file, map_location = 'cpu')
        # checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
        LOGGER.info(f'load_from_pretrained: {ckpt_file}')


    return checkpoint


def load_from_resume(run_cfg):
    ckpt_dir = os.path.join(run_cfg.output_dir,'ckpt')
    previous_optimizer_state = [i  for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
    steps = [i.split('.pt')[0].split('_')[-1] for i in  previous_optimizer_state] 
    steps = [ int(i) for i in steps]
    steps.sort()
    previous_step = steps[-1]
    previous_optimizer_state = f'optimizer_step_{previous_step}.pt'
    previous_model_state = f'model_step_{previous_step}.pt'
    previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
    previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
    previous_model_state = os.path.join(ckpt_dir, previous_model_state)
    
    assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
    LOGGER.info("choose previous model: {}".format(previous_model_state))
    LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    previous_model_state = torch.load(previous_model_state,map_location='cpu')
    previous_optimizer_state = torch.load(previous_optimizer_state,map_location='cpu')
    return previous_model_state, previous_optimizer_state, previous_step






