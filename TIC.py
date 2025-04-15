from typing import Dict, Tuple
import torch
import math
import torch

class TIC(object):
    def set_face_masks(self, face_masks):
        self.face_masks = face_masks
        # dilate the 32x32 and 64x64 masks
        kernel = torch.ones(3,3).to(self.pipe.device)
        face_mask_32_temp = face_masks[:,::16,::16]
        face_mask_64_temp = face_masks[:,::8,::8]
        
        # dilate twice
        for _ in range(2):
            face_mask_32_temp = torch.nn.functional.conv2d(
                face_mask_32_temp.float().unsqueeze(1), 
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze(1) > 0
            face_mask_64_temp = torch.nn.functional.conv2d(
                face_mask_64_temp.float().unsqueeze(1),
                kernel.unsqueeze(0).unsqueeze(0), 
                padding=1
            ).squeeze(1) > 0
            
        self.face_mask_32 = face_mask_32_temp
        self.face_mask_64 = face_mask_64_temp
        
    def __init__(self, pipe=None, 
                 num_steps=30,
                 cache_branch_id=0,
                 cache_interval=3,
                 cache_max_order=2,
                 first_enhance=2,
                 threshold=25,  
                 mask_style=None,
                 **model_kwargs):
        assert pipe is not None, "pipe is required"
        self.pipe = pipe
        self.face_masks = None
        self.threshold = threshold
        self.mask_style = mask_style
        if mask_style is None:
            self.face_mask_32 = torch.ones(32,32)>0
            self.face_mask_64 = torch.ones(64,64)>0
            
        self.cur_timestep = 0
        self.tile_num = 3
        self.function_dict = {}
        self.cached_output = {}
        cache_layer_id = cache_branch_id % 3
        cache_block_id = cache_branch_id // 3
        self.params = {
            'cache_interval': cache_interval,
            'cache_layer_id': cache_layer_id,
            'cache_block_id': cache_block_id,
            'skip_mode': 'uniform'
        }
        self.taylor_tags={
            ('up', 'block', 0, 0),
            ('up', 'attentions', 0, 0),
            ('up', 'motion_module', 0, 0),
            ('up', 'resnet', 0, 0),
            # ('mid', 'mid_block', 0, 0),
            ('down', 'block', 0, 0),
            ('down', 'attentions', 0, 0),
            ('down', 'motion_module', 0, 0),
            ('down', 'resnet', 0, 0),
            }
        cache_dic = {}
        cache = {}
        cache_dic['cache']                = cache
        cache_dic['flops']                = 0.0
        cache_dic['interval']             = cache_interval
        cache_dic['max_order']            = cache_max_order
        cache_dic['first_enhance']        = first_enhance
        
        current = {}
        current['num_steps'] = num_steps
        current['tile_idx'] = -1
        current['activated_steps'] = [-1]
        current['order'] = 0
        self.cache_dic = cache_dic
        self.current = current
    def set_tile_num(self, tile_num):
        self.tile_num = tile_num
    def fit_taylor_cache(self, blocktype, block_name, block_i, layer_i):
        return (blocktype, block_name, block_i, layer_i) in self.taylor_tags
    def wrap_unet_forward(self):
        self.function_dict['unet_forward'] = self.pipe.unet.forward
        def wrapped_forward(*args, **kwargs):
            if args[1].shape == torch.Size([]) or args[1].shape == torch.Size([1]):
                tid=args[1].item()
            else:
                tid=args[1][0].item()
            self.cur_timestep = list(self.pipe.scheduler.timesteps).index(tid)
            self.current['step'] = self.cur_timestep
            self.current['tile_idx'] =(self.current['tile_idx']+1)%self.tile_num
            self.cal_type()
            result = self.function_dict['unet_forward'](*args, **kwargs)
            return result
        self.pipe.unet.forward = wrapped_forward
        
    def is_skip_step(self, block_i, layer_i, blocktype = "down"):
        cache_interval, cache_layer_id, cache_block_id, skip_mode = \
            self.params['cache_interval'], self.params['cache_layer_id'], self.params['cache_block_id'], self.params['skip_mode']
        if skip_mode == 'uniform':
            if self.cur_timestep % cache_interval == 0: return False
        if block_i > cache_block_id or blocktype == 'mid':
            return True
        if block_i < cache_block_id: 
            return False
        return layer_i >= cache_layer_id if blocktype == 'down' else layer_i > cache_layer_id
        
    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype = "down"):
        self.function_dict[
            (blocktype, block_name, block_i, layer_i)
        ] = block.forward
        def wrapped_forward(*args, **kwargs):
            tile_idx=self.current['tile_idx']
            if self.current['type'] == 'Taylor':
                if self.fit_taylor_cache(blocktype, block_name, block_i, layer_i):
                    result = self.taylor_formula((blocktype, block_name, block_i, layer_i,tile_idx))
                elif self.is_skip_step(block_i, layer_i, blocktype):
                    result = self.cached_output[(blocktype, block_name, block_i, layer_i,tile_idx)]
                else:
                    result = self.function_dict[(blocktype, block_name,  block_i, layer_i)](*args, **kwargs)
            else:
                result = self.function_dict[(blocktype, block_name,  block_i, layer_i)](*args, **kwargs)
            if self.current['type'] == 'full': 
                if self.fit_taylor_cache(blocktype, block_name, block_i, layer_i):
                    self.taylor_cache_init()
                    self.derivative_approximation((blocktype, block_name, block_i, layer_i,tile_idx),result)
                else:
                    self.cached_output[(blocktype, block_name, block_i, layer_i,tile_idx)] = result
            return result
        block.forward = wrapped_forward

    def wrap_modules(self):
        # 1. wrap unet forward
        self.wrap_unet_forward()
        # 2. wrap downblock forward
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(attention, "attentions", block_i, layer_i)
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(resnet, "resnet", block_i, layer_i)
            for (layer_i, motion_module) in enumerate(getattr(block, "motion_modules", [])):
                self.wrap_block_forward(motion_module, "motion_module", block_i, layer_i)
            downsamplers = getattr(block, "downsamplers", []) or []
            for (layer_i, downsampler) in enumerate(downsamplers):
                self.wrap_block_forward(downsampler, "downsampler", block_i, layer_i)
            self.wrap_block_forward(block, "block", block_i, 0, blocktype = "down")
            
        # ================================================
        # 3. wrap midblock forward
        self.wrap_block_forward(self.pipe.unet.mid_block, "mid_block", 0, 0, blocktype = "mid")
        # ================================================
        
        # 4. wrap upblock forward
        block_num = len(self.pipe.unet.up_blocks) #4
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(attention, "attentions", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(resnet, "resnet", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            for (layer_i, motion_module) in enumerate(getattr(block, "motion_modules", [])):
                self.wrap_block_forward(motion_module, "motion_module", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            upsamplers = getattr(block, "upsamplers", []) or []
            for (layer_i, upsampler) in enumerate(upsamplers):
                self.wrap_block_forward(upsampler, "upsampler", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            self.wrap_block_forward(block, "block", block_num - block_i - 1, 0, blocktype = "up")
            
    def recursive_subtract(self, x, y, div=1):
        """
        recursively subtract x and y, keep the same structure
        Args:
            x: can be tensor, tuple or object with .sample attribute
            y: object with the same structure as x
            div: divisor
        Returns:
            z: result with the same structure as x and y
        """
        if isinstance(x, tuple):
            # if tuple, recursively process each element
            return tuple(self.recursive_subtract(xi, yi, div) for xi, yi in zip(x, y))
        elif hasattr(x, 'sample'):
            # if has sample attribute, subtract sample
            result = type(x)(sample=(x.sample - y.sample)/div)
            return result
        else:
            return x - y
        
        
    def error_inspect(self,x):
        if isinstance(x, tuple):
            return tuple(self.error_inspect(xi) for xi in x)
        elif hasattr(x, 'sample'):
            return self.error_inspect(x.sample)
        else:
            n=torch.isnan(x).sum().item()
            assert n==0, f"error: {n}"
            return n
        
    def derivative_approximation(self,tag_tuple:Tuple,feature: torch.Tensor):
        # cache write
        current = self.current
        cache_dic = self.cache_dic
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
        updated_taylor_factors = {}
        updated_taylor_factors[0] = feature
                
        for i in range(cache_dic['max_order']):
            if (
                cache_dic['cache'].get(tag_tuple, None) is not None and \
                cache_dic['cache'][tag_tuple].get(i, None) is not None and \
                current['step'] >= cache_dic['first_enhance']
            ):
                updated_taylor_factors[i + 1] = self.recursive_subtract(updated_taylor_factors[i], cache_dic['cache'][tag_tuple][i], difference_distance)
            else:
                break
        
        cache_dic['cache'][tag_tuple] = updated_taylor_factors

    def taylor_formula(self,tag_tuple:Tuple) -> torch.Tensor: 
        # cache read
        current = self.current
        cache_dic = self.cache_dic
        x = current['step'] - current['activated_steps'][-1]
        output = 0
        def recursive_taylor_term(a,b,m):
            """recursive taylor term"""
            if isinstance(a, float):
                raise NotImplementedError("recursive_multiply is not implemented")
            if isinstance(b, tuple):
                return tuple(recursive_taylor_term(term1, term2, m) for term1,term2 in zip(a,b))
            elif hasattr(b, 'sample'):
                result = type(b)(sample=recursive_taylor_term(a.sample, b.sample, m))
                return result
            else:
                B,C,T,H,W = b.shape
                assert H in [32,64], "H must be 32 or 64"
                start_frame = self.current['tile_idx'] * 16
                end_frame = start_frame + 16
                t = torch.zeros_like(b)  # set all to 0
                if self.face_masks is not None:
                    if self.face_masks.ndim == 3:
                        if H == 32:
                            t[:,:,self.face_mask_32[start_frame:end_frame]] = b[:,:,self.face_mask_32[start_frame:end_frame]] * m  # only multiply m in the mask position
                        else:
                            t[:,:,self.face_mask_64[start_frame:end_frame]] = b[:,:,self.face_mask_64[start_frame:end_frame]] * m  # only multiply m in the mask position
                    else:
                        if H == 32:
                            t[:,:,:,self.face_mask_32] = b[:,:,:,self.face_mask_32] * m  # only multiply m in the mask position
                        else:
                            t[:,:,:,self.face_mask_64] = b[:,:,:,self.face_mask_64] * m  # only multiply m in the mask position
                else:
                    t = b * m
                return a+t
        output = cache_dic['cache'][tag_tuple][0]
        L=min(current['order'],len(cache_dic['cache'][tag_tuple]))
        for i in range(1,L):
            m=(1 / math.factorial(i))*(x**i)
            output = recursive_taylor_term(output,cache_dic['cache'][tag_tuple][i], m)
        return output

    def taylor_cache_init(self):
        current = self.current
        cache_dic = self.cache_dic
        if current['step'] == (current['num_steps'] - 1):
            cache_dic['cache']= {}

    def cal_type(self):
        '''
        Determine calculation type for this step
        '''
        current = self.current
        cache_dic = self.cache_dic
        current_step = self.cur_timestep
        last_steps = (current_step >= (current['num_steps'] - 1))
        first_steps = (current_step < cache_dic['first_enhance'])
        fresh_interval = cache_dic['interval']
        current['order'] =0 if current_step < self.threshold else cache_dic['max_order']
        if (first_steps) or (self.cur_timestep % cache_dic['interval'] == 0) or (last_steps):
            current['type'] = 'full'
            if current['activated_steps'][-1] != current_step:
                current['activated_steps'].append(current_step)
        else:
            current['type'] = 'Taylor'