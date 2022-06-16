from kornia.augmentation.base import MixAugmentationBase, AugmentationBase2D
from kornia.constants import Resample, BorderType, SamplePadding
from kornia.augmentation import random_generator as rg
from typing import Callable, Tuple, Union, List, Optional, Dict, cast
import numpy as np
import torch

class RandomCropSlice(MixAugmentationBase):


    def __init__(self, height: int, width: int, depth: int,
                 cut_size = None, num_mix: int = 1, full_h : int = 64, full_w : int = 64, full_d : int = 28,
                 beta: Optional[Union[torch.Tensor, float]] = None, same_on_batch: bool = False,
                 p: float = 1., transf_labels: bool = False) -> None:
        super(RandomCropSlice, self).__init__(p=1., p_batch=p)
        self.height = height
        self.width = width
        self.depth = depth
        self.full_h = full_h
        self.full_w = full_w
        self.full_d = full_d
        self.p = p
        self.transf_labels = transf_labels
        self.same_on_batch = same_on_batch 
        self.num_mix = num_mix
        if beta is None:
            self.beta = torch.tensor(1.)
        else:
            self.beta = cast(torch.Tensor, beta) if isinstance(beta, torch.Tensor) else torch.tensor(beta)
            
        if cut_size is None:
            self.cut_size = torch.tensor([0.1, 0.5])
        else:
            self.cut_size = cut_size
    def __repr__(self) -> str:
        repr = f"cut_size={self.cut_size}, "
        f"height={self.height}, width={self.width}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"
    
    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_cutmix_generator(batch_shape[0], width=self.width, height=self.height, p=self.p,
                                          cut_size=self.cut_size, num_mix=self.num_mix, beta=self.beta,
                                          same_on_batch=self.same_on_batch)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        batch_size = input.shape[0]
        out_inputs = torch.zeros(batch_size, self.depth, self.width, self.height).to(torch.device('cuda:0'))

        if self.same_on_batch:
            ax = np.random.randint(0, self.full_w - self.width + 1)
            ay = np.random.randint(0, self.full_h - self.height + 1)
            d = np.random.randint(0, self.full_d - self.depth + 1)
            out_inputs = input[:, d:(d+self.depth), ax:(ax+self.width), ay:(ay+self.height)].float()
        else:
            for i in range(batch_size):
                ax = np.random.randint(0, self.full_w - self.width + 1)
                ay = np.random.randint(0, self.full_h - self.height + 1)
                d = np.random.randint(0, self.full_d - self.depth + 1)
                out_inputs[i, :, :,:] = input[i, d:(d+self.depth), ax:(ax+self.width), ay:(ay+self.height)]

        out_inputs = out_inputs[:, None, ...]
        return out_inputs, label

class RandomLocalCutMix(MixAugmentationBase):


    def __init__(self, height: int, width: int,
                 cut_size = None, num_mix: int = 1,
                 beta: Optional[Union[torch.Tensor, float]] = None, same_on_batch: bool = False,
                 p: float = 1., transf_labels: bool = False, dataset_type = "2D") -> None:
        super(RandomLocalCutMix, self).__init__(p=1., p_batch=p)
        self.height = height
        self.width = width
        self.depth = 28 if dataset_type == '3D' else 1
        self.p = p
        self.transf_labels = transf_labels
        
        self.num_mix = num_mix
        if beta is None:
            self.beta = torch.tensor(1.)
        else:
            self.beta = cast(torch.Tensor, beta) if isinstance(beta, torch.Tensor) else torch.tensor(beta)
            
        if cut_size is None:
            self.cut_size = torch.tensor([0.1, 0.5])
        else:
            self.cut_size = cut_size
    def __repr__(self) -> str:
        repr = f"cut_size={self.cut_size}, "
        f"height={self.height}, width={self.width}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"
    
    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_cutmix_generator(batch_shape[0], width=self.width, height=self.height, p=self.p,
                                          cut_size=self.cut_size, num_mix=self.num_mix, beta=self.beta,
                                          same_on_batch=self.same_on_batch)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        batch_size = input.shape[0]
        height_max_cut = torch.round(self.height*self.cut_size[1])
        width_max_cut = torch.round(self.width*self.cut_size[1])
        height_min_cut = torch.round(self.height*self.cut_size[0])
        width_min_cut = torch.round(self.width*self.cut_size[0])
        out_inputs = input.clone()
        out_labels = label
        
        for i in range(batch_size):
            if torch.rand(1) < self.p:
                 h_cut = np.random.randint(height_min_cut, height_max_cut)
                 w_cut = np.random.randint(width_min_cut, width_max_cut)
                 
                 s1_ax = np.random.randint(0, self.width - w_cut)
                 s1_ay = np.random.randint(0, self.height - h_cut)
                 
                 s2_ax = np.random.randint(0, self.width - w_cut)
                 s2_ay = np.random.randint(0, self.height - h_cut)
                 
                 if self.depth > 1:
                     depth_max_cut = torch.round(self.depth*self.cut_size[1])
                     depth_min_cut = torch.round(self.depth*self.cut_size[0])
                     d_cut = np.random.randint(depth_min_cut, depth_max_cut)
                     
                     s1_ad = np.random.randint(0, self.depth - d_cut)
                     s2_ad = np.random.randint(0, self.depth - d_cut)
                     
                     out_inputs[i, s1_ad:(s1_ad + d_cut), s1_ax:(s1_ax + w_cut), s1_ay:(s1_ay + h_cut)] = input[i, s2_ad:(s2_ad + d_cut), s2_ax:(s2_ax + w_cut), s2_ay:(s2_ay + h_cut)]
                     out_inputs[i, s2_ad:(s2_ad + d_cut), s2_ax:(s2_ax + w_cut), s2_ay:(s2_ay + h_cut)] = input[i, s1_ad:(s1_ad + d_cut), s1_ax:(s1_ax + w_cut), s1_ay:(s1_ay + h_cut)]
                     
                 else:    
                     out_inputs[i, :, s1_ax:(s1_ax + w_cut), s1_ay:(s1_ay + h_cut)] = input[i, :, s2_ax:(s2_ax + w_cut), s2_ay:(s2_ay + h_cut)]
                     out_inputs[i, :, s2_ax:(s2_ax + w_cut), s2_ay:(s2_ay + h_cut)] = input[i, :, s1_ax:(s1_ax + w_cut), s1_ay:(s1_ay + h_cut)]
                 
                 if self.transf_labels:
                     out_labels = label.clone()
                     out_labels[i, s1_ax:(s1_ax + w_cut), s1_ay:(s1_ay + h_cut)] = label[i, s2_ax:(s2_ax + w_cut), s2_ay:(s2_ay + h_cut)]
                     out_labels[i, s2_ax:(s2_ax + w_cut), s2_ay:(s2_ay + h_cut)] = label[i, s1_ax:(s1_ax + w_cut), s1_ay:(s1_ay + h_cut)]
        return out_inputs, out_labels
        
class RandomCropS(AugmentationBase2D):


    def __init__(
        self, size: Tuple[int, int], padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None,
        pad_if_needed: Optional[bool] = False, fill: int = 0, padding_mode: str = 'constant',
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = False, p: float = 1.0
    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super(RandomCropS, self).__init__(
            p=1., return_transform=return_transform, same_on_batch=same_on_batch, p_batch=p)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.resample = Resample.get(resample)
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = (f"crop_size={self.size}, padding={self.padding}, fill={self.fill}, pad_if_needed={self.pad_if_needed}, "
                f"padding_mode={self.padding_mode}, resample={self.resample.name}")
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), self.size,
                                        same_on_batch=self.same_on_batch)

    def precrop_padding(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding is not None:
            if isinstance(self.padding, int):
                self.padding = cast(int, self.padding)
                padding = [self.padding, self.padding, self.padding, self.padding]
            elif isinstance(self.padding, tuple) and len(self.padding) == 2:
                self.padding = cast(Tuple[int, int], self.padding)
                padding = [self.padding[1], self.padding[1], self.padding[0], self.padding[0]]
            elif isinstance(self.padding, tuple) and len(self.padding) == 4:
                self.padding = cast(Tuple[int, int, int, int], self.padding)
                padding = [self.padding[3], self.padding[2], self.padding[1], self.padding[0]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-2] < self.size[0]:
            padding = [0, 0, (self.size[0] - input.shape[-2]), self.size[0] - input.shape[-2]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-1] < self.size[1]:
            padding = [self.size[1] - input.shape[-1], self.size[1] - input.shape[-1], 0, 0]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        return input

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_crop_transformation(input, params, self.flags)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_crop(input, params, self.flags), F.apply_crop(label, params, self.flags)

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                params: Optional[Dict[str, torch.Tensor]] = None, return_transform: Optional[bool] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if type(input) == tuple:
            input = (self.precrop_padding(input[0]), input[1])
        else:
            input = cast(torch.Tensor, input)
            input = self.precrop_padding(input)
        return super().forward(input, params, return_transform)        
