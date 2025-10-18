#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Callable, Dict, List, Union

import torch
from PIL.Image import Image
from torch import Tensor

from lightly.utils import dependency as _dependency

if _dependency.torchvision_transforms_v2_available():
    from torchvision.transforms import v2 as _torchvision_transforms

    _TRANSFORMS_V2 = True
else:
    from torchvision import transforms as _torchvision_transforms

    _TRANSFORMS_V2 = False


class ToTensor:
    """Convert a PIL Image to a tensor with value normalization, similar to [0].

    This implementation is required since `torchvision.transforms.v2.ToTensor` is
    deprecated and will be removed in the future (see [1]).

    Input to this transform:
        PIL Image (H x W x C) of uint8 type in range [0,255]

    Output of this transform:
        torch.Tensor (C x H x W) of type torch.float32 in range [0.0, 1.0]

    - [0] https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    - [1] https://pytorch.org/vision/0.20/generated/torchvision.transforms.v2.ToTensor.html?highlight=totensor#torchvision.transforms.v2.ToTensor
    """

    def __init__(self) -> None:
        T = _torchvision_transforms
        if _TRANSFORMS_V2:
            self.transform: Callable[..., Tensor] = T.Compose(
                [T.ToImage(), T.ToDtype(dtype=torch.float32, scale=True)]
            )
        else:
            self.transform = T.ToTensor()

    def __call__(
        self,
        *args: Union[Tensor, Image],
        **kwargs: Dict[str, Any],
    ) -> Tensor:
        return self.transform(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.transform})"


class DeprecatedShim:
    """Shim class to replace deprecated transforms.

    This replaces existing, (soon to be) deprecated transforms with custom
    input transforms for compatibility reasons.

    Attributes:
        custom_transforms (dict of str: Callable[..., Any])
    """

    def __init__(self, custom_transforms: Dict[str, Callable[..., Any]]):
        self._torchvision_transforms = _torchvision_transforms
        self._custom_transforms = custom_transforms

    def __getattr__(self, name: str) -> Any:
        if name in self._custom_transforms:
            return self._custom_transforms[name]
        return getattr(self._torchvision_transforms, name)

    def __dir__(self) -> List[str]:
        return dir(self._torchvision_transforms)


# Set the compatibility layer to the shim, providing
# the functions to replace.
torchvision_transforms: Any = DeprecatedShim(dict(ToTensor=ToTensor))
functional = torchvision_transforms.functional
