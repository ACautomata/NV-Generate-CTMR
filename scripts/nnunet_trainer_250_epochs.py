#!/usr/bin/env python3
"""Issue #35 训练契约所需的可安装 nnU-Net trainer 变体。

在发起正式训练前，将本模块复制至已安装的 ``nnunetv2`` trainer variants
package。它只改变 upstream trainer 的 epoch 数。

``__init__`` 必须显式镜像 upstream 签名而非 ``*args/**kwargs`` 转发：
upstream 以 ``inspect.signature(self.__init__)`` 的参数名索引 ``locals()``
来重建 ``my_init_kwargs``，转发式子类的签名键（``args``/``kwargs``）在
base 帧的 ``locals()`` 中不存在，会在构造时抛 ``KeyError``。这与
nnunetv2 自带 ``nnUNetTrainer_Xepochs`` variants 的写法一致。
"""

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer250Epochs(nnUNetTrainer):
    """恰好运行 250 个 epoch，同时保留 upstream nnU-Net 其余训练配方。"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
