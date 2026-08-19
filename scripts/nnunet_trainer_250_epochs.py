#!/usr/bin/env python3
"""Issue #35 训练契约所需的可安装 nnU-Net trainer 变体。

在发起正式训练前，将本模块复制至已安装的 ``nnunetv2`` trainer variants
package。它只改变 upstream trainer 的 epoch 数。
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer250Epochs(nnUNetTrainer):
    """恰好运行 250 个 epoch，同时保留 upstream nnU-Net 其余训练配方。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 250
