import os
import datetime
import torch
from mpformer.data_provider import datasets_factory
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from mpformer.models.model_factory import Model
from torch.cuda.amp import autocast
from mpformer.utils.data_normalization import ZNorm
from torchinfo import summary
from pytorch_lightning.loggers import CSVLogger

def train_pytorch_loader(configs):
    norm = ZNorm()
    csv_logger = CSVLogger(save_dir="lightning_logs", name="", version=None)

    train_loader = datasets_factory.data_provider(configs)
    # print(len(train_loader))
    val_loader = datasets_factory.data_provider_val(configs)
    # print(len(val_loader))

    # 是否标准化
    # train_loader.dataset.transform = norm
    # val_loader.dataset.transform = norm

    model = Model(configs)
    
    # 如果配置中有预训练权重路径，加载预训练权重
    if configs.pretrained_model and os.path.exists(configs.pretrained_model):
        print(f"Loading pretrained weights from {configs.pretrained_model}")
        checkpoint = torch.load(configs.pretrained_model)
        model.load_state_dict(checkpoint['state_dict'], strict=False)  # 加载预训练权重
        print("Pretrained weights loaded successfully.")

        # # 添加 adapter 结构到模型中
        # model = add_adapter(model, configs)
        # print("Adapter added to the model.")
        
    # 假设 input_size 代表的输入张量形状为 torch.Size([8, 9, 512, 512, 2])
    input_size = (1, 9, 512, 512, 2)

    # 输出模型架构和每层的张量大小
    print("Model Architecture:")
    summary(model, input_size=input_size, col_names=("input_size", "output_size", "num_params", "trainable"), depth=6)

    checkpoint_callback = ModelCheckpoint(
        dirpath=configs.checkpoint_dir,
        filename='mpformer-{epoch:02d}-{val_neigh_csi2:.2f}',
        save_top_k=1, # 保存最好的一个模型
        mode='max',
        monitor='val_neigh_csi2'
    )

    trainer = Trainer(
        max_epochs = configs.epochs,
        num_nodes = 1 if configs.device.startswith('cuda') else 0,
        callbacks=[checkpoint_callback],
        log_every_n_steps=configs.log_interval,
        # precision=16,   # 混合精度训练
        # accumulate_grad_batches=2,   # 梯度累积
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False, # 多GPU时同步BatchNorm
        strategy='ddp_find_unused_parameters_true',
        logger=csv_logger  # 设置日志记录器为 CSVLogger
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )