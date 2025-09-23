from neuralforecast.models.tsmixerx import *
from neuralforecast.losses.pytorch import MAE
from typing import Optional
import torch
import torch.nn as nn



class TSMixerXEmbedding(TSMixerx):
    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
            self,
            h,
            input_size,
            n_series,
            futr_exog_list=None,
            hist_exog_list=None,
            stat_exog_list=None,
            exclude_insample_y=False,
            org_vocab_sizes=None,
            max_embedding_dim=50,
            n_block=2,
            ff_dim=8,
            dropout=0.0,
            revin=True,
            loss=MAE(),
            valid_loss=None,
            max_steps: int = 5000,
            learning_rate: float = 1e-3,
            num_lr_decays: int = -1,
            early_stop_patience_steps: int = -1,
            val_check_steps: int = 100,
            batch_size: int = 1,
            valid_batch_size: Optional[int] = None,
            windows_batch_size=32,
            inference_windows_batch_size=32,
            start_padding_enabled=False,
            step_size: int = 1,
            scaler_type: str = "identity",
            random_seed: int = 1,
            drop_last_loader: bool = False,
            alias: Optional[str] = None,
            optimizer=None,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            dataloader_kwargs=None,
            **trainer_kwargs
    ):

        # 保存初始化参数
        self.init_args = dict(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=list(futr_exog_list) if futr_exog_list is not None else [],
            hist_exog_list=list(hist_exog_list) if hist_exog_list is not None else [],
            stat_exog_list=list(stat_exog_list) if stat_exog_list is not None else [],
            exclude_insample_y=exclude_insample_y,
            org_vocab_sizes=org_vocab_sizes,
            max_embedding_dim=max_embedding_dim,
            n_block=n_block,
            ff_dim=ff_dim,
            dropout=dropout,
            revin=revin,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )

        super().__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=list(futr_exog_list) if futr_exog_list is not None else [],
            hist_exog_list=list(hist_exog_list) if hist_exog_list is not None else [],
            stat_exog_list=list(stat_exog_list) if stat_exog_list is not None else [],
            exclude_insample_y=exclude_insample_y,
            n_block=n_block,
            ff_dim=ff_dim,
            dropout=dropout,
            revin=revin,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )

        self.feature_mixer_hist_single_target = FeatureMixing(
            in_features=2,
            out_features=ff_dim,
            h=input_size,
            dropout=dropout,
            ff_dim=ff_dim,
        )

        mixing_layers = [
            MixingLayer(
                in_features=ff_dim,
                out_features=ff_dim,
                h=input_size,
                dropout=dropout,
                ff_dim=ff_dim,
            )
            for _ in range(n_block)
        ]
        self.mixing_block_single_target = nn.Sequential(*mixing_layers)

        self.fc = nn.Linear(
            input_size * ff_dim, n_series
        )
    def forward(self, windows_batch):
        # Parse batch
        x = windows_batch[
            "main"
        ]  # [batch_size (B), input_size (L)]
        sup = windows_batch[
            "sup"
        ]  # [batch_size (B), input_size (L)]
        batch_size, input_size = x.shape[:2]

        x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        sup = sup.unsqueeze(-1)  # [B, L] -> [B, L, 1]

        x = torch.cat(
            (x, sup), dim=-1
        ) # [B, input_size (L), 1] -> [B, input_size (L), 2]

        x = self.feature_mixer_hist_single_target(x)  # 单目标：[B, L, 2] -> [B, L, ff_dim]

        # N blocks of mixing layers

        x = self.mixing_block_single_target(x)  # [B, L, ff_dim] -> [B, L, ff_dim]

        x = x.reshape(
                batch_size, -1
            ) # [B, L, ff_dim] -> [B, L * ff_dim]
        # Fully connected output layer
        y = self.fc(x)  # [B, L * ff_dim -> [B, class_num]

        return y