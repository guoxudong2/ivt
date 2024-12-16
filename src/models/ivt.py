from tabnanny import check

from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
from torch import nn
from omegaconf.errors import MissingMandatoryValue
import torch.nn.functional as F
from hydra.utils import instantiate
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Dataset, Subset
from src.models.utils_transformer import TransformerEncoderBlock, TransformerDecoderBlock, AbsolutePositionalEncoding, \
    RelativePositionalEncoding
from src.models.utils import grad_reverse
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from functools import partial
import seaborn as sns
from sklearn.manifold import TSNE

from src.models.edct import EDCT
from src.models.time_varying_model import TimeVaryingCausalModel, BRCausalModel
from src.models.utils_transformer import TransformerMultiInputBlock, OnlyIVTransformerMultiInputBlock, LayerNorm
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import BRTreatmentOutcomeHead, bce

logger = logging.getLogger(__name__)

class BuildTreatmentDistribution(nn.Module):
    def __init__(self, args: DictConfig):
        """
        初始化治疗生成网络
        :param args: 配置参数，包含 br_size, dim_iv, fc_hidden_units, n_components
        """
        super(BuildTreatmentDistribution, self).__init__()
        self.br_size = args.br_size
        self.dim_iv = args.dim_iv
        self.fc_hidden_units = args.fc_hidden_units
        self.n_components = args.n_components  # 对应离散类别数，等价于治疗类型数

        # 定义 pi 网络，用于生成离散分布
        input_dim = self.br_size + self.dim_iv  # 输入特征维度 = BR 特征维度 + IV 特征维度
        self.pi_layers = nn.Sequential(
            nn.Linear(input_dim, self.fc_hidden_units),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fc_hidden_units, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, self.fc_hidden_units),
            nn.ELU(),
            nn.Linear(self.fc_hidden_units, self.n_components),  # 输出 n_components（离散类别数）
            #nn.Softmax(dim=-1)  # 归一化为概率分布
        )

    def forward(self, br, iv):
        """
        前向传播，生成离散分布的 pi
        :param br: 特征表示 (batch_size, time_steps, br_dim)
        :param iv: 工具变量 (batch_size, time_steps, iv_dim)
        :return: pi (batch_size, time_steps, n_components)
        """
        combined_features = torch.cat([br, iv], dim=-1)  # 拼接 BR 和 IV 特征
        logits = self.pi_layers(combined_features)  # 通过 pi 网络生成概率分布
        logits = torch.clamp(logits, min=-10, max=10)
        return logits

class FirstStageHead(nn.Module):
    def __init__(self, args: DictConfig):
        super(FirstStageHead, self).__init__()
        self.seq_hidden_units = args.seq_hidden_units
        self.br_size = args.br_size
        self.dim_iv = args.dim_iv

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.bn1 = nn.BatchNorm1d(self.br_size)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.1)

        self.linear2 = nn.Linear(self.seq_hidden_units, self.seq_hidden_units)
        self.bn_chemo_iv = nn.BatchNorm1d(self.seq_hidden_units)
        self.elu2 = nn.ELU()
        self.linear3 = nn.Linear(self.seq_hidden_units, self.dim_iv)
        self.dropout2 = nn.Dropout(0.1)

        self.linear4 = nn.Linear(self.seq_hidden_units, self.seq_hidden_units)
        self.bn_radio_iv = nn.BatchNorm1d(self.seq_hidden_units)
        self.elu3 = nn.ELU()
        self.linear5 = nn.Linear(self.seq_hidden_units, self.dim_iv)
        self.dropout3 = nn.Dropout(0.1)

        self.build_chemo_treatment_distr = BuildTreatmentDistribution(args)
        self.build_radio_treatment_distr = BuildTreatmentDistribution(args)

    def forward(self, br, iv):
        print('seq 6')
        pi_logits = self.build_chemo_treatment_distr(br, iv)
        return pi_logits

    def transform_br(self, seq_output):
        print('seq 3')
        batch_size, seq_length, _ = seq_output.size()
        x = self.linear1(seq_output)  # x: (batch_size, seq_length, br_size)

        # 调整维度以匹配 BatchNorm1d 的输入
        x = x.view(-1, self.br_size)  # (batch_size * seq_length, br_size)
        x = self.bn1(x)
        x = x.view(batch_size, seq_length, self.br_size)  # 调整回原来的维度

        x = self.elu1(x)
        return x
        '''br = self.elu1(self.linear1(seq_output))
        br = self.dropout1(br)
        return br'''

    '''def transform_iv(self, seq_chemo_iv_output, seq_radio_iv_output):
        print('seq 5')
        chemo_iv = self.linear3(self.elu2(self.linear2(seq_chemo_iv_output)))
        radio_iv = self.linear5(self.elu3(self.linear4(seq_radio_iv_output)))
        chemo_iv = self.dropout2(chemo_iv)
        radio_iv = self.dropout3(radio_iv)
        return chemo_iv, radio_iv'''

    def transform_iv(self, seq_iv_output):
        batch_size, seq_length, _ = seq_iv_output.size()

        # 化疗 IV 处理
        x = self.linear2(seq_iv_output)
        x = x.view(-1, self.seq_hidden_units)
        x = self.bn_chemo_iv(x)
        x = x.view(batch_size, seq_length, self.seq_hidden_units)
        x = self.elu2(x)
        x = self.dropout2(x)
        iv = self.linear3(x)

        return iv

class SecondStageHead(nn.Module):
    def __init__(self, args: DictConfig, dim_treatments, dim_outcome):
        super(SecondStageHead, self).__init__()
        self.treatment_types = args.treatment_types
        self.seq_hidden_units = args.seq_hidden_units
        self.br_size = args.br_size
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome

        self.linear1 = nn.Linear(self.br_size + self.dim_treatments, self.seq_hidden_units)
        self.bn1 = nn.BatchNorm1d(self.seq_hidden_units)
        self.elu1 = nn.ELU()
        self.linear2 = nn.Linear(self.seq_hidden_units, self.dim_outcome)
        self.bn2 = nn.BatchNorm1d(self.dim_outcome)
        self.elu2 = nn.ELU()
        self.dropout1 = nn.Dropout(0.1)

    '''def build_outcome(self, br, chemo_treatment, radio_treatment):
        print('seq 8')
        x = torch.cat((br, chemo_treatment, radio_treatment), dim=-1)
        x = self.elu1(self.linear1(x))
        outcome = self.elu2(self.linear2(x))
        outcome = self.dropout1(outcome)
        return outcome'''

    def build_outcome(self, br, treatment):
        # br: (batch_size, seq_length, br_size)
        # chemo_treatment, radio_treatment: (batch_size, seq_length, treatment_dim)

        x = torch.cat((br, treatment), dim=-1)  # (batch_size, seq_length, br_size + dim_treatments)
        batch_size, seq_length, _ = x.size()

        x = self.linear1(x)  # (batch_size, seq_length, seq_hidden_units)

        # 调整维度并应用 BatchNorm1d
        x = x.view(-1, self.seq_hidden_units)  # (batch_size * seq_length, seq_hidden_units)
        x = self.bn1(x)
        x = x.view(batch_size, seq_length, self.seq_hidden_units)

        x = self.elu1(x)

        x = self.linear2(x)  # (batch_size, seq_length, dim_outcome)

        # 调整维度并应用第二个 BatchNorm1d
        x = x.view(-1, self.dim_outcome)  # (batch_size * seq_length, dim_outcome)
        x = self.bn2(x)
        x = x.view(batch_size, seq_length, self.dim_outcome)

        x = self.elu2(x)
        return x

#class IVT(EDCT):
class IVT(BRCausalModel):
    model_type = 'ivt'
    possible_model_types = ['ivt']
    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None,
                 alpha: float = 0.0,
                 update_alpha: bool=True,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        assert self.autoregressive  # prev_outcomes are obligatory
        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args.model.ivt)
        self.first_stage_output_dropout = nn.Dropout(self.dropout_rate)
        self.chemo_iv_dropout = nn.Dropout(self.dropout_rate)
        self.radio_iv_dropout = nn.Dropout(self.dropout_rate)
        self.iv_dropout = nn.Dropout(self.dropout_rate)
        self.second_stage_output_dropout = nn.Dropout(self.dropout_rate)

        self.balancing = args.exp.balancing
        #self.alpha = alpha if not update_alpha else 0.0
        self.alpha = alpha
        self.alpha_max = alpha
        self.save_hyperparameters(args)
        # Used in hparam tuning
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome, self.dim_iv)
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')

    def init_first_stage(self, sub_args: DictConfig):
        self.first_stage_treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
        self.first_stage_vitals_input_transformation = \
            nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
        self.first_stage_vitals_input_transformation = nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
        self.first_stage_outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
        self.first_stage_static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)
        self.first_stage_chemo_iv_input_transformation = nn.Linear(self.dim_iv, self.seq_hidden_units)
        self.first_stage_radio_iv_input_transformation = nn.Linear(self.dim_iv, self.seq_hidden_units)

        self.first_stage_self_positional_encoding = self.first_stage_self_positional_encoding_k = self.first_stage_self_positional_encoding_v = None
        if sub_args.self_positional_encoding.absolute:
            self.first_stage_self_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.self_positional_encoding.trainable)
        else:
            # Relative positional encoding is shared across heads
            self.first_stage_self_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)
            self.first_stage_self_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)

        self.first_stage_cross_positional_encoding = self.first_stage_cross_positional_encoding_k = self.first_stage_cross_positional_encoding_v = None
        if 'cross_positional_encoding' in sub_args and sub_args.cross_positional_encoding.absolute:
            self.first_stage_cross_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.cross_positional_encoding.trainable)
        elif 'cross_positional_encoding' in sub_args:
            # Relative positional encoding is shared across heads
            self.first_stage_cross_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.cross_positional_encoding.trainable, cross_attn=True)
            self.first_stage_cross_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.cross_positional_encoding.trainable, cross_attn=True)

        self.first_stage_transformer_blocks = nn.ModuleList(
            [self.basic_block_cls(self.seq_hidden_units, self.num_heads, self.head_size, self.seq_hidden_units * 4,
                                  self.dropout_rate,
                                  self.dropout_rate if sub_args.attn_dropout else 0.0,
                                  self_positional_encoding_k=self.first_stage_self_positional_encoding_k,
                                  self_positional_encoding_v=self.first_stage_self_positional_encoding_v,
                                  n_inputs=self.n_inputs,
                                  disable_cross_attention=sub_args.disable_cross_attention,
                                  isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])

        self.iv_self_positional_encoding = self.iv_self_positional_encoding_k = self.iv_self_positional_encoding_v = None
        if sub_args.self_positional_encoding.absolute:
            self.iv_self_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.self_positional_encoding.trainable)
        else:
            # Relative positional encoding is shared across heads
            self.iv_self_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)
            self.iv_self_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)

        self.iv_block_cls = OnlyIVTransformerMultiInputBlock
        self.onlyiv_transformer_blocks = nn.ModuleList(
            [self.iv_block_cls(self.seq_hidden_units, self.num_heads, self.head_size, self.seq_hidden_units * 4,
                               self.dropout_rate,
                               self.dropout_rate if sub_args.attn_dropout else 0.0,
                               self_positional_encoding_k=self.iv_self_positional_encoding_k,
                               self_positional_encoding_v=self.iv_self_positional_encoding_v,
                               n_inputs=self.n_iv_inputs,
                               disable_cross_attention=sub_args.disable_cross_attention,
                               isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])

        self.first_stage_head = FirstStageHead(sub_args)


    def init_second_stage(self, sub_args: DictConfig, dim_treatments, dim_outcome):
        self.second_stage_head = SecondStageHead(sub_args, dim_treatments, dim_outcome)

    def _init_specific(self, sub_args: DictConfig):
        """
        Initialization of specific sub-network (only multi)
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            #super(IVT, self)._init_specific(sub_args)
            #------------------------------------
            self.dim_iv = sub_args.dim_iv
            self.max_seq_length = sub_args.max_seq_length
            self.br_size = sub_args.br_size  # balanced representation size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            # self.attn_dropout_rate = sub_args.attn_dropout_rate

            self.num_layer = sub_args.num_layer
            self.num_heads = sub_args.num_heads
            #------------------------------------

            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.head_size = sub_args.seq_hidden_units // sub_args.num_heads

            self.n_inputs = 3 if self.has_vitals else 2  # prev_outcomes and prev_treatments
            self.n_iv_inputs = 2
            self.init_first_stage(sub_args)
            self.init_second_stage(sub_args, self.dim_treatments, self.dim_outcome)

            # self.last_layer_norm = LayerNorm(self.seq_hidden_units)
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    '''def compute_discrete_loss(self,y_true, pi, active_entries=None):
        """
        计算离散分布的交叉熵损失
        :param y_true: 真实的 one-hot 数据，形状为 (batch_size, num_time_steps, num_treatments)
        :param pi: 预测的离散分布，形状为 (batch_size, num_time_steps, num_treatments)
        :param active_entries: 活跃掩码 (batch_size, num_time_steps, 1)
        :return: 标量损失
        """
        epsilon = 1e-8  # 避免 log(0) 的问题
        pi = torch.clamp(pi, min=epsilon, max=1 - epsilon)  # 确保数值稳定性
        print(f'y_true.shape:{y_true.shape},pi.shape:{pi.shape}')
        cross_entropy_loss = -(y_true * torch.log(pi)).sum(dim=-1)  # 计算交叉熵
        if active_entries is not None:
            cross_entropy_loss = cross_entropy_loss * active_entries.squeeze(-1)
            return cross_entropy_loss.sum() / active_entries.sum()
        return cross_entropy_loss.mean()'''

    '''def compute_discrete_loss(self, y_true, pi, active_entries=None):
        """
        计算离散分布的二元交叉熵损失。
        :param y_true: One-hot 编码的真实值 (batch_size, num_time_steps, num_treatments)。
        :param pi: 预测的分布 (batch_size, num_time_steps, num_treatments)。
        :param active_entries: 掩码 (batch_size, num_time_steps, 1)。
        :return: 标量损失值。
        """
        # 确保 pi 的数值稳定性
        epsilon = 1e-8
        pi = torch.clamp(pi, min=epsilon, max=1 - epsilon)

        # 计算交叉熵损失
        cross_entropy_loss = -(y_true * torch.log(pi) + (1 - y_true) * torch.log(1 - pi)).sum(
            dim=-1)  # (batch_size, num_time_steps)

        # 应用 active_entries 掩码
        if active_entries is not None:
            cross_entropy_loss = cross_entropy_loss * active_entries.squeeze(-1)  # 确保维度一致
            return cross_entropy_loss.sum() / active_entries.sum()  # 平均有效条目的损失

        return cross_entropy_loss.mean()  # 如果没有掩码，则计算所有条目的平均损失'''

    def compute_discrete_loss(self, y_true, logits, active_entries=None):
        print('seq 6.1')
        #print(f'y_true.shape:{y_true.shape},logits.shape:{logits.shape}')
        #print(f'y_true:{y_true},logits:{logits}')

        y_true_idx = torch.argmax(y_true, dim=-1)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_true_idx.view(-1),
            reduction='none'
        ).view(logits.size(0), logits.size(1))
        if active_entries is not None:
            ce_loss = ce_loss * active_entries.squeeze(-1)
            active_entries_sum = active_entries.sum()
            if active_entries_sum == 0:
                active_entries_sum = 1e-8
            return ce_loss.sum() / active_entries_sum
        return ce_loss.mean()

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def first_stage_build_br(self, prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split):
        print('seq 2')
        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:  # Test sequence data / Train augmented data
            for i in range(len(active_entries)):

                # Masking vitals in range [fixed_split: ]
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0
                vitals[i, int(fixed_split[i]):] = 0.0

        x_t = self.first_stage_treatments_input_transformation(prev_treatments)
        x_o = self.first_stage_outputs_input_transformation(prev_outputs)
        x_v = self.first_stage_vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.first_stage_static_input_transformation(static_features.unsqueeze(1))  # .expand(-1, x_t.size(1), -1)

        # if active_encoder_br is None and encoder_r is None:  # Only self-attention
        for block in self.first_stage_transformer_blocks:
            if self.first_stage_self_positional_encoding is not None:
                x_t = x_t + self.first_stage_self_positional_encoding(x_t)
                x_o = x_o + self.first_stage_self_positional_encoding(x_o)
                x_v = x_v + self.first_stage_self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None:  # Test seq data
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    # Masking vitals in range [fixed_split: ]
                    x[i, :int(fixed_split[i])] = \
                        (x_o[i, :int(fixed_split[i])] + x_t[i, :int(fixed_split[i])] + x_v[i, :int(fixed_split[i])]) / 3
                    x[i, int(fixed_split[i]):] = (x_o[i, int(fixed_split[i]):] + x_t[i, int(fixed_split[i]):]) / 2
            else:  # Train data always has vitals
                x = (x_o + x_t + x_v) / 3

        output = self.first_stage_output_dropout(x)
        return output
        #br = self.first_stage_head.transform_br(output)
        #return br

    def first_stage_build_iv(self, active_entries, chemo_iv, radio_iv, fixed_split):
        print('seq 4')
        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:  # Test sequence data / Train augmented data
            for i in range(len(active_entries)):

                # Masking vitals in range [fixed_split: ]
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0

        x_chemo_iv = self.first_stage_chemo_iv_input_transformation(chemo_iv)
        x_radio_iv = self.first_stage_radio_iv_input_transformation(radio_iv)

        # if active_encoder_br is None and encoder_r is None:  # Only self-attention
        for block in self.onlyiv_transformer_blocks:
            if self.iv_self_positional_encoding is not None:
                x_chemo_iv = x_chemo_iv + self.iv_self_positional_encoding(x_chemo_iv)
                x_radio_iv = x_radio_iv + self.iv_self_positional_encoding(x_radio_iv)
                #print(f'x_chemo_ivB:{x_chemo_iv}, x_radio_ivB:{x_radio_iv}')

            x_chemo_iv, x_radio_iv = block((x_chemo_iv, x_radio_iv), active_entries_treat_outcomes)
        #print(f'x_chemo_ivC:{x_chemo_iv}, x_radio_ivC:{x_radio_iv}')

        if not self.has_vitals:
            x = (x_chemo_iv + x_radio_iv) / 2
        else:
            if fixed_split is not None:  # Test seq data
                x = torch.empty_like(x_chemo_iv)
                for i in range(len(active_entries)):
                    # Masking vitals in range [fixed_split: ]
                    x[i, :int(fixed_split[i])] = \
                        (x_chemo_iv[i, :int(fixed_split[i])] + x_radio_iv[i, :int(fixed_split[i])]) / 2 #这没加vitals
                    x[i, int(fixed_split[i]):] = (x_chemo_iv[i, int(fixed_split[i]):] + x_radio_iv[i, int(fixed_split[i]):]) / 2
            else:
                x = (x_chemo_iv + x_radio_iv) / 2

        iv = self.iv_dropout(x)
        #iv = self.first_stage_head.transform_iv(x)
        return iv

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')

        predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))

        for t in range(self.hparams.dataset.projection_horizon + 1):
            logger.info(f't = {t + 1}')
            outputs_scaled = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data['future_past_split'][i])
                if t < self.hparams.dataset.projection_horizon:
                    dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        """
        Vizualizes attention scores
        :param dataset: dataset
        :param index: index of an instance
        :param artifacts_path: Path for saving
        """
        fig_keys = ['self_attention_o', 'self_attention_t', 'cross_attention_ot', 'cross_attention_to']
        if self.has_vitals:
            fig_keys += ['cross_attention_vo', 'cross_attention_ov', 'cross_attention_vt', 'cross_attention_tv',
                         'self_attention_v']
        self._visualize(fig_keys, dataset, index, artifacts_path)

    def get_predictions_first_stage(self, batch):
        print('seq 1')
        '''first_stage_params = [name for name, _ in self.named_parameters() if 'first_stage' in name or 'onlyiv_transformer_blocks' in name or 'iv_' in name]
        for name, param in self.named_parameters():
            if name in first_stage_params and param.requires_grad:
                if param.grad is None or torch.all(param.grad == 0):
                    print(f"Warning: First stage parameter {name} is not being updated!")'''

        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None

        # 训练时进行数据增强
        if self.training and self.hparams.model.ivt.augment_with_masked_vitals and self.has_vitals:
            assert fixed_split is None  # 仅用于训练数据
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries'])

            # 创建增强数据（将 vitals 随机遮掩，固定增强分界点）
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len  # 原始数据
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item()  # 增强数据

            # 增强 batch，将每个特征复制
            for k, v in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        active_entries = batch['active_entries']
        chemo_iv = batch['chemo_iv']
        radio_iv = batch['radio_iv']

        br = self.first_stage_build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
        br = self.first_stage_head.transform_br(br)
        iv = self.first_stage_build_iv(active_entries, chemo_iv, radio_iv, fixed_split)
        iv = self.first_stage_head.transform_iv(iv)
        pi = self.first_stage_head(br, iv)
        batch['br'] = br
        return pi

    def get_predictions_second_stage(self, batch, treatment=None):
        print('seq 7')
        '''second_stage_params = [name for name, _ in self.named_parameters() if 'second_stage' in name]
        for name, param in self.named_parameters():
            if name in second_stage_params and param.requires_grad:
                if param.grad is None or torch.all(param.grad == 0):
                    print(f"Warning: Second stage parameter {name} is not being updated!")'''

        # 训练时进行数据增强
        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None
        if self.training and self.hparams.model.ivt.augment_with_masked_vitals and self.has_vitals:
            # 为训练数据生成 vitals 掩盖增强数据
            assert fixed_split is None  # 仅用于训练数据
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries'])

            # 创建增强数据
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len  # 原始数据
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item()

            # 增强 batch，将每个特征复制
            for k, v in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        active_entries = batch['active_entries']

        #br = self.second_stage_build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
        if 'br' in batch and (self.training or getattr(self, 'validation', False)):
            br = batch['br']
            br = br.detach()  # 使用第一阶段存储的 br
        else:
            br = self.first_stage_build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
            br = self.first_stage_head.transform_br(br)
        outcome_pred = self.second_stage_head.build_outcome(br, treatment)
        return outcome_pred, br

    def log_updated_params(self, optimizer_idx):
        print(f"Checking parameters updated in optimizer {optimizer_idx}...")
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_mean = param.grad.mean()
                    grad_std = param.grad.std(unbiased=False)
                    print(f"Optimizer {optimizer_idx} updates: {name} -> Grad Mean: {grad_mean}, Grad Std: {grad_std}")
                    print(f"Gradient shape: {param.grad.shape}")
                    print(f"Gradient content: {param.grad}")
                else:
                    print(f"Optimizer {optimizer_idx} requires grad, but grad is None: {name}")

    def configure_optimizers(self):
        first_stage_params = []  # 用于存放 first_stage 的参数
        second_stage_params = []  # 用于存放 second_stage 的参数

        # 遍历所有模型参数，根据名字进行分类
        for name, param in self.named_parameters():
            if 'first_stage_' in name or 'onlyiv_transformer_blocks' in name or 'iv_' in name or 'first_stage_head' in name:
                # 归为第一阶段参数
                first_stage_params.append((name, param))
            elif 'second_stage_' in name:
                # 归为第二阶段参数
                second_stage_params.append((name, param))
            else:
                # 未明确属于第一阶段或第二阶段的参数
                print(f"Unclassified Parameter: {name} -> Shape: {param.shape}")

        # 打印分类结果
        print("\nFirst Stage Parameters (Treatment Head):")
        for name, param in first_stage_params:
            print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

        print("\nSecond Stage Parameters (Non-Treatment Head):")
        for name, param in second_stage_params:
            print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

        if self.hparams.exp.weights_ema:
            self.ema_non_treatment = ExponentialMovingAverage([par[1] for par in first_stage_params],
                                                          decay=self.hparams.exp.beta)
            self.ema_treatment = ExponentialMovingAverage([par[1] for par in second_stage_params],
                                                              decay=self.hparams.exp.beta)

        first_stage_optimizer = self._get_optimizer(first_stage_params)
        second_stage_optimizer = self._get_optimizer(second_stage_params) #如果用了all_head_params,即所有参数，则optimizer是所有参数的optimizer

        if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
            return self._get_lr_schedulers([first_stage_optimizer, second_stage_optimizer])

        all_named_parameters = set(name for name, _ in self.named_parameters())
        assigned_parameters = set(name for name, _ in first_stage_params + second_stage_params)
        unassigned_parameters = all_named_parameters - assigned_parameters
        print(f"Unassigned Parameters: {unassigned_parameters}")

        if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
            return self._get_lr_schedulers([first_stage_optimizer, second_stage_optimizer])

        return [first_stage_optimizer, second_stage_optimizer]

    '''def sample_from_pi(self, pi, n_samples=5):
        """
        从离散分布中采样 one-hot 数据
        :param pi: 预测的离散分布 (batch_size, num_time_steps, num_treatments)
        :return: 采样得到的 one-hot 数据 (batch_size, num_time_steps, num_treatments)
        """
        batch_size, num_time_steps, num_treatments = pi.shape
        sampled_indices = torch.multinomial(pi.view(-1, num_treatments), num_samples=1).view(batch_size, num_time_steps)
        one_hot_samples = F.one_hot(sampled_indices, num_classes=num_treatments).float()
        return one_hot_samples'''

    '''def sample_from_pi(self, pi_logits, n_samples=1):
        """
        从离散分布中采样 one-hot 数据。
        :param pi: 预测的离散分布 (batch_size, num_time_steps, num_treatments)。
        :param n_samples: 每个时间步的采样次数。
        :return: 采样得到的 one-hot 数据 (batch_size, num_time_steps, num_treatments)。
        """
        pi = torch.softmax(pi_logits, dim=-1)
        batch_size, num_time_steps, num_treatments = pi.shape
        # 多次采样组件索引
        sampled_indices = torch.multinomial(pi.view(-1, num_treatments), num_samples=n_samples, replacement=True)
        sampled_indices = sampled_indices.view(batch_size, num_time_steps, n_samples)  # (batch_size, num_time_steps, n_samples)

        # 将采样的组件索引转换为 one-hot 格式
        one_hot_samples = F.one_hot(sampled_indices,
                                    num_classes=num_treatments).float()  # (batch_size, num_time_steps, n_samples, num_treatments)
        one_hot_samples = one_hot_samples.sum(dim=-2)  # 汇总多次采样的 one-hot 结果 (batch_size, num_time_steps, num_treatments)

        # 如果需要规范化，确保最终的 one-hot 矩阵是有效的概率分布
        one_hot_samples = torch.clamp(one_hot_samples, max=1.0)  # 防止多次采样叠加导致值超出 1

        return one_hot_samples'''

    def sample_from_pi(self, logits, temperature=1.0, hard=True):
        """
        使用 Gumbel-Softmax 从 logits 中采样。

        :param logits: 模型输出的 logits，形状为 (batch_size, num_time_steps, num_treatments)。
        :param temperature: 控制分布平滑度的温度参数，默认为 1.0。
        :param hard: 是否返回 one-hot 向量，默认为 True。
        :return: 采样得到的 one-hot 张量，形状为 (batch_size, num_time_steps, num_treatments)。
        """
        # 使用 Gumbel-Softmax 进行可微分采样
        print('seq 6.2')
        y = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        #print(f'sampled y:{y}')
        return y

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0:
            #self.log_updated_params(optimizer_idx=0)
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    pi_logits = self.get_predictions_first_stage(batch)
                    discrete_loss = self.compute_discrete_loss(batch['current_treatments'], pi_logits, batch['active_entries'])
            else:
                pi_logits = self.get_predictions_first_stage(batch)
                discrete_loss = self.compute_discrete_loss(batch['current_treatments'], pi_logits, batch['active_entries'])

            self.log(f'{self.model_type}_train_discrete_loss', discrete_loss, on_epoch=True, on_step=False,
                     sync_dist=True)


            #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            batch['treatment_discrete_params'] = pi_logits

            return discrete_loss

        elif optimizer_idx == 1:
            #self.log_updated_params(optimizer_idx=1)
            pi_logits = batch['treatment_discrete_params'].detach()
            #chemo_samples = self.sample_from_pi(chemo_pi_logits, n_samples=5)
            #radio_samples = self.sample_from_pi(radio_pi_logits, n_samples=5)
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    samples = self.sample_from_pi(pi_logits, temperature=1.0, hard=True)
                    outcome_pred, _ = self.get_predictions_second_stage(batch, samples)
            else:
                samples = self.sample_from_pi(pi_logits, temperature=1.0, hard=True)
                outcome_pred, _ = self.get_predictions_second_stage(batch, samples)

            mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduction='none')
            mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            return mse_loss

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalized output predictions
        """
        # 获取真实治疗数据并预测
        real_treatment = batch['current_treatments']
        if self.hparams.exp.weights_ema:
            with self.ema_treatment.average_parameters():
                outcome_pred, br = self.get_predictions_second_stage(batch, treatment=real_treatment)
        else:
            outcome_pred, br = self.get_predictions_second_stage(batch, treatment=real_treatment)
        logger.debug(f'my predict_step return: outcome_pred shape :{outcome_pred.shape}')
        return outcome_pred.cpu(), br

    def test_step(self, batch, batch_idx):
        if self.hparams.exp.weights_ema:
            with self.ema_treatment.average_parameters():
                pi_logits = self.get_predictions_first_stage(batch)
                discrete_loss = self.compute_discrete_loss(batch['current_treatments'], pi_logits, batch['active_entries'])

                #chemo_samples = self.sample_from_pi(chemo_pi_logits, n_samples=5)
                #radio_samples = self.sample_from_pi(radio_pi_logits, n_samples=5)
                samples = self.sample_from_pi(pi_logits, temperature=1.0, hard=True)
                outcome_pred, _ = self.get_predictions_second_stage(batch, samples)
        else:
            pi_logits = self.get_predictions_first_stage(batch)
            discrete_loss = self.compute_discrete_loss(batch['current_treatments'], pi_logits, batch['active_entries'])

            # chemo_samples = self.sample_from_pi(chemo_pi_logits, n_samples=5)
            # radio_samples = self.sample_from_pi(radio_pi_logits, n_samples=5)
            samples = self.sample_from_pi(pi_logits, temperature=1.0, hard=True)
            outcome_pred, _ = self.get_predictions_second_stage(batch, samples)

        mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduction='none')
        mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
        total_loss = discrete_loss + mse_loss

        subset_name = self.test_dataloader().dataset.subset_name
        self.log(f'{self.model_type}_{subset_name}_total_loss', total_loss, on_epoch=True, on_step=False,
                 sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_ce_loss', discrete_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
