"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

# 导入必要的库和模块
from __future__ import absolute_import  # 确保导入行为与Python 3一致
from __future__ import division         # 确保除法行为与Python 3一致
from __future__ import print_function   # 确保print为函数形式

from data.data_utils import computeFFT  # 导入FFT计算工具函数
from model.cell import SGLCell          # 导入SGLC（图卷积循环单元）基础组件
from torch.autograd import Variable     # 用于自动求导的变量包装器
import utils                            # 自定义工具函数
import numpy as np                      # 数值计算库
import pickle                           # 对象序列化工具
import torch                            # PyTorch核心库
import torch.nn as nn                   # 神经网络层模块
import torch.nn.functional as F         # 神经网络函数
import random                           # 随机数生成工具


def apply_tuple(tup, fn):
    """
    对元组中的每个Tensor应用函数，非Tensor元素保持不变
    Args:
        tup: 输入元组（可能包含Tensor或其他类型）
        fn: 要应用的函数（通常是设备迁移函数如.to(device)）
    Returns:
        处理后的元组
    """
    if isinstance(tup, tuple):
        # 对元组中的每个元素应用函数（仅对Tensor生效）
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x) for x in tup)
    else:
        # 如果输入不是元组，直接应用函数
        return fn(tup)


def concat_tuple(tups, dim=0):
    """
    拼接多个Tensor或多个元组中的Tensor（按指定维度）
    Args:
        tups: Tensor列表或元组列表（元组中包含Tensor）
        dim: 拼接维度
    Returns:
        拼接后的Tensor或元组
    """
    if isinstance(tups[0], tuple):
        # 对元组列表，按位置拼接每个元素
        return tuple(
            (torch.cat(xs, dim) if isinstance(xs[0], torch.Tensor) else xs[0])
            for xs in zip(*tups)
        )
    else:
        # 对Tensor列表，直接拼接
        return torch.cat(tups, dim)


class SGLCEncoder(nn.Module):
    """
    SGLC（图卷积循环网络）编码器，用于处理输入序列并提取特征
    """

    def __init__(self, args, input_dim, hid_dim, num_nodes, num_steps, num_rnn_layers,
                 gcgru_activation=None, device=None):
        # 1. 继承nn.Module，初始化父类（PyTorch中所有神经网络模块的标准操作）
        # 目的：获得nn.Module的所有功能（如参数管理、设备迁移、前向传播接口等）
        super(SGLCEncoder, self).__init__()

        # 2. 保存核心超参数（用于后续层构建和前向传播）
        self.hid_dim = hid_dim  # 每个RNN层的隐藏单元维度（如128、256）
        self.num_rnn_layers = num_rnn_layers  # RNN的总层数（如2层、3层，控制模型深度）
        self._device = device  # 计算设备（CPU/GPU，确保数据和模型在同一设备）

        # 3. 初始化编码器的细胞单元列表（存储多层SGLCell）
        # 作用：多层RNN需要多个循环单元，用列表统一管理
        encoding_cells = list()

        # 4. 构建第一层SGLCell（输入维度与后续层不同，需单独定义）
        encoding_cells.append(
            SGLCell(
                args=args,  # 模型配置参数（如动态图学习相关设置）
                input_dim=input_dim,  # 第一层输入维度（原始数据特征维度，如EEG每个通道的特征数）
                num_units=hid_dim,  # 隐藏单元数量（与hid_dim一致，每层隐藏层维度固定）
                num_nodes=num_nodes,  # 图的节点数量（如EEG的通道数，如32、64通道）
                num_steps=num_steps,  # GGNN的信息传播步数（控制图卷积的感受野）
                nonlinearity=gcgru_activation,  # 激活函数（如tanh、relu，控制非线性变换）
                device=self._device  # 绑定计算设备（确保细胞单元在指定设备上运行）
            )
        )

        # 5. 构建后续多层SGLCell（从第2层到第num_rnn_layers层）
        # 循环次数：num_rnn_layers-1次（已构建第一层，剩余层数需循环创建）
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                SGLCell(
                    args=args,
                    input_dim=hid_dim,  # 后续层输入维度=上一层隐藏层维度（多层RNN的标准设计）
                    # 原因：上一层的输出是隐藏状态（维度hid_dim），作为当前层输入
                    num_units=hid_dim,
                    num_nodes=num_nodes,
                    num_steps=num_steps,
                    nonlinearity=gcgru_activation,
                    device=self._device
                )
            )

        # 6. 将细胞单元列表包装为nn.ModuleList（PyTorch专属的模块容器）
        # 核心作用：
        # - 自动管理列表中所有SGLCell的可训练参数（如权重、偏置）
        # - 支持整体设备迁移（如model.to(device)时，所有细胞单元同步迁移）
        # - 支持参数保存/加载（如torch.save(model.state_dict())）
        self.encoding_cells = nn.ModuleList(encoding_cells)
        """多层 RNN 的必要性：
        单层 RNN 捕捉的时空依赖有限，多层 RNN 可通过 “逐层抽象” 提取更复杂的特征（如第一层捕捉局部时空特征，第二层捕捉全局时空特征）。
        适用于 EEG 等复杂生理信号：EEG 信号包含不同频段（α、β、γ 波）和跨通道关联，多层结构能更好地建模这些复杂模式。
        第一层与后续层的输入维度差异：
        第一层输入：原始数据的特征维度（如 EEG 每个时间步的通道特征，输入维度 = 通道数 × 每个通道的特征数）。
        后续层输入：上一层的隐藏状态维度（hid_dim），因为 RNN 的核心是 “用隐藏状态传递历史信息”，后续层需基于前一层的抽象特征继续建模。
        SGLCell 的核心作用：
        每个 SGLCell 是 “图卷积 + 循环门控” 的结合体：既通过动态图学习捕捉 EEG 通道间的空间关联（如通道同步性），又通过 GRU 门控捕捉时间维度的依赖（如脑电信号的时序变化）。
        编码器的每一层都由 SGLCell 构成，确保每一步都能同时建模时空依赖。
        该类的整体作用
        SGLCEncoder 是 “动态图卷积循环编码器”，核心功能是：
        接收输入序列（如 EEG 的历史时间步数据，形状为 [batch_size, seq_len, num_nodes, input_dim]）。
        通过多层 SGLCell 逐层处理序列，提取时空融合特征（隐藏状态）。
        输出最终的隐藏状态（形状为 [num_rnn_layers, batch_size, num_nodes×hid_dim]），作为解码器的输入（用于序列预测）或分类头的输入（用于分类任务）。
        简单说：把原始的时空序列（如 EEG）转化为高维抽象的特征表示，为后续任务（预测、分类）提供强表征支持。"""

    def forward(self, inputs, initial_hidden_state, supports):
        """
        前向传播：处理输入序列，输出隐藏状态
        Args:
            inputs: 输入序列，形状为(seq_length, batch_size, num_nodes, input_dim)
                    - seq_length：时间步数量（如输入100个连续时间步的EEG数据）
                    - batch_size：批次大小（一次训练/推理的样本数）
                    - num_nodes：图节点数（如EEG通道数，32/64通道）
                    - input_dim：每个节点的输入特征维度（如每个EEG通道的特征数）
            initial_hidden_state: 初始隐藏状态，形状为(num_layers, batch_size, num_nodes*hid_dim)
                    - num_layers：RNN层数（与编码器初始化时的num_rnn_layers一致）
                    - 其余维度：每层的初始隐藏状态（全零或预加载状态）
            supports: 图的邻接矩阵支持（用于图卷积）
                    - 形状通常为(batch_size, num_nodes, num_nodes)，表示节点间的初始连接关系
        Returns:
            output_hidden: 最终隐藏状态，形状为(num_layers, batch_size, num_nodes*hid_dim)
                    - 所有层最后一个时间步的隐藏状态（用于解码器初始化）
            current_inputs: 编码器输出序列，形状为(seq_length, batch_size, num_nodes*hid_dim)
                    - 最后一层所有时间步的隐藏状态序列（可用于辅助任务或中间特征提取）
        """
        # 1. 提取输入序列的核心维度：时间步长度和批次大小
        seq_length = inputs.shape[0]  # 获得输入序列的时间步数量（如100）
        batch_size = inputs.shape[1]  # 获得批次大小（如32）

        # 2. 重塑输入：将4维张量转为3维，适配SGLCell的输入格式
        # 原始形状：(seq_length, batch_size, num_nodes, input_dim)
        # 重塑后：(seq_length, batch_size, num_nodes*input_dim)
        # 目的：SGLCell接收的输入是“每个节点特征拼接后的向量”，需将节点数和特征数合并为1维
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # -1表示自动计算最后一维（num_nodes*input_dim）

        # 3. 初始化当前层输入和最终隐藏状态存储列表
        current_inputs = inputs  # 当前层的输入（初始为重塑后的原始输入，后续逐层更新）
        output_hidden = []  # 存储每一层最后一个时间步的隐藏状态（最终堆叠为输出）

        # 4. 逐层处理多层RNN（从第1层到第num_rnn_layers层）
        # 核心逻辑：多层RNN是“串行堆叠”的，前一层的输出序列是后一层的输入序列
        for i_layer in range(self.num_rnn_layers):
            # 4.1 提取当前层的初始隐藏状态
            # initial_hidden_state是(num_layers, ...)，i_layer索引对应当前层的初始状态
            hidden_state = initial_hidden_state[i_layer]  # 形状：(batch_size, num_nodes*hid_dim)

            # 4.2 初始化当前层所有时间步的隐藏状态存储列表
            output_inner = []  # 存储当前层每个时间步的隐藏状态（后续堆叠为该层的输出序列）

            # 5. 逐个时间步处理当前层（从第1个时间步到第seq_length个）
            # 核心逻辑：循环神经网络的本质是“时间步迭代”，每个时间步用前一个状态更新当前状态
            for t in range(seq_length):
                # 5.1 SGLCell前向传播：计算当前时间步的隐藏状态
                # 调用当前层的SGLCell，输入：
                # - supports：图邻接矩阵（动态图学习时会被更新）
                # - current_inputs[t, ...]：当前时间步的输入（形状：(batch_size, num_nodes*input_dim)）
                # - hidden_state：上一个时间步的隐藏状态（初始为该层的initial_hidden_state）
                # 返回值：
                # - 第一个返回值（_）：SGLCell的输出（与隐藏状态一致，此处无需单独存储）
                # - hidden_state：更新后的当前时间步隐藏状态（形状：(batch_size, num_nodes*hid_dim)）
                # - supports：更新后的邻接矩阵（动态图学习时，SGLCell会自适应调整节点连接）
                _, hidden_state, supports = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state
                )

                # 5.2 记录当前时间步的隐藏状态
                output_inner.append(hidden_state)  # 列表元素形状：(batch_size, num_nodes*hid_dim)

            # 6. 记录当前层的最终隐藏状态（最后一个时间步的隐藏状态）
            output_hidden.append(hidden_state)  # 后续堆叠为(num_layers, ...)格式

            # 7. 生成当前层的输出序列，作为下一层的输入序列
            # 把当前层所有时间步的隐藏状态（列表）堆叠为张量：
            # 堆叠后形状：(seq_length, batch_size, num_nodes*hid_dim)
            # 并移动到指定设备（CPU/GPU），确保数据和模型在同一设备
            current_inputs = torch.stack(output_inner, dim=0).to(self._device)

        # 8. 堆叠所有层的最终隐藏状态，形成输出格式
        # 列表output_hidden形状：[num_layers个元素，每个元素形状为(batch_size, num_nodes*hid_dim)]
        # 堆叠后形状：(num_layers, batch_size, num_nodes*hid_dim)
        # 移动到指定设备，确保输出格式统一
        output_hidden = torch.stack(output_hidden, dim=0).to(self._device)

        # 9. 返回最终结果：所有层的最终隐藏状态 + 最后一层的完整输出序列
        return output_hidden, current_inputs
        """1. 多层 RNN 的 “串行堆叠” 逻辑
以 2 层 RNN 为例，数据流向如下：
第 1 层输入：原始输入序列（seq_length, batch_size, num_nodes*input_dim）
第 1 层输出：第 1 层所有时间步的隐藏状态序列（seq_length, batch_size, num_nodes*hid_dim）
第 2 层输入：第 1 层的输出序列（作为第 2 层的输入序列）
第 2 层输出：第 2 层所有时间步的隐藏状态序列（即最终的 current_inputs）
最终 output_hidden：第 1 层最后一个时间步的隐藏状态 + 第 2 层最后一个时间步的隐藏状态（堆叠后形状：(2, batch_size, num_nodes*hid_dim)）
2. 动态图的更新逻辑
supports（邻接矩阵）在每个时间步都会被 SGLCell 更新：
SGLCell 内部通过GraphLearner根据当前输入特征自适应调整节点间的连接关系
同一层的不同时间步共享更新后的supports，下一层则基于上一层最终的supports继续优化
目的：适配 EEG 等动态信号的特性（通道间的关联会随时间变化，如任务切换时）
3. 隐藏状态的核心作用
每个时间步的隐藏状态：是 “当前时间步输入 + 上一个时间步隐藏状态 + 图结构” 的融合特征，包含了历史时空信息
每层最后一个时间步的隐藏状态（output_hidden）：浓缩了该层对整个输入序列的时空特征总结，用于初始化解码器（序列预测任务）
最后一层的完整序列（current_inputs）：包含了每个时间步的高层时空特征，可用于辅助任务（如注意力机制、中间监督）
4. 与 EEG 数据的适配性
时间步迭代：对应 EEG 的连续时间序列（如每秒 250 采样点，100 个时间步即 0.4 秒的数据）
节点融合：对应 EEG 的多个通道（如 32 通道，num_nodes=32），将通道特征拼接后输入，适配图卷积对节点特征的要求
多层特征抽象：第一层捕捉局部时空特征（如相邻通道的同步性 + 短时间依赖），深层捕捉全局特征（如跨脑区通道关联 + 长时间依赖）
该方法的整体作用
forward 方法是 SGLCEncoder 的核心执行逻辑，功能是：
对输入的 4 维时空序列（如 EEG）进行格式转换，适配多层 SGLCell 的输入要求；
逐层、逐时间步迭代处理序列，通过 SGLCell 融合 “当前输入、历史隐藏状态、图结构”，生成每层的特征序列；
输出所有层的最终隐藏状态（用于解码器初始化）和最后一层的完整特征序列（用于后续任务）；
动态更新图结构（supports），适配输入数据的时空动态变化（如 EEG 通道关联的动态调整）。
简单说：将原始 EEG 时空序列，通过多层动态图卷积循环操作，转化为高层抽象的时空特征表示。"""

    def init_hidden(self, batch_size):
        """
        初始化隐藏状态
        Args:
            batch_size: 批次大小（一次训练/推理的样本数，如32、64）
        Returns:
            初始隐藏状态，形状为(num_layers, batch_size, num_nodes*hid_dim)
            - num_layers：编码器的RNN层数（如2层）
            - batch_size：输入的批次大小（与参数一致）
            - num_nodes*hid_dim：每层隐藏状态的维度（节点数×隐藏单元数，如32通道×128隐藏单元=4096）
        """
        # 1. 初始化空列表，用于存储每层的初始隐藏状态
        # 作用：多层RNN需要为每一层单独初始化隐藏状态，列表按“层索引”顺序存储
        init_states = []

        # 2. 遍历每一层RNN，为每层生成初始隐藏状态
        # 循环次数 = 编码器的RNN层数（self.num_rnn_layers）
        for i in range(self.num_rnn_layers):
            # 2.1 调用当前层SGLCell的init_hidden方法，生成该层的初始隐藏状态
            # 核心逻辑：
            # - self.encoding_cells[i]：第i层的SGLCell（从ModuleList中索引）
            # - 调用SGLCell的init_hidden(batch_size)：返回该层的初始隐藏状态，形状为(batch_size, num_nodes*hid_dim)
            # - 初始值为全零张量（SGLCell的init_hidden方法默认实现），确保训练开始时无先验信息干扰
            layer_init_state = self.encoding_cells[i].init_hidden(batch_size)

            # 2.2 将当前层的初始隐藏状态添加到列表
            init_states.append(layer_init_state)  # 列表元素形状：(batch_size, num_nodes*hid_dim)

        # 3. 堆叠所有层的初始隐藏状态，形成最终输出格式
        # 输入：init_states是列表，长度=num_layers，每个元素形状=(batch_size, num_nodes*hid_dim)
        # 堆叠操作：torch.stack(init_states, dim=0)
        # - dim=0：在第0维（新维度）堆叠，生成形状=(num_layers, batch_size, num_nodes*hid_dim)
        # 目的：适配编码器forward方法对初始隐藏状态的输入要求（第一层维度必须是num_layers）
        return torch.stack(init_states, dim=0)
        """1. 为什么需要单独初始化隐藏状态？
RNN（包括 SGLC 这种图卷积 RNN）的核心是 “用隐藏状态传递历史信息”，每个时间步的隐藏状态依赖上一个时间步的结果。
对于序列的第一个时间步，没有 “上一个时间步的隐藏状态”，因此需要手动初始化一个初始值（通常为全零）。
多层 RNN 的每一层都有独立的隐藏状态，必须为每层单独初始化（不能共享），否则会导致层间信息混淆。
2. 初始隐藏状态的形状设计逻辑
以 “2 层 RNN + 32 通道 EEG + 128 隐藏单元 + 32 批次” 为例：
每层隐藏状态形状：(32, 32×128) = (32, 4096)（batch_size=32，num_nodes×hid_dim=32×128）
堆叠后整体形状：(2, 32, 4096)（num_layers=2，batch_size=32，num_nodes×hid_dim=4096）
适配性：与编码器 forward 方法的initial_hidden_state参数形状完全一致，确保前向传播时可直接使用。
3. 与 SGLCell 的关联
该方法的核心是 “委托初始化”：不直接创建张量，而是调用每层 SGLCell 自身的init_hidden方法。
好处：解耦设计，若后续修改 SGLCell 的隐藏状态初始化逻辑（如改为随机初始化、基于数据统计的初始化），编码器的init_hidden无需修改，只需更新 SGLCell 的实现。
4. 与 EEG 数据的适配性
num_nodes对应 EEG 的通道数（如 32、64），hid_dim是每个通道的隐藏特征维度（如 128），两者乘积是 “单批次单个样本的隐藏状态维度”。
全零初始化的合理性：EEG 信号是时序生理信号，训练初期无历史信息可依赖，全零初始化能避免引入无关先验，让模型从数据中自主学习时空依赖。
该方法的整体作用
init_hidden 方法的核心功能是：为编码器的多层 SGLCell 生成统一格式的初始隐藏状态，具体作用包括：
为 RNN 的第一个时间步提供 “初始历史状态”（全零），确保前向传播可正常启动；
生成符合forward方法输入要求的张量形状（num_layers, batch_size, num_nodes×hid_dim），避免维度不匹配错误；
支持多层 RNN 的独立初始化，确保每层隐藏状态的独立性和正确性。
简单说：为编码器的时空序列处理提供 “启动条件”，是连接模型初始化与前向传播的关键方法。"""


class SGLCDecoder(nn.Module):
    """
    SGLC解码器，用于根据编码器输出预测未来序列
    """

    def __init__(self, args, input_dim, num_nodes, hid_dim, num_steps, output_dim,
                 num_rnn_layers, gcgru_activation=None, device=None, dropout=0.0):
        # 1. 继承nn.Module并初始化父类（PyTorch神经网络模块的标准操作）
        # 目的：获得nn.Module的参数管理、设备迁移、前向传播接口等核心功能
        super(SGLCDecoder, self).__init__()

        # 2. 保存核心超参数（用于后续组件构建和前向传播）
        self.input_dim = input_dim  # 解码器输入维度（与目标序列特征维度一致，如EEG预测任务中输出维度=输入维度）
        self.hid_dim = hid_dim  # 隐藏层维度（与编码器隐藏层维度一致，确保特征维度匹配）
        self.num_nodes = num_nodes  # 图节点数量（与编码器一致，如EEG通道数32/64）
        self.output_dim = output_dim  # 解码器最终输出维度（如每个EEG通道的预测特征数）
        self.num_rnn_layers = num_rnn_layers  # RNN层数（与编码器一致，保证模型深度匹配）
        self._device = device  # 计算设备（CPU/GPU，需与编码器、数据保持一致）
        self.dropout = dropout  # Dropout概率（用于防止过拟合）

        # 3. 定义基础SGLC细胞（用于解码器第2层及以后的所有层）
        # 设计逻辑：解码器除第一层外，其余层的输入维度、隐藏维度等参数完全一致，可复用同一个基础细胞（共享结构，参数独立）
        base_cell = SGLCell(
            args=args,  # 模型配置参数（如动态图学习相关设置，与编码器共用）
            input_dim=hid_dim,  # 基础细胞输入维度=隐藏层维度（后续层输入是前一层隐藏状态）
            num_units=hid_dim,  # 隐藏单元数量=hid_dim（与编码器保持一致）
            num_steps=num_steps,  # GGNN信息传播步数（与编码器一致，保证图卷积感受野匹配）
            num_nodes=num_nodes,  # 节点数量（与编码器一致）
            nonlinearity=gcgru_activation,  # 激活函数（与编码器一致，保证特征变换逻辑统一）
            device=self._device  # 绑定计算设备
        )

        # 4. 初始化解码器的细胞单元列表（存储多层SGLCell）
        decoding_cells = list()

        # 5. 构建第一层SGLCell（输入维度与后续层不同，需单独定义）
        decoding_cells.append(
            SGLCell(
                args=args,
                input_dim=input_dim,  # 第一层输入维度=解码器输入维度（如目标序列的特征维度）
                num_units=hid_dim,  # 隐藏单元数量=hid_dim
                num_steps=num_steps,
                num_nodes=num_nodes,
                nonlinearity=gcgru_activation,
                device=self._device
            )
        )

        # 6. 构建后续多层SGLCell（从第2层到第num_rnn_layers层）
        # 循环次数：num_rnn_layers-1次（已构建第一层，剩余层复用base_cell）
        for _ in range(1, num_rnn_layers):
            decoding_cells.append(base_cell)  # 将基础细胞添加到列表，实现多层堆叠

        # 7. 将细胞单元列表包装为nn.ModuleList（PyTorch专属模块容器）
        # 核心作用：
        # - 自动管理所有SGLCell的可训练参数（权重、偏置）
        # - 支持整体设备迁移（如model.to(device)时，所有细胞同步迁移）
        # - 支持参数保存/加载，与编码器保持一致的参数管理逻辑
        self.decoding_cells = nn.ModuleList(decoding_cells)

        # 8. 定义输出投影层（将隐藏状态映射到最终输出维度）
        # 输入：每个节点的隐藏状态（维度=hid_dim）
        # 输出：每个节点的预测特征（维度=output_dim）
        # 目的：SGLCell的输出是隐藏状态（维度hid_dim），需通过线性变换转为任务所需的输出维度
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)

        # 9. 定义Dropout层（用于投影前的正则化）
        # 作用：在隐藏状态输入投影层前随机失活部分神经元，防止模型过拟合（仅训练时生效）
        self.dropout_layer = nn.Dropout(p=dropout)
        """. 解码器与编码器的适配性设计
维度一致：隐藏层维度（hid_dim）、RNN 层数（num_rnn_layers）、节点数量（num_nodes）均与编码器保持一致，确保编码器的最终隐藏状态能直接作为解码器的初始隐藏状态（维度匹配）。
结构对称：编码器是 “多层 SGLCell 堆叠”，解码器同样采用对称结构，保证时空特征的编码与解码逻辑一致，提升模型拟合能力。
参数独立：编码器与解码器的 SGLCell 参数完全独立，避免训练时相互干扰，确保解码器专注于 “从编码特征重构 / 预测目标序列”。
2. 第一层与后续层的输入维度差异
第一层输入：来自 “目标序列的前一个时间步”（或go_symbol），维度为input_dim（与目标序列特征维度一致）。
后续层输入：来自前一层的隐藏状态，维度为hid_dim（隐藏层维度），因此后续层可复用base_cell（输入维度固定为hid_dim）。
3. 输出投影层的作用
解码器的核心是 “生成序列的隐藏状态”，但隐藏状态维度（hid_dim）通常大于任务所需的输出维度（output_dim）。
例如：EEG 预测任务中，hid_dim=128（隐藏层特征），output_dim=1（预测每个通道的未来采样值），需通过nn.Linear(128, 1)将隐藏状态映射为最终预测值。
4. Dropout 层的位置设计
Dropout 层位于 “隐藏状态” 与 “投影层” 之间，目的是对高维隐藏特征进行正则化，避免模型过度依赖某些隐藏单元，提升泛化能力。
若直接对输入或输出 Dropout，可能会破坏序列的时空关联性（尤其是 EEG 这类连续生理信号），因此该位置是兼顾正则化和特征完整性的最优选择。
该方法的整体作用
__init__ 方法的核心功能是：初始化解码器的多层动态图卷积循环结构，构建与编码器对称的网络架构，为序列生成任务（如 EEG 未来时间步预测）提供模型基础。
具体来说：
构建多层 SGLCell，捕捉生成过程中的时空依赖（与编码器一致的图卷积 + 循环门控机制）；
设计输入维度适配逻辑（第一层与后续层差异化），满足序列生成的输入要求；
添加输出投影层和 Dropout 层，完成 “隐藏状态→预测输出” 的转换并防止过拟合；
确保与编码器的维度、结构、设备一致，为编码器 - 解码器的协同工作奠定基础。"""

    def forward(self, inputs, initial_hidden_state, supports, teacher_forcing_ratio=None):
        """
        前向传播：基于编码器的隐藏状态生成预测序列（如EEG未来时间步）
        Args:
            inputs: 目标序列（训练时用于教师强制），形状为(seq_len, batch_size, num_nodes, output_dim)
                    - seq_len：预测序列的长度（如要预测未来10个时间步的EEG）
                    - batch_size：批次大小
                    - num_nodes：节点数（EEG通道数）
                    - output_dim：每个节点的输出维度（如每个EEG通道的采样值维度）
            initial_hidden_state: 编码器的最终隐藏状态，形状为(num_layers, batch, num_nodes*hid_dim)
                    - 作为解码器的初始隐藏状态，传递编码器提取的时空特征
            supports: 图的邻接矩阵支持（从编码器传递而来，动态图已更新）
            teacher_forcing_ratio: 教师强制概率（训练时使用，如0.5表示50%概率用真实值）
        Returns:
            outputs: 预测序列，形状为(seq_len, batch_size, num_nodes*output_dim)
                    - 每个时间步的预测结果，可用于计算损失（训练）或直接输出（推理）
        """
        # 1. 提取目标序列的核心维度：序列长度、批次大小（忽略其他维度）
        seq_length, batch_size, _, _ = inputs.shape  # 解构inputs形状，获取seq_len和batch_size

        # 2. 重塑输入：将4维目标序列转为3维，适配教师强制时的输入格式
        # 原始形状：(seq_len, batch_size, num_nodes, output_dim)
        # 重塑后：(seq_len, batch_size, num_nodes*output_dim)
        # 目的：与解码器的输入格式（current_input：batch_size, num_nodes*output_dim）保持一致
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # -1自动计算为num_nodes*output_dim

        # 3. 定义解码器的起始输入（go_symbol：全零向量）
        # 形状：(batch_size, num_nodes*output_dim)
        # 作用：序列生成的“起始标志”，第一个时间步没有前一个预测值，用全零向量启动生成
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.output_dim)
        ).to(self._device)  # 移动到指定设备（与模型/数据一致）

        # 4. 初始化输出存储张量：存储每个时间步的预测结果
        # 形状：(seq_length, batch_size, num_nodes*output_dim)
        # 初始值全零，后续每个时间步的预测值会覆盖对应位置
        outputs = torch.zeros(
            seq_length, batch_size, self.num_nodes * self.output_dim
        ).to(self._device)

        # 5. 初始化当前输入：第一个时间步的输入为go_symbol
        current_input = go_symbol  # 形状：(batch_size, num_nodes*output_dim)

        # 6. 逐个时间步生成预测（核心循环：从第0个到第seq_length-1个时间步）
        # 逻辑：每个时间步的输入依赖上一个时间步的输出（或真实值，教师强制时）
        for t in range(seq_length):
            # 6.1 初始化列表，存储当前时间步各层的隐藏状态（用于下一时间步）
            next_input_hidden_state = []  # 元素形状：(batch_size, num_nodes*hid_dim)

            # 6.2 逐层处理多层RNN（与编码器一致，逐层堆叠）
            for i_layer in range(self.num_rnn_layers):
                # 6.2.1 获取当前层的初始隐藏状态（上一时间步的隐藏状态）
                # initial_hidden_state是(num_layers, batch_size, ...)，i_layer索引对应当前层
                hidden_state = initial_hidden_state[i_layer]  # 形状：(batch_size, num_nodes*hid_dim)

                # 6.2.2 SGLCell前向传播：计算当前层的输出和新隐藏状态
                # 输入：supports（图邻接矩阵）、current_input（当前输入）、hidden_state（上一状态）
                # 输出：output（当前层输出，与新隐藏状态一致）、hidden_state（更新后的隐藏状态）、supports（更新后的图）
                output, hidden_state, supports = self.decoding_cells[i_layer](
                    supports, current_input, hidden_state
                )

                # 6.2.3 当前层输出作为下一层的输入（多层RNN串行逻辑）
                current_input = output  # 形状：(batch_size, num_nodes*hid_dim)

                # 6.2.4 记录当前层的新隐藏状态（用于下一时间步的初始隐藏状态）
                next_input_hidden_state.append(hidden_state)

            # 6.3 更新初始隐藏状态：将当前时间步各层的隐藏状态堆叠，作为下一时间步的初始状态
            # 堆叠后形状：(num_layers, batch_size, num_nodes*hid_dim)（与initial_hidden_state格式一致）
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)

            # 6.4 将最后一层的隐藏状态投影到输出维度（生成当前时间步的预测值）
            # 步骤1：重塑隐藏状态为(batch_size, num_nodes, hid_dim)——分离节点和隐藏维度
            output_reshaped = output.reshape(batch_size, self.num_nodes, -1)  # -1=hid_dim
            # 步骤2：Dropout正则化（训练时随机失活神经元，防止过拟合）
            output_dropout = self.dropout_layer(output_reshaped)
            # 步骤3：线性投影：(batch_size, num_nodes, hid_dim) → (batch_size, num_nodes, output_dim)
            projected = self.projection_layer(output_dropout)
            # 步骤4：重塑为解码器输出格式：(batch_size, num_nodes*output_dim)
            projected = projected.reshape(batch_size, self.num_nodes * self.output_dim)

            # 6.5 记录当前时间步的预测结果
            outputs[t] = projected  # 写入outputs的第t个时间步位置

            # 6.6 教师强制机制（仅训练时生效，推理时teacher_forcing_ratio为None）
            if teacher_forcing_ratio is not None:
                # 随机判断是否使用教师强制（用真实值作为下一时间步输入）
                teacher_force = random.random() < teacher_forcing_ratio  # 生成0-1随机数，判断是否小于设定概率
                # 若教师强制：下一时间步输入=目标序列的第t个时间步真实值（inputs[t]）
                # 否则：下一时间步输入=当前时间步的预测值（projected）
                current_input = inputs[t] if teacher_force else projected
            else:
                # 推理时：直接使用当前预测值作为下一时间步输入（自回归生成）
                current_input = projected

        # 7. 返回完整的预测序列（形状：(seq_len, batch_size, num_nodes*output_dim)）
        return outputs
    """ 序列生成的核心逻辑（自回归）
解码器的本质是 “自回归生成”：每个时间步的预测依赖上一个时间步的输出，形成 “输入→预测→下一轮输入” 的循环。例如 EEG 预测任务中：
第 0 个时间步输入：go_symbol → 预测第 1 个未来时间步的 EEG；
第 1 个时间步输入：第 0 个时间步的预测结果 → 预测第 2 个未来时间步的 EEG；
以此类推，直到生成所有seq_length个时间步的预测。
2. 教师强制（Teacher Forcing）的作用
训练稳定性：训练初期模型预测不准确，若用错误的预测值作为下一轮输入，会导致误差累积（训练发散）。教师强制以一定概率用真实目标值作为输入，帮助模型快速收敛。
参数说明：teacher_forcing_ratio=0.7 表示 70% 的时间用真实值，30% 的时间用预测值，平衡收敛速度和泛化能力。
推理时关闭：推理时没有真实目标值，必须用预测值自回归生成，因此teacher_forcing_ratio=None。
3. 多层 RNN 的处理逻辑
与编码器一致，解码器的多层 RNN 是 “串行堆叠”：
第 1 层输入：current_input（上一时间步的预测 / 真实值）；
第 1 层输出：作为第 2 层的输入；
最后一层输出：作为当前时间步的隐藏状态，经投影后得到预测值；
每层的隐藏状态都会被记录，堆叠后作为下一时间步的初始隐藏状态，确保时间维度的信息传递。
4. 与 EEG 预测任务的适配性
时间步对应：seq_length 是要预测的未来 EEG 时间步数量（如预测未来 50 个采样点，对应 0.2 秒数据）；
节点对应：num_nodes 是 EEG 通道数（如 32 通道），num_nodes*output_dim 是每个时间步的总输出维度（如 32 通道 ×1 个采样值 = 32）；
动态图适配：supports 从编码器传递而来，且在每个时间步被 SGLCell 更新，适配 EEG 通道关联的动态变化（如预测过程中通道同步性变化）。
5. 输出投影的关键作用
最后一层 SGLCell 的输出是隐藏状态（维度hid_dim，如 128），而 EEG 预测任务需要输出每个通道的采样值（维度output_dim，如 1）；
通过 nn.Linear(hid_dim, output_dim) 将高维隐藏特征映射为低维预测值，完成 “特征→输出” 的转换；
Dropout 层在投影前添加，避免模型过度依赖某些隐藏单元，提升对噪声 EEG 数据的泛化能力。
该方法的整体作用
forward 方法是 SGLCDecoder 的核心执行逻辑，功能是：
接收编码器的高层时空特征（初始隐藏状态）和动态图结构（supports）；
以全零向量（go_symbol）启动，通过自回归循环生成指定长度的预测序列；
训练时通过教师强制机制稳定训练过程，推理时通过自回归完成独立预测；
将每层的隐藏状态经投影和正则化后，输出最终的预测序列（如 EEG 未来时间步数据）。
简单说：基于编码器提取的时空特征，生成与目标序列长度一致的预测结果，是序列预测任务（如 EEG 信号预测、癫痫发作预警）的核心模块。"""


########## 用于癫痫分类/检测的模型 ##########
class SGLCModel_classification(nn.Module):
    """
    基于SGLC的分类模型，用于癫痫检测或类型分类
    """

    def __init__(self, args, num_classes, device=None):
        # 1. 继承nn.Module并初始化父类（PyTorch神经网络模块的标准操作）
        # 目的：获得nn.Module的参数管理、设备迁移、前向传播接口等核心功能
        super(SGLCModel_classification, self).__init__()

        # 2. 从参数对象args中提取模型配置（避免硬编码，提升灵活性）
        num_nodes = args.num_nodes  # 图节点数量（对应EEG通道数，如32/64通道）
        num_rnn_layers = args.num_rnn_layers  # RNN层数（与编码器一致，如2/3层）
        rnn_units = args.rnn_units  # RNN隐藏单元数（如128/256，控制特征维度）
        enc_input_dim = args.input_dim  # 编码器输入维度（每个EEG通道的特征维度，如1个采样值/多特征融合）
        num_steps = args.n_steps  # 输入序列的时间步数量（如EEG的连续采样点数，50/100个时间步）

        # 3. 保存核心配置为实例属性（用于后续前向传播或组件调用）
        self.num_nodes = num_nodes  # 节点数（EEG通道数）
        self.num_rnn_layers = num_rnn_layers  # RNN层数
        self.rnn_units = rnn_units  # 隐藏单元数
        self._device = device  # 计算设备（CPU/GPU，需与数据、编码器保持一致）
        self.num_classes = num_classes  # 分类任务的类别数（如EEG情绪分类：3类；癫痫检测：2类）
        self.num_steps = num_steps  # 输入时间步数量

        # 4. 初始化编码器（复用之前实现的SGLCEncoder）
        # 核心作用：提取输入序列的高层时空特征（EEG的通道间空间关联+时间动态依赖）
        self.encoder = SGLCEncoder(
            args=args,  # 模型配置参数（与编码器、解码器共用，保证一致性）
            input_dim=enc_input_dim,  # 编码器输入维度（每个节点的特征维度）
            hid_dim=rnn_units,  # 隐藏层维度（与RNN隐藏单元数一致）
            num_nodes=num_nodes,  # 节点数（EEG通道数）
            num_steps=num_steps,  # 时间步数量（输入序列长度）
            num_rnn_layers=num_rnn_layers,  # RNN层数（与整体配置一致）
            gcgru_activation=args.gcgru_activation  # 激活函数（如tanh/relu，控制非线性变换）
        )

        # 5. 定义分类头的全连接层（核心分类组件）
        # 输入：编码器输出的高层特征（维度=rnn_units，每个节点的隐藏状态）
        # 输出：类别预测得分（维度=num_classes，如2类输出2个得分）
        # 逻辑：将编码器提取的时空特征映射到类别空间，完成分类任务
        self.fc = nn.Linear(rnn_units, num_classes)

        # 6. 定义Dropout层（正则化，防止过拟合）
        # 作用：在特征输入全连接层前随机失活部分神经元，避免模型过度依赖某些特征
        # 适配场景：EEG数据存在个体差异和噪声，Dropout能提升模型泛化能力
        self.dropout = nn.Dropout(args.dropout)  # dropout概率从参数中读取

        # 7. 定义ReLU激活函数（引入非线性，增强分类能力）
        # 作用：在Dropout后、全连接层前添加非线性变换，让模型能拟合更复杂的特征-类别映射
        self.relu = nn.ReLU()
        """ 分类任务与序列预测任务的核心差异
之前的 SGLCDecoder 用于序列预测（生成未来时间步的输出，如预测 EEG 未来采样值），输出是 “序列”；
本类 SGLCModel_classification 用于分类任务（输入序列映射到固定类别，如 EEG 情绪分类、癫痫检测），输出是 “类别得分”；
因此，模型结构去掉了解码器，直接用编码器提取特征，再通过 “分类头” 完成类别映射。
2. 编码器在分类任务中的作用
编码器的核心能力是 “时空特征提取”：
空间维度：通过动态图学习捕捉 EEG 通道间的同步性 / 关联性（如情绪任务中前额叶通道的协同激活）；
时间维度：通过 SGLCell 的门控机制捕捉 EEG 信号的时序动态（如癫痫发作前的脑电节律变化）；
编码器输出的 “最后一层所有时间步的隐藏状态” 或 “最后一个时间步的隐藏状态”，是浓缩了时空信息的高层特征，适合作为分类输入。
3. 分类头的设计逻辑
分类头的结构为：Dropout → ReLU → 全连接层，是分类任务的经典轻量结构：
顺序逻辑：先通过 Dropout 正则化 → 再通过 ReLU 引入非线性 → 最后通过全连接层映射到类别；
维度适配：编码器输出的每个节点隐藏状态维度为 rnn_units（如 128），全连接层直接将该维度映射到 num_classes（如 2），无需额外维度转换（与 EEG 每个通道的特征独立映射后融合或取平均兼容）。
4. 与 EEG 分类场景的适配性
节点数适配：num_nodes 对应 EEG 通道数（如 32 通道），编码器能自动建模通道间的空间关联（无需手动设计通道连接规则）；
动态图适配：SGLCEncoder 的动态图学习能力，能自适应不同被试、不同任务下的 EEG 通道关联变化（如不同人情绪唤醒时的通道激活模式差异）；
噪声鲁棒性：Dropout 层和 ReLU 激活函数的组合，能缓解 EEG 数据的噪声和个体差异带来的过拟合问题，提升模型在新被试上的泛化能力。
5. 参数复用与一致性设计
所有核心参数（如 num_nodes、rnn_units、num_rnn_layers）从 args 中读取，确保编码器、分类头的配置一致，避免维度不匹配；
激活函数、dropout 概率等超参数统一配置，便于后续调参和实验对比。
该方法的整体作用
__init__ 方法的核心功能是：构建适用于时空序列分类任务（如 EEG 分类）的端到端模型，具体作用包括：
解析模型配置参数，统一编码器和分类头的核心超参数；
初始化 SGLCEncoder，用于提取输入序列（如 EEG）的高层时空特征；
构建轻量高效的分类头（Dropout+ReLU + 全连接层），将时空特征映射到类别空间；
引入正则化和非线性变换，提升模型的分类精度和泛化能力。
简单说：将 “动态图卷积循环编码器” 与 “分类头” 结合，形成专门处理 EEG 等时空序列分类任务的模型，实现从原始信号到类别的端到端学习。"""

    def forward(self, input_seq, seq_lengths, supports):
        """
        前向传播：处理EEG等时空序列，输出类别预测得分（logits）
        Args:
            input_seq: 输入序列，形状为(batch, seq_len, num_nodes, input_dim)
                    - batch：批次大小（如32）
                    - seq_len：序列长度（输入的时间步数量，如100个EEG采样点）
                    - num_nodes：节点数（EEG通道数，如32）
                    - input_dim：每个节点的输入维度（如每个EEG通道的特征数，1个采样值）
            seq_lengths: 每个样本的实际序列长度（不含填充），形状为(batch,)
                    - 作用：处理变长序列（如部分EEG样本有效时间步不足seq_len，需忽略填充）
            supports: 图的邻接矩阵支持，形状为(batch, num_nodes, num_nodes)
                    - 初始通道连接关系（动态图学习会更新）
        Returns:
            pool_logits: 类别预测得分，形状为(batch_size, num_classes)
                    - 每个样本对应num_classes个得分，后续可通过softmax转概率
        """
        # 1. 提取输入序列的核心维度：批次大小、最大序列长度（填充后的长度）
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # 2. 转置输入序列维度，适配编码器的输入格式
        # 原始形状：(batch, seq_len, num_nodes, input_dim)（batch_first=True格式）
        # 转置后：(seq_len, batch, num_nodes, input_dim)（seq_first格式）
        # 原因：编码器SGLCEncoder的forward方法要求输入第一维为时间步（seq_len）
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # 3. 初始化编码器的隐藏状态
        # 调用编码器的init_hidden方法，生成全零初始状态，形状为(num_layers, batch, num_nodes*rnn_units)
        # 移动到指定设备（与输入、模型一致，避免设备不匹配错误）
        init_hidden_state = self.encoder.init_hidden(batch_size).to(self._device)

        # 4. 编码器前向传播，提取高层时空特征
        # 输入：转置后的输入序列、初始隐藏状态、图邻接矩阵
        # 输出：
        # - 第一个返回值（_）：所有层的最终隐藏状态（num_layers, batch, num_nodes*rnn_units），此处无需使用
        # - final_hidden：最后一层的完整输出序列（seq_len, batch, num_nodes*rnn_units），包含每个时间步的高层特征
        _, final_hidden = self.encoder(input_seq, init_hidden_state, supports)

        # 5. 转置输出序列维度，恢复为batch_first格式
        # 转置前：(seq_len, batch, num_nodes*rnn_units)
        # 转置后：(batch, seq_len, num_nodes*rnn_units)
        # 目的：后续提取“每个样本的有效最后时间步”时，方便按batch索引操作
        output = torch.transpose(final_hidden, dim0=0, dim1=1)

        # 6. 提取每个样本的有效最后一个时间步特征（关键步骤：忽略填充）
        # 背景：变长序列训练时，短样本会被填充到max_seq_len，填充部分无意义需剔除
        # 工具函数utils.last_relevant_pytorch：根据seq_lengths（实际长度），提取每个样本的最后有效时间步特征
        # 输入：output（batch, seq_len, num_nodes*rnn_units）、seq_lengths（batch,）、batch_first=True
        # 输出：last_out（batch_size, num_nodes*rnn_units）—— 每个样本的最终时空特征
        last_out = utils.last_relevant_pytorch(
            output, seq_lengths, batch_first=True
        )

        # 7. 重塑特征维度，分离节点和隐藏状态维度
        # 重塑前：(batch_size, num_nodes*rnn_units)（节点和隐藏特征拼接）
        # 重塑后：(batch_size, num_nodes, rnn_units)（每个节点对应独立的隐藏特征）
        # 目的：后续分类头需对每个节点的特征单独处理，再聚合所有节点信息
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        # 确保特征在指定设备上（防止因工具函数导致的设备偏移）
        last_out = last_out.to(self._device)

        # 8. 分类头前向传播：将节点特征映射到类别得分
        # 步骤拆解：
        # 1) dropout：对节点特征做正则化（随机失活部分节点的隐藏特征，防止过拟合）
        # 2) relu：引入非线性变换，增强模型对复杂特征的拟合能力
        # 3) fc：全连接层，将每个节点的隐藏特征（rnn_units维）映射到类别得分（num_classes维）
        # 输出形状：(batch_size, num_nodes, num_classes)—— 每个节点对应num_classes个得分
        logits = self.fc(self.relu(self.dropout(last_out)))

        # 9. 节点维度最大池化：聚合所有节点的类别得分（关键步骤）
        # 操作：对维度1（num_nodes）做最大池化，取每个类别的最大得分
        # 输入：logits（batch_size, num_nodes, num_classes）
        # 输出：pool_logits（batch_size, num_classes）—— 每个样本的最终类别得分
        # 逻辑：EEG分类中，不同通道对类别的贡献不同，最大池化能突出最具判别力的通道特征
        pool_logits, _ = torch.max(logits, dim=1)

        # 10. 返回最终类别得分（logits），后续可通过softmax转换为概率
        return pool_logits
    """变长序列处理（seq_lengths 与 last_relevant_pytorch）
背景：EEG 数据可能存在变长情况（如不同被试的任务时长不同），训练时需将序列填充到相同长度（max_seq_len），但填充部分无生理意义。
核心作用：last_relevant_pytorch 函数根据 seq_lengths 提取每个样本的 “真实最后一个时间步特征”，避免填充部分干扰分类结果。
示例：若某样本实际长度为 80（seq_lengths[i]=80），填充后长度为 100，则提取第 80 个时间步的特征，忽略 81-100 步的填充值。
2. 编码器输出的选择逻辑
编码器返回两个值：output_hidden（所有层的最终隐藏状态）和 final_hidden（最后一层的完整序列）。
分类任务选择 final_hidden（最后一层所有时间步特征），再提取最后有效时间步，原因是：最后一层的特征是最抽象的时空特征，且最后一个时间步浓缩了整个序列的历史信息，最适合用于分类。
3. 分类头的特征处理流程
维度变化链：
原始特征 → (batch, num_nodes*rnn_units) → 重塑 → (batch, num_nodes, rnn_units) → dropout → ReLU → 全连接 → (batch, num_nodes, num_classes) → 池化 → (batch, num_classes)
核心逻辑：先对每个节点的特征单独做非线性变换和类别映射，再聚合所有节点的结果，既保留了单个通道的判别信息，又融合了通道间的全局信息。
4. 节点聚合（最大池化）的合理性
EEG 场景适配：不同 EEG 通道对分类任务的贡献不同（如情绪分类中前额叶通道更重要，癫痫检测中颞叶通道更重要）。
最大池化优势：相比平均池化，最大池化能突出 “最具判别力的通道”（如某通道的类别得分最高，说明该通道对当前样本的分类贡献最大），更适合 EEG 这种通道功能差异化的场景。
可选方案：也可使用平均池化、注意力池化（给重要通道分配更高权重），但最大池化实现简单且鲁棒性强。
5. 与 EEG 分类任务的深度适配
时空特征融合：编码器通过动态图学习捕捉 EEG 通道间的空间关联（如同步激活），通过循环门控捕捉时间动态（如脑电节律变化），最终提取的 last_out 是 “时空融合特征”，比单独的时间特征或空间特征更具判别力。
噪声鲁棒性：dropout 层和 ReLU 激活函数的组合，能缓解 EEG 数据的噪声和个体差异带来的过拟合问题；最大池化则能降低个别噪声通道的干扰。
该方法的整体作用
forward 方法是分类模型的核心执行逻辑，功能是：
对输入的 EEG 时空序列做格式转换，适配编码器输入；
通过编码器提取高层时空特征（空间：通道关联；时间：时序动态）；
处理变长序列，提取每个样本的有效最终特征；
通过分类头将节点特征映射到类别得分，并聚合所有通道信息；
输出最终的类别预测得分，为后续损失计算（如交叉熵）或推理提供输入。
简单说：将原始 EEG 时空序列，通过 “特征提取→有效特征筛选→类别映射→通道聚合” 的全流程，输出类别预测结果，完成端到端分类。"""


########## 用于未来时间步预测的模型 ##########
class SGLCModel_nextTimePred(nn.Module):
    """
    基于SGLC的序列预测模型，用于预测未来时间步的信号（如EEG信号预测）
    """

    class SGLCModel_nextTimePred(nn.Module):
        def __init__(self, args, device=None):
            # 1. 继承nn.Module并初始化父类（PyTorch神经网络模块的标准操作）
            # 目的：获得参数管理、设备迁移、前向传播接口等核心功能
            super(SGLCModel_nextTimePred, self).__init__()

            # 2. 从参数对象args中提取核心配置（统一编码器、解码器的超参数，避免硬编码）
            num_nodes = args.num_nodes  # 节点数量（如EEG通道数32/64）
            num_rnn_layers = args.num_rnn_layers  # RNN层数（编码器、解码器保持一致，如2/3层）
            rnn_units = args.rnn_units  # 隐藏单元数（编码器、解码器维度一致，确保特征匹配）
            enc_input_dim = args.input_dim  # 编码器输入维度（每个EEG通道的输入特征数，如1个采样值）
            dec_input_dim = args.output_dim  # 解码器输入维度（与输出维度一致，因预测任务输入输出格式相同）
            output_dim = args.output_dim  # 最终输出维度（每个EEG通道的预测特征数，如1个采样值）
            num_steps = args.n_steps  # 输入序列的时间步数量（如编码器输入100个EEG采样点）

            # 3. 保存核心配置为实例属性（用于后续前向传播或组件调用）
            self.num_nodes = num_nodes  # 节点数（EEG通道数）
            self.num_steps = num_steps  # 编码器输入时间步数量
            self.num_rnn_layers = num_rnn_layers  # RNN层数
            self.rnn_units = rnn_units  # 隐藏单元数
            self._device = device  # 计算设备（CPU/GPU，需与数据、组件一致）
            self.output_dim = output_dim  # 输出维度
            self.cl_decay_steps = args.cl_decay_steps  # 课程学习衰减步数（控制教师强制概率的下降速度）
            self.use_curriculum_learning = bool(args.use_curriculum_learning)  # 是否启用课程学习

            # 4. 初始化编码器（复用SGLCEncoder，提取输入序列的高层时空特征）
            self.encoder = SGLCEncoder(
                args=args,  # 共享模型配置（如动态图学习参数）
                input_dim=enc_input_dim,  # 编码器输入维度
                hid_dim=rnn_units,  # 隐藏层维度（与解码器一致）
                num_nodes=num_nodes,  # 节点数（与解码器一致）
                num_steps=num_steps,  # 输入时间步数量
                num_rnn_layers=num_rnn_layers,  # RNN层数（与解码器一致）
                gcgru_activation=args.gcgru_activation  # 激活函数（统一非线性变换逻辑）
            )

            # 5. 初始化解码器（复用SGLCDecoder，基于编码器特征生成未来序列）
            self.decoder = SGLCDecoder(
                args=args,  # 共享模型配置
                input_dim=dec_input_dim,  # 解码器输入维度（与输出维度一致）
                num_nodes=num_nodes,  # 节点数（与编码器一致）
                hid_dim=rnn_units,  # 隐藏层维度（与编码器一致）
                num_steps=num_steps,  # GGNN传播步数（与编码器一致，保证图卷积感受野匹配）
                output_dim=output_dim,  # 输出维度（预测目标的维度）
                num_rnn_layers=num_rnn_layers,  # RNN层数（与编码器一致）
                gcgru_activation=args.gcgru_activation,  # 激活函数（与编码器一致）
                device=device,  # 计算设备
                dropout=args.dropout  # Dropout概率（正则化，防止过拟合）
            )

        def forward(self, encoder_inputs, decoder_inputs, supports, batches_seen=None):
            """
            前向传播：基于历史EEG序列预测未来时间步序列（如预测未来50个采样点）
            Args:
                encoder_inputs: 编码器输入（历史序列），形状为(batch, input_seq_len, num_nodes, input_dim)
                        - input_seq_len：编码器输入时间步数量（如100个历史EEG采样点）
                decoder_inputs: 解码器输入（目标序列，用于教师强制），形状为(batch, output_seq_len, num_nodes, output_dim)
                        - output_seq_len：要预测的未来时间步数量（如50个未来EEG采样点）
                supports: 图的邻接矩阵支持，形状为(batch, num_nodes, num_nodes)
                batches_seen: 已训练的批次数量（用于课程学习中计算教师强制概率）
            Returns:
                outputs: 预测序列，形状为(batch, output_seq_len, num_nodes, output_dim)
                        - 与解码器输入格式一致，可直接与真实目标序列计算损失
            """
            # 1. 提取解码器输入的核心维度：批次大小、预测序列长度、节点数
            batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

            # 2. 转置编码器输入维度，适配编码器的输入格式
            # 原始形状：(batch, input_seq_len, num_nodes, input_dim)（batch_first=True）
            # 转置后：(input_seq_len, batch, num_nodes, input_dim)（seq_first=True）
            # 原因：编码器SGLCEncoder要求输入第一维为时间步
            encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)

            # 3. 转置解码器输入维度，适配解码器的输入格式
            # 原始形状：(batch, output_seq_len, num_nodes, output_dim)（batch_first=True）
            # 转置后：(output_seq_len, batch, num_nodes, output_dim)（seq_first=True）
            # 原因：解码器SGLCDecoder要求输入第一维为时间步（用于教师强制时读取真实序列）
            decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

            # 4. 初始化编码器的隐藏状态
            # 调用编码器init_hidden方法生成全零初始状态，形状为(num_layers, batch, num_nodes*rnn_units)
            # 移动到指定设备，避免设备不匹配错误
            init_hidden_state = self.encoder.init_hidden(batch_size).to(self._device)

            # 5. 编码器前向传播：提取历史序列的高层时空特征
            # 输入：转置后的编码器输入、初始隐藏状态、图邻接矩阵
            # 输出：
            # - encoder_hidden_state：所有层的最终隐藏状态（num_layers, batch, num_nodes*rnn_units）—— 传递给解码器作为初始状态
            # - 第二个返回值（_）：最后一层的完整输出序列（此处无需使用，因预测任务只需最终隐藏状态）
            encoder_hidden_state, _ = self.encoder(encoder_inputs, init_hidden_state, supports)

            # 6. 课程学习：计算当前批次的教师强制概率（训练时动态调整）
            if self.training and self.use_curriculum_learning and (batches_seen is not None):
                # 训练模式 + 启用课程学习 + 已知训练批次 → 计算动态教师强制概率
                # utils.compute_sampling_threshold：随训练批次增加，教师强制概率逐渐降低（从1→0）
                # 逻辑：训练初期用高教师强制（帮助模型收敛），后期用低教师强制（提升自回归能力）
                teacher_forcing_ratio = utils.compute_sampling_threshold(
                    self.cl_decay_steps, batches_seen
                )
            else:
                # 测试模式 / 未启用课程学习 / 未知训练批次 → 不使用教师强制（纯自回归生成）
                teacher_forcing_ratio = None

            # 7. 解码器前向传播：基于编码器特征生成未来序列
            # 输入：转置后的解码器输入（真实目标序列，用于教师强制）、编码器最终隐藏状态、图邻接矩阵、教师强制概率
            # 输出：outputs（seq_first格式），形状为(output_seq_len, batch, num_nodes*output_dim)
            outputs = self.decoder(
                decoder_inputs,
                encoder_hidden_state,
                supports,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            # 8. 重塑输出维度：分离节点和输出维度
            # 重塑前：(output_seq_len, batch, num_nodes*output_dim)
            # 重塑后：(output_seq_len, batch, num_nodes, output_dim)
            # 目的：恢复为与输入一致的4维格式，方便后续计算损失（与真实目标序列维度匹配）
            outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))  # -1=output_dim

            # 9. 转置输出维度：恢复为batch_first格式
            # 转置前：(output_seq_len, batch, num_nodes, output_dim)（seq_first=True）
            # 转置后：(batch, output_seq_len, num_nodes, output_dim)（batch_first=True）
            # 目的：符合PyTorch训练中“批次在前”的习惯，便于后续损失计算（如MSE损失）
            outputs = torch.transpose(outputs, dim0=0, dim1=1)

            # 10. 返回最终预测序列（与解码器输入格式完全一致）
            return outputs
        """模型核心架构：编码器 - 解码器（Encoder-Decoder）
这是时序预测任务的经典架构，适配 EEG 未来时间步预测场景：
编码器：输入 “历史 EEG 序列”（如 100 个时间步），通过动态图卷积循环结构提取高层时空特征（通道间空间关联 + 时序动态依赖），输出最终隐藏状态（浓缩整个历史序列的信息）。
解码器：接收编码器的隐藏状态（作为初始状态），通过自回归生成 “未来 EEG 序列”（如 50 个时间步），训练时用教师强制稳定收敛，推理时纯自回归生成。
2. 课程学习（Curriculum Learning）的核心作用
背景：时序预测的自回归生成容易出现 “误差累积”（早期预测错误导致后续预测偏差），训练初期模型能力弱，直接自回归难以收敛。
逻辑：随训练批次增加，教师强制概率从 1 逐渐降至 0：
训练初期（batches_seen 小）：teacher_forcing_ratio≈1，用真实目标序列作为解码器输入，帮助模型学习正确的时空映射。
训练后期（batches_seen 大）：teacher_forcing_ratio≈0，用模型预测值作为解码器输入，提升自回归生成能力。
适配 EEG 场景：EEG 信号噪声多、动态变化复杂，课程学习能让模型循序渐进掌握预测规律，避免初期因误差累积导致训练发散。
3. 维度转换的核心逻辑
整个前向传播的维度转换围绕 “编码器 / 解码器的输入要求” 和 “损失计算的格式适配”：
编码器 / 解码器要求输入为 (seq_len, batch, num_nodes, dim)（时间步在前），因此需对输入做转置。
输出需恢复为 (batch, seq_len, num_nodes, dim)（批次在前），与真实目标序列格式一致，便于计算 MSE 等损失。
4. 与 EEG 未来时间步预测的适配性
空间维度适配：num_nodes 对应 EEG 通道数，动态图学习能自适应通道间的同步性变化（如预测过程中通道关联的动态调整）。
时间维度适配：编码器处理历史时序信息，解码器生成未来时序信息，循环结构天然适配 EEG 的连续时间特性。
噪声鲁棒性：解码器的 Dropout 层和课程学习策略，能缓解 EEG 噪声带来的预测误差累积问题，提升预测精度。
5. 超参数一致性设计
编码器与解码器的 num_rnn_layers、rnn_units、num_nodes 等核心超参数完全一致，确保：
编码器输出的隐藏状态维度与解码器的初始状态维度匹配。
两者的图卷积感受野、特征抽象能力一致，避免维度不匹配或特征表达能力失衡。
该类的整体作用
SGLCModel_nextTimePred 是专门用于EEG 等时空序列未来时间步预测的端到端模型，核心功能是：
接收历史 EEG 序列和目标序列（训练时），通过编码器提取时空特征；
解码器基于编码器特征，结合课程学习和教师强制机制，生成未来时间步的 EEG 序列；
输出与目标序列格式一致的预测结果，可直接用于损失计算（训练）或实际预测（推理）。
简单说：基于历史 EEG 数据，预测未来一段时间内的 EEG 信号，适用于癫痫发作预警、脑电信号趋势预测等场景。"""
########## 用于未来时间步预测的模型 ##########
