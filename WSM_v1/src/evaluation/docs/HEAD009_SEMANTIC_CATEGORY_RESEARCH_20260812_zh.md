# Head 009 语义类别研究备忘录（2026-08-12）

## 问题定义

Head 009 输入当前 RGB 与当前帧 AST 音频，直接生成通道 195 的 AVIS 语义 mask。当前主要失败模式不是前景几何完全错误，而是前景区域整体被赋成错误类别颜色。因此必须分开报告：

- 前景形状：foreground IoU；
- 全局类别：dominant category accuracy；
- 像素语义：foreground category pixel accuracy。

旧的 `prototype_ce` 不能等同于固定 26 类监督。它在每张目标 latent 内独立聚类，聚类编号不具备跨样本的固定 AVIS 语义，只能约束局部颜色/结构分离。

## 已完成基线

同一组 24 个确定性验证样本、10 步真实推理：

| 方法 | Dominant acc. | Macro acc. | Foreground pixel acc. | Foreground IoU |
|---|---:|---:|---:|---:|
| 原 Head 009（1000 steps） | 12.50% | 9.52% | 5.91% | 59.42% |
| Head 038 初始化 + gated audio + fixed-26 CE（500 steps） | 41.67% | 25.00% | 44.03% | 50.21% |
| 上一行几何 + 5 帧 Head 006 类别投影 | **83.33%** | **83.33%** | **83.54%** | **50.21%** |

最后两行使用完全相同的 24 个样本和 Head 009 checkpoint。类别投影把正确主类别从
10/24 提高到 20/24，同时 foreground IoU 从 0.5020936859637452 保持到
0.5020936859637452，证明它只纠正类别、不改坏已生成的前景几何。完整原始结果位于：

`src/evaluation_output/head009_semantic_projection_20260812/exp07_temporal5_fixed/semantic-evaluation/heldout24_step10/results.json`

## 关键诊断

1. 目标 VAE 表示不是主要上限：缓存目标经 VAE 重建后在先前固定样本上保留主类别约 95.65%。
2. 存在 IoU 很高但整块类别错误的样本，例如 GT 20→预测 21、GT 14→预测 16。对这类错误做无语义聚类不会选回正确类别。
3. 类别频率最大/最小约 42 倍，普通 shuffle 明显偏向常见类。
4. 原 dominant CE 在 batch size=1 时使用 `cross_entropy(weight=..., reduction="mean")`，样本权重被归一化抵消。本轮改为 unreduced CE 后显式乘样本权重。
5. Head 006 的冻结音频 prototype 分类器在完整 762 个验证样本上为 550/762=72.18%；在 Head 009 固定 24 样本上，当前帧音频为 13/24=54.17%，相邻 5 帧平均为 20/24=83.33%。类别信号存在，主要问题是它没有直接决定最终输出类别。

## 已实现并正在验证的方法

### 生成流内部改进

- 修正后的固定 26 类 CE；
- 类别/视频平衡采样（inverse-sqrt 与 full inverse）；
- 从 Gaussian 到 mask 的普通 flow；
- 从当前 RGB latent 到 mask latent 的配对 deterministic flow；
- 单步 x0 预测。

以上六组均从 Head 038 初始化，训练 500 steps，每 50 steps 验证、保存和可视化。

### 形状与类别解耦（当前最有希望）

冻结 Head 006 作为显式 AVIS 类别判别器；Head 009 只负责生成前景几何。推理时保留生成结果的前景区域，并将整块前景投影到判别器选出的固定 AVIS 颜色。评估同时保存 raw 与 projected 输出，避免隐藏生成流本身的错误。

这一方法不同于无监督聚类，也不同于把类别 posterior 作为弱 condition token：类别 ID 直接决定最终语义，不再要求连续 flow 同时完成形状生成和离散类别选择。

### RGB 与音频类别融合

额外训练了两个冻结 ImageNet ResNet-18 骨干的轻量融合分类器：

- 当前帧音频 + 整图 RGB：完整验证集由初始化时约 55.50% 提高到 67.48%；
- 5 帧音频 + 整图 RGB：最佳 73.14%，只比 5 帧音频先验小幅提高。

整图 RGB 容易看见多个物体，却不知道声音具体对应哪一个，因此它不能稳定替代音频先验。
“目标裁剪 RGB + 5 帧音频”实验已经完成：完整验证集最佳为 **74.83%**（step 900），
高于纯 5 帧音频约 2.65 个百分点；固定 24 个困难样本仍为 20/24=83.33%，没有进一步提升。
这里的 74.83% 使用 GT 框裁剪，属于部署上限；实际部署应改用 Head 009 生成 mask 的包围框，
并重新评估裁剪误差。因此现阶段默认投影仍使用更简单、已完成真实端到端验证的纯 5 帧音频先验。

## 文献对应关系

- [SemFlow（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fad7c708dda11f3e72cc1629bb130379-Abstract-Conference.html)：指出 Gaussian 随机性与确定性分割标签之间的矛盾，图像到 mask 采用直接 rectified flow。
- [D3PM](https://arxiv.org/abs/2107.03006) / [CCDM](https://arxiv.org/abs/2303.08888)：用离散类别状态与最终 argmax，避免把类别仅表示成连续 RGB 回归。
- [DiffusionInst](https://arxiv.org/abs/2212.02773)：将生成式实例/形状表示与显式类别预测解耦。
- [Pix2Seq-D](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_A_Generalist_Framework_for_Panoptic_Segmentation_of_Images_and_Videos_ICCV_2023_paper.pdf)：将类别标签与实例 ID 分成两个离散通道生成，而不是混为 RGB 连续距离。
- [Denoising Diffusion Semantic Segmentation with Mask Prior Modeling](https://arxiv.org/abs/2306.01721)：把判别式分割器负责的像素类别与生成式 mask prior 结合；其消融也指出简单加入扩散过程可能降低分割性能。
- UniGS：location-aware palette 与 progressive dichotomy 主要解决实体身份/实例解码；不能代替 AVIS 固定语义类别监督。
- Conditional Discrete Flow Matching + discriminative model（ICAART 2026）：报告离散 flow 可得到更锐利边界，但类别标签准确率仍降低；判别分支与当前解耦方案直接相关。

## 后续判断标准

1. 24 样本的 dominant category accuracy 已超过 70%；下一步在完整验证集确认；
2. 在相同 24 样本上比较 raw/projected foreground pixel accuracy；
3. 前景 IoU 不应因类别投影下降；
4. 扩展到完整验证集时必须报告 macro accuracy、每类召回与 confusion matrix；
5. 若 5 帧音频显著优于当前帧，应将其保留为类别分支的可选增强，但生成分支仍使用当前帧条件，避免改变 Head 009 的原始任务定义。
