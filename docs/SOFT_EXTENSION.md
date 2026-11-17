# 后续软体开发边界

当前只有刚体基线；没有占位软体环境冒充可运行实现。

建议先在独立 Newton 环境构建一个最小任务：同形状、不同刚度的三维弹性物体桌面推移。刚体对应高刚度对照；第一阶段暂不同时覆盖布、绳、塑性材料或拓扑变化。这样能先检验“形变改变接触后，现有状态表示和未来监督哪里失效”。这是一项待验证的研究选择，不是现成实验结论。

## 可复用部分

- 视觉/历史编码与动作解码：`dywa/exp/train/distill.py`、`dywa/src/models/modules.py`。
- SEER 辅助未来头与损失：`student_FT_distill.py`、`modules_teacher_FT.py`。这些组件仅依赖 Torch，CPU 测试可先运行。
- 时间窗口、终止掩码与重编码逻辑：`distill_FT_distill_kN.py`、`train_rma_FT_kN.py`。
- DAgger 数据流与学生部署：`train_rma.py`、`policy_runtime.py`。

## 必须重新设计的部分

1. 仿真适配：Isaac Gym actor / rigid-body state、接触、重置与相机接口不能直接当作 Newton 软体接口。先明确批量 `reset / step / done` 和 policy 观测契约，再接入后端。
2. 状态：目前 `object_state_dim=13` 描述刚体 pose/velocity，不能代表软体形变。软体需要形状/形变特征；先决定可观测量与教师特权量各自包含什么。
3. 结构化未来目标：`rel_goal_delta / object_delta / contact / progress` 中的刚体状态差和二值接触需要替换或扩展。不要只扩大张量维数就宣称支持软体。
4. 奖励与成功：位置达标与形变约束应分开记录，避免靠过度压扁物体换取位置奖励。
5. 材料泛化：训练/测试按材料参数与物体形状划分，并记录质量、摩擦、刚度、阻尼，区分记住参数与依据交互适应。

Newton 后端、材料课程、系统辨识/适应机制均作为后续独立改动；旧刚体基线先保留用于回归比较。
