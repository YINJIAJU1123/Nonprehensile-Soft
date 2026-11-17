# 迁移记录

迁移日期：2026-11-17。目标是新的开发基线，不是封存论文结果的完整复现仓库。

## 来源与选择

- 原项目：YINJIAJU1123/Non-prehensile-Manipulation-with-Latent-World-Model。
- 主要来源：`experiment_work_20260809/contact_conditioned_future_v1` 本地工作树；其基准提交为 `dbc3aa85339b8b151835066c5ac18d0c4c5c632f`。复制的是本地文件，不能仅用 Git SHA 代替文件来源记录。
- SEER 缓存修复：`experiment_work_20261021/seer_history` 的 `seer_history.py`、`distill.py`、`distill_FT_distill.py`。
- DyWA 缓存修复：`experiment_work_20261021/dywa_history/modules.py`。
- `source_manifest.json` 逐文件记录迁移源的 SHA-256 和迁移后初始文件 SHA-256。它是迁移时的静态记录，后续开发以 Git 为准。

未复制历史训练日志、结果、论文、checkpoint、外部对象数据、临时脚本、虚拟环境、旧远端路径或另一套 Git 历史。原始源码未修改。

## 有意保留及修改

- 保留 `dywa/src`、`dywa/exp/train` 的原导入与类名；共用模块按训练入口的导入依赖选择。
- SEER 是 contact-conditioned + K=3 future supervision 版本。DyWA 是共享基础学生，不启用 SEER 的接触注意力和未来头。
- 两者开启持久历史缓存；SEER 使用安全切片复制和独立快照。学生全量 reset 同时清空历史。**旧历史缓存有缺陷时得到的指标不能直接当作修复后的指标。**
- `scripts/run.py`、`configs/*.json` 统一外部资源与启动入口；预设使用三视角相机、Franka CRM 模型、T80 训练划分。U20 只用于明确指定的评估。
- `evaluate.py` / `policy_runtime.py` 从原 fixed-trial 执行流程整理，只支持 DyWA 和标准 SEER，严格加载网络与环境统计。SEER 初始化时读取教师以获取 latent 维度，随后释放，动作循环没有教师。暂不提供 residual / advantage / planner 的评估。
- 删除上一篇论文的专用 `ral_v2_student_protocol.py` 和 formal recipe 分支，改用 `train/source_lineage.py`。保留代码、权重、数据划分 SHA 校验；DGN 只记录 metadata hash 和资源路径，**未保留整套资产锁认证**。
- 去掉 `generic_state_encoder.py` 中依赖缺失 `RandomEnv` 的内嵌演示函数。
- 仅保留完整的 `crm-panda` 等支持配置所需资产。旧 `franka_description` 本身缺 collision OBJ，因此未迁移；代码中的其他机器人配置、离线碰撞辅助函数和备用机器人分支不是这里支持的运行路径，启用前需要补全对应资产。
- 规范化迁移文本的换行和行尾空白，二进制资产保持原样。
- 原 `setup.cfg` 错写 MIT，已与根 LICENSE 对齐为 CC-BY-NC-4.0；保留 CUDA 扩展源码，补齐 `cxx` 包和源码分发清单。

部分 residual、advantage、planner、state-query 和备用 robot 模块因共享类直接导入而保留，默认关闭。之后可以独立重构依赖；本次不通过删除这些成员改变 checkpoint schema。

## 验证范围

已执行 CPU 单元测试：历史帧 FIFO / 独立快照 / 分 lane reset、K=3 未来监督反向传播与 target detach、终止边界窗口屏蔽、严格 checkpoint 加载、仅学生动作的执行循环，以及 6 种启动配置的 Hydra 组合。运行环境为 Python 3.11 / Torch 2.7 CPU；历史方法通过 AST 提取执行，**不代表完整旧 policy 类在 Python 3.11 下兼容**。

另已执行 `python scripts/smoke_offline.py`：合成数据跑通离线 F_T 一轮训练、验证、权重保存、冻结重载与三步 rollout。使用 116 个训练窗口和 6 个验证窗口；这不验证真实采集数据或任务性能。临时数据与权重自动删除。

仓库静态检查覆盖 Python 解析、本地导入依赖、CRM URDF mesh 引用、预设关闭预测动作分支，以及超过 10 MB 的意外文件。

尚未执行：完整 Python 3.8 GPU 依赖安装、CUDA 扩展构建、Isaac Gym 环境创建、真实策略 checkpoint 加载、刚体训练或仿真成功率评估。不要把 CPU 检查通过等同于这些集成检查通过。Newton 与软体训练不属于本次迁移完成项。
