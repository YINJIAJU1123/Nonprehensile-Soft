# Nonprehensile-Soft

基于现有 **DyWA + SEER** 整理的连续研究代码库，用于后续统一刚体与软体的非抓取操作研究。

**当前交付是 Isaac Gym 刚体基线。Newton、可变形物体仿真和软体策略尚未实现。** 原代码目录与网络类名保留，以降低迁移时的权重兼容风险。

## 保留的内容

| 模块 | 入口 / 配置 |
| --- | --- |
| 刚体 PPO 教师 | `dywa/exp/train/train_ppo_arm.py` |
| DyWA 学生与 DAgger | `dywa/exp/train/train_rma.py`、`configs/dywa.json` |
| SEER 接触感知、K=3 未来监督 | `dywa/exp/train/train_rma_FT_kN.py`、`configs/seer.json` |
| 教师轨迹采集、离线未来模型 F_T | `collect_teacher_buffer.py`、`train_FT_offline.py` |
| 学生策略固定次数评估 | `dywa/exp/train/evaluate.py` |
| 共享环境、视觉编码、Franka 控制 | `dywa/src/` |
| 原刚体 T80 / U20 对象划分 | `data/splits/ral_v2/` |

DyWA 和 SEER 默认都使用修复后的持久历史缓存与安全快照。SEER 使用接触感知编码和未来辅助监督，标准部署路径不运行未来预测来选动作。这不是公共项目中同名的通用 SEER 模型。

不包含旧论文、实验日志、评估结果、数据集、策略权重、二进制扩展及其他实验仓库。部分关闭的实验分支仍作为共享类的内部依赖保留，避免在迁移阶段大改网络；它们不是这里支持的启动配置。

## 环境与外部资源

仿真基线沿用 Linux、Python 3.8、Isaac Gym 1.0rc4、PyTorch 1.11 + CUDA 11.3，以及匹配的 PyTorch3D / flash-attn / nvdiffrast。`environment-simulation.txt` 是继承的版本参考，**不是本次重新安装验证过的 lockfile**。优先复用现有训练环境；不要直接把全部旧依赖装到 Newton 的新环境里。

在已安装上述 GPU 运行依赖、CUDA toolkit 与 Eigen3 头文件的旧环境中：

```bash
python -m pip install --no-build-isolation -e ./dywa
cp .env.example .env
# 编辑 .env，填入本机数据与权重路径
set -a
source .env
set +a
```

`DGN_ROOT` 需要旧任务预处理后的 `metadata.json`、URDF、点云、碰撞几何等；`DGN_POSE_ROOT` 需要相应稳定姿态。ICP 编码器权重与教师、F_T、学生权重均由外部路径提供。

**保留教师的原目录结构**：`ckpt/step-N.ckpt` 必须配有同级 `stat/env-N.ckpt`。环境加载器自动读取这个精确配对。评估时 `ENV_CKPT` 必须对应被测学生，含归一化统计与 Phase-2 环境状态；不能随意换成另一份统计文件。

## 启动

先检查生成的参数；`--dry-run` 不导入 Isaac Gym / Torch，也不创建训练任务：

```bash
python scripts/run.py train --policy dywa --dry-run
python scripts/run.py train --policy seer --dry-run
```

以下命令需要完整 GPU 环境与 `.env` 中的相应资源。每次用新的输出目录；启动器拒绝覆盖已有目录。

```bash
# 教师：第一阶段；第二阶段先设置 TEACHER_CKPT 为第一阶段权重
python scripts/run.py teacher --stage 1 --steps 500000 --output outputs/teacher-stage1
python scripts/run.py teacher --stage 2 --steps 500000 --output outputs/teacher-stage2

# 已有合适教师可从这里开始。采集会自动读取配对的教师统计。
python scripts/run.py collect --steps 5000 --output outputs/buffer
# 把 BUFFER_PATH 设为采集生成的 .pt 文件
EPOCHS=50 BATCH=1024 K=3 python scripts/run.py world-model --output outputs/ft

# 设置 FT_CKPT 与 ENV_CKPT 后训练；两者默认从头训练学生。
python scripts/run.py train --policy dywa --steps 200000 --output outputs/dywa
python scripts/run.py train --policy seer --steps 200000 --output outputs/seer

# 设置 STUDENT_CKPT 与精确配对的 ENV_CKPT
python scripts/run.py eval --policy seer --steps 30000 --trials 200 \
  --split data/splits/ral_v2/unseen20_test_v2.json --output outputs/eval-seer-u20
```

`--gpu 1` 会把物理 GPU 1 映射为进程内 `cuda:0`；`--python /path/to/python` 可指定旧环境解释器。学生训练可用 `--seed`、`--num-env`；追加 Hydra 参数示例：

```bash
python scripts/run.py train --policy seer --dry-run -- ++student.learning_rate=0.0001
```

采集与离线 F_T 训练要求同一 Git commit 和代码指纹，修改代码后应重新采集。新输出标记为 `NONPREHENSILE_SOFT_DEV`，不沿用旧论文的正式实验认证。默认训练时长、DAgger 调度是开发起点，不是论文复现实验承诺。恢复旧权重须核对架构和历史缓存模式；启动器不提供自动恢复训练。

## 验证与开发边界

```bash
python scripts/check_repository.py
# CPU 测试只需要 torch、hydra-core；不需要 Isaac Gym
python -m unittest discover -s tests -v
# 可选：合成数据的一轮离线训练/重载检查，额外需要 tensorboard
python scripts/smoke_offline.py
```

本次迁移检查覆盖 Python 语法、本地导入、配置组合、支持的机器人资产、历史缓存、未来监督梯度、episode 边界和学生执行循环。**完整 Isaac Gym 训练 / 仿真评估没有在本次迁移环境运行验证**；新评估入口完成了静态与模拟环境单元测试，仍需 GPU 冒烟测试。

来源、改动与尚待验证的事项见 [迁移说明](docs/MIGRATION.md)。软体扩展的改动位置见 [开发边界](docs/SOFT_EXTENSION.md)。继承原仓库 [CC BY-NC 4.0](LICENSE) 许可；第三方组件保留其各自许可。
