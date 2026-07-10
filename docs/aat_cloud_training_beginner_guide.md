# AAT 模型 AutoDL 云端训练：初学者操作指南

本指南面向第一次租用 GPU 的使用者。当前阶段**不要购买实例**；收到“现在可以买”的明确通知后再执行第 1 步。

## 你需要做的事情只有三件

1. 按指定配置购买一台实例。
2. 把临时 SSH 登录信息发给负责训练的人，或配置 SSH 公钥。
3. 训练结束后确认结果已备份，然后关机。

代码部署、Python 环境、数据校验、训练命令、日志和结果下载都由脚本处理。

## 1. 购买时怎么选

- GPU 数量：`1`。
- 显存：至少 `24 GB`；优先 RTX 4090 24GB，价格接近时可选 RTX 5090 32GB。
- 不需要 H100，也不需要多卡。
- CPU 内存：建议至少 `32 GB`，截图中的 90GB 足够。
- 系统盘：30GB 可以使用，但不要把训练数据和 checkpoint 放在系统盘。
- 数据盘：建议扩展到 `100–200 GB`；训练文件统一放到 `/root/autodl-tmp/AAT-Training`。
- 镜像：选择已包含 PyTorch 2.x、CUDA 12.x 的官方基础镜像即可。

AutoDL 文档说明数据盘默认挂载到 `/root/autodl-tmp`，系统盘和数据盘用途不同；本地数据盘没有可靠性承诺，所以最终结果还必须下载或同步到其他存储：[实例目录说明](https://www.autodl.com/docs/env/)、[数据保留说明](https://www.autodl.com/docs/instance_data/)。

## 2. 买完后发什么信息

控制台会显示类似：

```text
ssh -p 12345 root@connect.example.autodl.com
密码：一次性或临时密码
```

需要提供：

- 主机名，例如 `connect.example.autodl.com`；
- 端口，例如 `12345`；
- 用户名，通常为 `root`；
- 临时密码。

更安全的方式是配置 SSH 公钥，只发送主机、端口和用户名。**私钥文件绝对不能发送**。AutoDL 官方的 Windows SSH 和公钥配置步骤见：[SSH 远程连接](https://www.autodl.com/docs/ssh/)。

训练完成后应立即修改/作废临时密码，或移除临时公钥。不要把密码写入 Git、聊天截图、代码、配置或训练日志。

## 3. 上传什么，不上传什么

GitHub 仓库只托管：

- Python 训练代码；
- YAML 配置；
- 测试；
- 文档和启动脚本。

以下内容不得上传公共 GitHub：

- 原始 gel 图像；
- `snapshot_v0`、`frozen_v1` 及裁剪图；
- checkpoint；
- OOF 预测、实验输出和可能包含样本标识的日志。

数据单独压缩后传到 AutoDL 数据盘，并在上传前后核对 SHA-256。网盘适合传输和备份，训练前再复制到本地数据盘以获得更好的读取速度：[AutoDL 文件存储说明](https://www.autodl.com/docs/nas/)。

## 4. 登录后的目录

```text
/root/autodl-tmp/AAT-Training/
├── repo/                 # 从 GitHub 获取的纯代码
├── data/frozen_v1/       # 私有正式数据，不进 Git
├── experiments/          # checkpoints、OOF 预测、指标
├── reports/              # 图表与论文证据
└── logs/                 # 训练日志
```

脚本会先执行：

```bash
bash scripts/cloud/bootstrap_autodl.sh
python scripts/cloud/verify_cloud_environment.py
```

只有验证报告同时显示以下项目通过，才允许正式训练：

- CUDA 可用且 GPU 型号/显存符合要求；
- `frozen_v1` manifest 哈希正确；
- 固定 folds 哈希正确；
- 5×3 分组切分无 gel 泄漏；
- 六个常见类别均可评估；
- Git revision 和配置已记录。

## 5. 为什么使用 tmux

SSH 断线不应该终止训练。AutoDL 官方也建议使用 `screen` 或 `tmux` 守护训练进程：[SSH 文档](https://www.autodl.com/docs/ssh/)。基本操作：

```bash
tmux new -s aat
bash scripts/cloud/run_snapshot_smoke.sh
```

按 `Ctrl+B`，再按 `D`，可以离开但不中止任务。重新连接后：

```bash
tmux attach -t aat
```

## 6. 训练分两步，不直接烧完整预算

第一步是 GPU 冒烟：只跑少量数据/少量 batch，检查显存、速度、checkpoint、OOF 格式和日志。冒烟失败就停止，不开始完整嵌套交叉验证。

第二步才是正式运行，顺序固定：

1. ResNet-18；
2. EfficientNet-B0；
3. ResNet-50 proposal 对照；
4. Inception-v3 proposal 对照；
5. 根据 OOF macro-F1、Rare Top-3、校准和效率选择最佳 backbone；
6. 最佳 backbone 的等位基因头、embedding 检索和转诊校准。

所有模型复用同一个 `frozen_v1` 和同一个固定 fold 文件。

## 7. 省钱和关机

- 上传、下载、查看日志等不需要 GPU 的工作可以用 AutoDL “无卡模式”。官方页面显示该模式适合调试和数据操作，当前页面标价为 0.1 元/小时，但购买时仍以控制台实际价格为准：[省钱说明](https://www.autodl.com/docs/save_money/)。
- GPU 正式训练结束后先备份 `experiments/`、`reports/` 和日志，再关机。
- 关机不等于永久备份；AutoDL 文档提示连续关机达到释放条件或主机下架时，实例数据可能被清除，因此重要结果必须另存：[数据保留说明](https://www.autodl.com/docs/instance_data/)。
- 不要在未经检查的命令后自动关机。先让训练脚本写出完成 manifest 和哈希，再由监控脚本决定是否关机。

## 8. 何时算训练完成

看到终端显示 `finished` 还不够。必须同时存在：

- 每个外层折的 checkpoint；
- 完整且无重复的 OOF 预测；
- resolved config、seed、代码 revision；
- 数据和 folds manifest 哈希；
- common、calibration、rare、bootstrap 指标；
- `run_manifest.json` 状态为 `complete`；
- 结果备份校验通过。

这些条件全部通过后，才通知你关机并停止计费。

