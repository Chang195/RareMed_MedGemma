# 在VS Code用Python运行

打开整个项目文件夹 RareMed_MedGemma，菜单“终端 → 新建终端”，逐条运行：

```text
python run.py validate --run my-first-run
python run.py processor --run my-first-run
python run.py forward --run my-first-run
```

确认显示 `Forward PASS loss=...` 后，再单独运行：

```text
python run.py smoke --steps 2 --run my-first-run
```

四步使用相同实验名。下一轮换成 my-second-run。结果和日志在 G:\RareMed_MedGemma_Local\runs\my-first-run。已有基线或训练结果的目录会拒绝再次训练，保护旧结果。

run.py 使用Python标准库，自动选择项目 .venv，设置G盘缓存和凭据路径并保存日志。不需要激活环境、运行.ps1、修改PowerShell执行策略或重新登录。现有Python 3.8也能启动入口，实际模型使用项目Python 3.11。

若终端不识别python，使用明确路径：

```text
.\.venv\Scripts\python.exe run.py validate --run my-first-run
```

## 点击运行和调试

需要Microsoft Python和Python Debugger扩展。按Ctrl+Shift+D，选择RareMed 1至RareMed 4，顺序按Ctrl+F5运行，或F5调试。每次提示实验名时填写相同名称，下一轮换新名。配置调用Python入口，同样保存控制台日志。

第1项校验数据；第2项检查processor与掩码；第3项只做单样本前向；第4项先保存基线再训练2步。默认NF4/FP32/SDPA，不改变原图或来源目标文本。

产物包括 dataset-audit.json、processor-check.json、mask-*.json、forward.json、baseline.json、smoke-log.json、adapter/，以及每阶段带时间戳的.log。

Ctrl+C停止；不要同时启动两个GPU任务。这是同病例两张图的冒烟检查，training_ready保持false，不报告泛化性能。
