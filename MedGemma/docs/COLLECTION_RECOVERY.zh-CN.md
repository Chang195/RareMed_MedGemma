# v2 采集中断修复

保留原始 RareMed-Pairs 和旧运行。修复范围仅为 Windows checkpoint I/O 与访问受限状态处理。

- JSON 保存使用同目录唯一临时文件、flush/fsync、最多 9 次替换尝试。
- 替换持续失败时保留完整临时文件，原 checkpoint 不覆盖。
- 从原 checkpoint 和旧 .tmp 中验证来源清单哈希、配置、历史进度一致性和图片哈希，恢复较完整版本。
- 保存修复前代码、原 checkpoint、旧 .tmp、校验记录到 local_run/runtime-repair-backup-*。
- 验证码页面标记 held_access_challenge，不是疾病阴性，也不是医学筛选排除。
- 已标记的 held 记录由原续跑逻辑跳过；新遇到验证码同样保留为待处理，不绕过验证。
- 非验证码的无图/不支持页面与 figure ID 解析错误不擅自改写或当作成功。

当前恢复点：144 篇，40 张候选图像；40 个原始图像哈希通过。
103 篇确认验证码页面单独保留。其余 7 篇失败需后续逐项审查。
这些数字只表示中断时进度，不代表完整队列结果。

在 VS Code 的 RareMed_MedGemma 终端可续跑：

```powershell
.\.venv\Scripts\python.exe run_original_test_pipeline_v2.py check
.\.venv\Scripts\python.exe run_original_test_pipeline_v2.py collect
```

不要重新 search、不要删除历史 checkpoint 或缓存；只开一个采集进程。
已完成图像会验证哈希后复用。访问受限条目不自动重试，因此续跑也不能保证取得这些原图。
需另行处理正常来源访问，不能靠本次文件保存修复解除网站验证。
本次修复未启动联网续跑或付费抽取。

核验记录：local_run/runtime_repair_receipt.json 和 local_run/file_verification.json。
