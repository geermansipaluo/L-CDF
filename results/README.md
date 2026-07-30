# LCDF实验结果目录

这里统一保存训练、一般评测和随机环境评测结果。每次运行使用
`<实验版本>_<YYYYMMDD_HHMMSS>` 作为独立目录名，禁止不同运行互相覆盖。

- `training/`：后续训练日志、checkpoint索引和训练曲线。
- `evaluation/`：一般闭环评测。
- `random_environment/`：20场景随机环境基准。

运行产物默认不提交到Git；各目录只保留 `.gitignore` 和本文档。
