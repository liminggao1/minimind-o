# Issue Tracker: Local Markdown

本项目的问题、规格和实现任务保存在 `.scratch/` 目录中。

## 文件约定

- 一个功能对应一个 `.scratch/<feature-slug>/` 目录
- 规格文件：`.scratch/<feature-slug>/spec.md`
- 任务文件：`.scratch/<feature-slug>/issues/NN-<slug>.md`
- 每个任务使用 `Status:` 记录当前状态
- 讨论记录追加在文件底部的 `## Comments` 下

## 工作方式

当技能要求发布问题或规格时，在对应的 `.scratch/<feature-slug>/` 目录创建 Markdown 文件。

读取任务时，直接读取用户提供的任务文件路径或任务编号对应的文件。
