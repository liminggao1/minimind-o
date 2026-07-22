# Git 常用命令笔记

## 1. Git 的基本工作过程

```text
工作区修改文件
    ↓ git add
暂存区
    ↓ git commit
本地 Git 提交历史
    ↓ git push
远程仓库
```

- **工作区**：当前正在编辑的文件。
- **暂存区**：准备放进下一次提交的修改。
- **本地仓库**：已经通过 `git commit` 保存的版本历史。
- **远程仓库**：GitHub 等服务器上的仓库。

## 2. 克隆和查看仓库

### 克隆项目

```powershell
git clone https://github.com/用户名/项目名.git
```

下载项目代码及其 Git 历史。以后应使用 `git pull` 获取更新，不需要重复下载 ZIP。

### 查看仓库状态

```powershell
git status
```

显示当前分支、已修改文件、已暂存文件和未跟踪文件。这是最常用且安全的检查命令。

简洁显示：

```powershell
git status --short --branch
```

### 查看远程仓库地址

```powershell
git remote -v
```

通常 `origin` 表示克隆项目时使用的远程仓库。

本项目的输出是：

```text
origin  https://github.com/jingyaogong/minimind-o (fetch)
origin  https://github.com/jingyaogong/minimind-o (push)
```

每一部分的含义如下：

```text
origin                                     远程仓库的本地别名
https://github.com/jingyaogong/minimind-o  远程仓库地址
(fetch)                                    拉取代码时使用
(push)                                     推送代码时使用
```

- `origin` 只是 Git 自动设置的默认名称，不是固定关键字，也可以改名。
- `git fetch origin` 和 `git pull origin master` 会使用标记为 `(fetch)` 的地址。
- `git push origin study` 会使用标记为 `(push)` 的地址。
- 拉取和推送显示相同地址是正常现象。
- 显示 `(push)` 地址只代表 Git 知道向哪里推送，不代表当前账号拥有写入权限。

`jingyaogong/minimind-o` 是别人的官方仓库。通常可以正常拉取，但直接执行下面的命令可能因为没有写入权限而失败：

```powershell
git push origin study
```

只在本地学习时，不需要修改远程配置。使用下面的命令就可以获取官方更新：

```powershell
git fetch origin
```

本项目的官方默认分支是 `master`，因此实际更新流程是：

```powershell
git switch master
git pull --ff-only origin master
git switch study
git merge master
```

### 将自己的分支保存到 GitHub

需要把 `study` 推送到自己的 GitHub 时，先在 GitHub 页面 Fork 官方项目。Fork 会在自己的账号下创建一份远程仓库副本。

然后将远程仓库整理为：

```text
upstream  官方仓库，用于获取官方更新
origin    自己的 Fork，用于保存和推送个人代码
```

先把现在的 `origin` 改名为 `upstream`：

```powershell
git remote rename origin upstream
```
变成了：
```powershell
upstream  https://github.com/jingyaogong/minimind-o (fetch)
upstream  https://github.com/jingyaogong/minimind-o (push)
```


再添加自己的 Fork。下面地址中的 `你的GitHub用户名` 必须替换为真实用户名，不能原样输入：

```powershell
git remote add myorigin https://github.com/liminggao1/minimind-o.git
git remote -v
```
输出类似：
```
myorigin    https://github.com/你的用户名/minimind-o.git (fetch)
myorigin    https://github.com/你的用户名/minimind-o.git (push)
upstream  https://github.com/jingyaogong/minimind-o (fetch)
upstream  https://github.com/jingyaogong/minimind-o (push)
```
先暂存更改：
```
git switch study
git add note/gitnote.md
git commit -m "完善 Git 学习笔记"
```

第一次推送 `study` 分支：

```powershell
git push -u myorigin study
```
git push：固定动作，意为“我要把本地代码推送到远程仓库”。

-u：是 --set-upstream 的缩写，意为“建立上下游的追踪关联”（这是重点）。

myorigin：远程仓库的别名（你在上一步添加的那个自己账号下的仓库）。

study：你本地的分支名字。

`-u` 会建立本地 `study` 与远程 `origin/study` 的跟踪关系。以后在 `study` 分支上可以简写为：

```powershell
git push
```

配置 Fork 后，从官方仓库更新代码应使用 `upstream`：

```powershell
# ================= 第一步：获取官方最新代码 =================
# 从官方仓库（upstream）下载所有最新的提交记录到本地，
# 但此时只是暂存在本地缓存中，还没有合并到你任何工作分支里。
git fetch upstream

# ================= 第二步：更新本地主线（master） =================
# 切换到本地的默认主分支 master。
git switch master

# 将刚才下载的官方 upstream/master 代码，以“快进”方式合并到本地的 master。
# --ff-only 确保若本地 master 有分叉改动则直接报错，强制保持本地主分支绝对纯净，
# 使其与官方 master 处于完全一致的线性状态。
git merge --ff-only upstream/master

# ================= 第三步：同步自己的 GitHub 仓库（myorigin） =================
# 将本地已同步好的 master 分支，推送到你自己账号下的 GitHub 远程仓库（myorigin）。
# 这一步是为了让你 GitHub 上的 master 分支也和官方保持同步，方便后续发 PR。
git push myorigin master

# ================= 第四步：将新代码融入你的开发分支（study） =================
# 切换到你用来写代码/学习的开发分支 study。
git switch study

# 将刚刚同步好的本地 master（包含官方最新代码）合并进 study。
git merge master

# ================= 第五步：推送自己的开发成果 =================
# 将合并了官方更新、且包含你个人代码修改的 study 分支，
# 推送到你自己 GitHub 仓库（myorigin）的远程 study 分支。
git push myorigin study

```

## 3. 分支操作

建议保持 `master` 与官方代码一致，在 `study` 分支中学习和修改：

```text
master  官方代码，不进行个人修改
study   自己的学习和修改
```

### 查看分支

```powershell
git branch
```

带 `*` 的分支是当前分支。

只显示当前分支名：

```powershell
git branch --show-current
```

### 创建并切换分支

```powershell
git switch -c study
```

`-c` 是 `--create` 的缩写，表示创建新分支并立即切换过去。

它等价于：

```powershell
git branch study
git switch study
```

### 切换已有分支

```powershell
git switch study
git switch master
```

回到上一次所在的分支：

```powershell
git switch -
```

### 删除已经不需要的分支

先切换到其他分支，再删除：

```powershell
git switch master
git branch -d study
```

`-d` 会在分支尚未合并时阻止删除，避免误删提交。不要随意使用强制删除参数 `-D`。

## 4. 保存本地修改

### 查看修改

```powershell
git status
git diff
```

`git diff` 显示尚未加入暂存区的具体代码变化。

### 将修改加入暂存区

添加当前目录及其子目录中的全部修改：

```powershell
git add .
```

其中 `.` 表示当前目录。这个命令只把修改加入暂存区，不会创建提交，也不会上传到 GitHub。

只添加指定文件：

```powershell
git add train.py
```

取消暂存，但保留文件修改：

```powershell
git restore --staged train.py
```

### 创建本地提交

```powershell
git commit -m "修改训练代码"
```

`-m` 后面是本次修改的说明。`git commit` 只保存到本地 Git 历史，不会自动上传。

### 查看提交历史

```powershell
git log --oneline
```

用图形方式查看所有分支的关系：

```powershell
git log --graph --oneline --decorate --all
```

## 5. 获取并合并官方更新

本项目的官方主分支名为 `master`，下面的命令均按本项目的真实分支名称编写。

### 只获取远程信息

```powershell
git fetch origin
```

`fetch` 下载远程分支和提交信息，但不会修改当前文件，适合先检查官方更新。

查看官方新增但 `study` 中还没有的提交：

```powershell
git log --oneline study..origin/master
```

查看官方从共同版本开始修改的代码：

```powershell
git diff study...origin/master
```

查看自己从共同版本开始修改的代码：

```powershell
git diff origin/master...study
```

### 更新本地 master

```powershell
git switch master
git pull --ff-only origin master
```

`pull` 获取并更新本地分支。`--ff-only` 要求只能快进更新，可以防止在 `master` 上意外产生合并提交。

### 将官方更新合入 study

```powershell
git switch study
git merge master
```

这表示把 `master` 的更新合并进当前的 `study` 分支。没有冲突时 Git 会自动完成。

## 6. 处理合并冲突

发生冲突后先查看文件：

```powershell
git status
```

在 VS Code 合并编辑器中：

- **Current / 当前更改**：当前 `study` 分支的代码。
- **Incoming / 传入更改**：准备合入的 `master` 代码。
- **Result / 结果**：最终保留的代码，可以手动编辑成不同于两边的新结果。

解决所有冲突后：

```powershell
git add .
git merge --continue
```

放弃本次合并并恢复到合并前：

```powershell
git merge --abort
```

## 7. 临时保存未提交修改

切换分支前如果暂时不想提交，可以使用：

```powershell
git stash
git switch master
```

回到原分支后恢复：

```powershell
git switch study
git stash pop
```

## 8. 推荐的日常流程

```powershell
# 进入自己的学习分支
git switch study

# 修改代码后检查并提交
git status
git diff
git add .
git commit -m "描述本次修改"

# 获取并检查官方更新
git fetch origin
git log --oneline study..origin/master
git diff study...origin/master

# 更新干净的 master
git switch master
git pull --ff-only origin master

# 将官方更新合入自己的分支
git switch study
git merge master
```

## 9. 常见注意事项

- 命令示例中的分支名要替换为仓库实际使用的名称。
- `<旧提交SHA>` 这类文字是说明用的占位符，不能连同尖括号原样输入。
- 切换分支前先运行 `git status`，确认修改已经提交或暂存。
- `git add .` 前先检查文件列表，避免提交数据集、模型权重、缓存或密钥。
- 不要随意使用 `git reset --hard`，它可能永久丢弃尚未提交的修改。

## 10. LF 和 CRLF 换行符警告

在 Windows 中执行 `git add .` 时，可能看到：

```text
warning: LF will be replaced by CRLF in note/gitnote.md.
The file will have its original line endings in your working directory
```

这只是换行符格式警告，不是错误，`git add` 仍然成功执行，也不会改变文字或代码逻辑。

两种换行符的含义：

```text
LF    Linux、macOS 和 Git 仓库中常用的换行符
CRLF  Windows 中常用的换行符
```

当前电脑的 Git 配置是：

```powershell
git config --show-origin --get core.autocrlf
```

输出中的 `true` 表示启用了 Windows 自动换行转换：

```text
提交到 Git 仓库时：通常将 CRLF 规范化为 LF
从 Git 检出到 Windows 工作区时：可能将 LF 转换为 CRLF
```

因此，这段警告表示 `gitnote.md` 当前使用 LF；Git 提醒该文件以后被重新检出时，工作区版本可能转换为 CRLF。通常不需要处理，可以继续提交：

```powershell
git add note/gitnote.md
git commit -m "完善 Git 笔记"
```

查看一个已跟踪文件的实际换行状态：

```powershell
git ls-files --eol -- note/gitnote.md
```

本文件当前显示：

```text
i/lf  w/lf  attr/  note/gitnote.md
```

各字段含义如下：

```text
i/lf   暂存区中的文件使用 LF
w/lf   工作区中的文件使用 LF
attr/  仓库没有通过 .gitattributes 为该文件指定换行规则
```

不要仅为了消除这条警告就随意修改全局 Git 配置。多人跨平台协作时，应由项目通过 `.gitattributes` 统一换行规则；当前学习项目保持现有设置即可。
