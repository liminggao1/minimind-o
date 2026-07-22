# Git 常用命令笔记
## 克隆仓库
```powershell
### 克隆项目
#下载项目代码及其 Git 历史。以后应使用 `git pull` 获取更新，不需要重复下载 ZIP。
git clone https://github.com/用户名/项目名.git

### 查看仓库状态
git status
### 查看远程仓库地址
git remote -v

### 通常 `origin` 表示克隆项目时使用的远程仓库（可以是官方也可以是fork的仓库）。
origin  https://github.com/jingyaogong/minimind-o (fetch)
origin  https://github.com/jingyaogong/minimind-o (push)

## `jingyaogong/minimind-o` 是别人的官方仓库。通常可以正常拉取，但直接执行下面的命令可能因为没有写入权限而失败：
git push origin main
```

### 将自己的分支保存到 GitHub

需要把 `main` 推送到自己的 GitHub 时，先在 GitHub 页面 Fork 官方项目。Fork 会在自己的账号下创建一份远程仓库副本。
然后将仓库整理为：
```text
## 远程仓库
upstream  官方仓库，用于获取官方更新
origin    自己的 Fork，用于保存和推送个人代码
## 本地仓库 
master -> upstream/master
main   -> origin/main
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
git remote add origin https://github.com/liminggao1/minimind-o.git
git remote -v
```
输出类似：
```
origin    https://github.com/你的用户名/minimind-o.git (fetch)
origin    https://github.com/你的用户名/minimind-o.git (push)
upstream  https://github.com/jingyaogong/minimind-o (fetch)
upstream  https://github.com/jingyaogong/minimind-o (push)
```
先暂存更改：
```
# 从官方主分支创建自己的学习分支
##`-c` 是 `--create` 的缩写，表示创建新分支并立即切换过去。
git switch -c main
git add note/gitnote.md
git commit -m "完善 Git 学习笔记"
```

第一次推送 `main` 分支：

```powershell
git push -u origin main
```
git push：固定动作，意为“我要把本地代码推送到远程仓库”。

-u：是 --set-upstream 的缩写，意为“建立上下游的追踪关联”（这是重点）。

origin：远程仓库的别名（你在上一步添加的那个自己账号下的仓库）。

main：你本地的分支名字。

`-u` 会建立本地 `main` 与远程 `origin/main` 的跟踪关系。以后在 `main` 分支上可以简写为：

```powershell
git push
```

配置 Fork 后，从官方仓库更新代码应使用 `upstream`：

```powershell
# 从官方仓库（upstream）下载所有最新的提交记录到本地，
# 但此时只是暂存在本地缓存中，还没有合并到你任何工作分支里。
git fetch upstream

# 切换到本地的默认主分支 master。
git switch master

# 将刚才下载的官方 upstream/master 代码，以“快进”方式合并到本地的 master。
# --ff-only 确保若本地 master 有分叉改动则直接报错，强制保持本地主分支绝对纯净，
# 使其与官方 master 处于完全一致的线性状态。
git merge --ff-only upstream/master

# 将本地已同步好的 master 分支，推送到你自己账号下的 GitHub 远程仓库（origin）。
# 这一步是为了让你 GitHub 上的 master 分支也和官方保持同步，方便后续发 PR。
git push origin master

# 切换到你用来写代码/学习的开发分支 main。
git switch main

# 将刚刚同步好的本地 master（包含官方最新代码）合并进 main。
git merge master

# 将合并了官方更新、且包含你个人代码修改的 main 分支，
# 推送到你自己 GitHub 仓库（origin）的远程 main 分支。
git push origin main

```

## 日常流程

```powershell
# 进入自己的学习分支
git switch main

# 修改代码后检查并提交
git status
git diff
git add .
git commit -m "描述本次修改"

# 获取并检查官方更新
git fetch upstream
git log --oneline main..upstream/master
git diff main...upstream/master

# 更新干净的 master
git switch master
git pull --ff-only upstream master
# 将官方更新合入自己的分支
git switch main
git merge master
```
