# Git 常用命令笔记

## 1. 当前仓库结构

```powershell
# 远程仓库
# upstream：GitHub 官方仓库，只用于获取官方更新
# origin：自己的 GitHub Fork，用于保存个人代码

# 本地分支及跟踪关系
# master -> upstream/master
# main   -> origin/main

# 使用原则
# master 只跟随官方代码，不提交个人修改
# main 用于学习、修改、提交和推送个人代码
```

## 2. 首次克隆和配置远程仓库

以下流程只在第一次建立项目时执行一次。当前仓库已经配置完成，不要重复执行。

```powershell
# 1. 克隆官方仓库
git clone https://github.com/jingyaogong/minimind-o.git

# 2. 进入项目目录
Set-Location minimind-o

# 3. 查看当前远程仓库
# 刚克隆时，origin 默认指向官方仓库
git remote -v

# 4. 将官方仓库的本地别名从 origin 改成 upstream
git remote rename origin upstream

# 5. 添加自己的 GitHub Fork，并命名为 origin
git remote add origin https://github.com/liminggao1/minimind-o.git

# 6. 检查配置
git remote -v

# 预期关系：
# origin   -> https://github.com/liminggao1/minimind-o.git
# upstream -> https://github.com/jingyaogong/minimind-o
```

## 3. 首次创建个人 main 分支

以下流程只在本地还没有 `main` 时执行一次。当前仓库已经存在 `main`，不要重复执行。

```powershell
# 1. 确保从干净的官方 master 开始
git switch master

# 2. 从当前 master 创建并切换到 main
# -c 是 --create 的缩写
git switch -c main

# 3. 第一次推送 main，并建立跟踪关系
# -u 是 --set-upstream 的缩写
git push -u origin main

# 4. 检查分支及跟踪关系
# 应看到 master 跟踪 upstream/master，main 跟踪 origin/main
git branch -vv
```

## 4. 在 main 上修改和提交代码

```powershell
# 1. 切换到个人开发分支
git switch main

# 2. 修改前先确认状态
git status

# 3. 修改代码后查看具体变化
git diff

# 4. 推荐只暂存本次需要提交的文件
git add note/gitnote.md

# 如果确认当前目录下所有修改都需要提交，才使用下面这条命令
# git add .

# 5. 查看已经进入暂存区、即将提交的修改
git diff --cached

# 6. 创建本地提交
git commit -m "描述本次修改"

# 7. main 已跟踪 origin/main，可以直接推送
git push
```

`git add` 只把修改放入暂存区，不会创建提交，也不会上传到 GitHub。

```powershell
# 取消指定文件的暂存，但保留工作区修改
git restore --staged note/gitnote.md
```

## 5. 查看官方仓库更新

```powershell
# 1. 获取官方最新提交，但不修改当前工作区和本地分支
git fetch upstream

# 2. 查看官方有、但个人 main 还没有的提交
git log --oneline main..upstream/master

# 3. 先查看官方代码变化的文件统计
# 三个点表示从共同祖先开始查看 upstream/master 的变化
git diff --stat main...upstream/master

# 4. 查看官方具体修改了哪些代码
git diff main...upstream/master
```

## 6. 将官方更新合入个人 main

```powershell
# 1. 切换分支前确认工作区干净
# 如果存在未提交修改，先提交或使用 git stash
git status

# 2. 获取官方最新提交
git fetch upstream

# 3. 更新本地 master
git switch master

# 只允许快进更新，保证 master 不产生个人分叉
git merge --ff-only upstream/master

# 4. 可选：让自己 Fork 中的 origin/master 也与官方同步
git push origin master

# 5. 回到个人开发分支
git switch main

# 6. 将最新的官方 master 合入 main
git merge master

# 7. 将合并后的 main 推送到自己的 GitHub
# main 已跟踪 origin/main，因此可以简写为 git push
git push
```

这里使用 `fetch + merge --ff-only`，不再重复执行 `pull`。两者的职责更清楚：先获取，再更新本地 `master`。

## 7. 处理合并冲突

在 `main` 上执行 `git merge master` 时，如果两边修改了相同位置，Git 会暂停并提示冲突。

```powershell
# 1. 查看发生冲突的文件
git status

# 2. 使用 VS Code 打开冲突文件并编辑最终结果
# Current：个人 main 中的代码
# Incoming：准备合入的 master 中的官方代码

# 3. 解决后暂存冲突文件
git add 冲突文件路径

# 4. 继续完成合并
git merge --continue

# 5. 合并完成后推送个人 main
git push

# 如果不想继续本次合并，使用下面命令恢复到合并前
# git merge --abort
```

`冲突文件路径` 是说明用的占位文字，执行时必须替换成真实路径，不能原样输入。

## 8. 临时保存未提交修改

需要切换分支，但当前修改还不适合提交时，可以临时保存。

```powershell
# 临时保存当前未提交修改
git stash push -m "临时保存"

# 此时可以安全切换分支并执行其他操作
git switch master

# 回到个人开发分支
git switch main

# 恢复刚才临时保存的修改
git stash pop
```


## 10. 日常工作流程速查

```powershell
# 一、在个人 main 上修改、提交并推送
git switch main
git status
git diff
git add 需要提交的文件路径
git diff --cached
git commit -m "描述本次修改"
git push

# 二、获取并查看官方更新
git fetch upstream
git log --oneline main..upstream/master
git diff --stat main...upstream/master

# 三、更新干净的 master
git switch master
git merge --ff-only upstream/master
git push origin master

# 四、将官方更新合入个人 main
git switch main
git merge master
git push
```
