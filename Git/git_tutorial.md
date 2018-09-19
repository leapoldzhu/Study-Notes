# GIT Tutorial
## Contents  
- [Create/Modify Repository](#1-create-or-modify-repository)  
- [Back to Previous Version](#2-back-to-previous-version)
- [Branch management](#3-branch-management)
- [Reference](#reference)
## 1 Create or Modify Repository
### 1.1 Setup
- [Setup Guide](https://morvanzhou.github.io/tutorials/others/git/1-2-install/)

### 1.2 First Repository
#### 1.2.1 Initial
- cd: Change directory
- git config user.name/user.email: confirm user name & email
- git config --global user.name "": Change user name
- git config --global user.email "": Change user email
- git init: Initial a empty repository

#### 1.2.2 Add file
- ls -a: Display all files
- open .git: Open Git
- touch *filename*: Create file in the PATH
- git status (-s): Check git status; use -s to compress status
- git add *filename*: Add file into the Repository, without *filename*, it will add all files not in repository into it

#### 1.2.3 Commit
- git commit -m "your commit here": commit the change and record it

![Git](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter1/gitprocess.png  "Git Process")

### 1.3 Log & Diff
- git log (--oneline): Check user information and changes made last time; use --oneline to compress information
- git diff (--cached/HEAD): Check different between each version of codes; use --cached when you change the modified file to staged station; use HEAD to check difference between staged & unstaged situation

    When you do some change to files, enter "git status", you'll find red words remind you to "git add", then you can "git commit -m """. When you use "git log" this time, you'll find sth. different.

    '''
    #### 对比三种不同 diff 形式
        $ git diff HEAD     # staged & unstaged

        @@ -1 +1,3 @@
        -a = 1  # 已 staged
        +a = 2  # 已 staged
        +b = 1  # 已 staged
        +c = b  # 还没 add 去 stage (unstaged)
        -----------------------
        $ git diff          # unstaged

        @@ -1,2 +1,3 @@
        a = 2  # 注: 前面没有 +
        b = 1  # 注: 前面没有 +
        +c = b  # 还没 add 去 stage (unstaged)
        -----------------------
        $ git diff --cached # staged

        @@ -1 +1,2 @@
        -a = 1  # 已 staged
        +a = 2  # 已 staged
        +b = 1  # 已 staged
        '''
    
## 2 Back to Previous Version
### 2.1 Reset
- git commit --amend --no-edit: use --amend to put changed stage into last log information; use --no-edit to sustain last commend
- git reset: Bring file back from staged to unstaged
- git reset --hard HEAD(~no.): Return to last staged file; use ~+no. to decide back how many versions
- git reset --hard IDno.: Return to the exact version
- git reflog: Check all changes done with HEAD
![Git-Head](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter2/2-2-1.png "Git Head-1")
![Git-Head](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter2/2-2-2.png "Git Head-2")
![Git-Head](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter2/2-2-3.png "Git Head-3")
![Git-Head](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter2/2-2-4.png "Git Head-4m")

### 2.2 Checkout
- git checkout *Commit no.* -- *filename*: Reset single file to exact version

## 3 Branch management
Different branches for different users/situations.
### 3.1 Build/Switch/Merge
- git log --oneline --graph: On left side, use * to represent different branches
- git checkout -b *name*: Create a branch name as *name*
- git branch *name*: Create a branch name as *name*
- git branch: Star on the left side represent active branch
- git checkout -*name*: Switch to branch *name*
- git branch -d *name*: delete the branch *name*
- git commit -am "comment": For files already in the repository, use -am to add and comment the change
- git merge --no-ff -m "comment" *branch name*: Merge the branch to active branch

### 3.2 Merge confilct - merge
In Master branch, there's a modify different from branch to be merged

![Branch conflict](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter3/Branchconflict.png "Branch conflict")

Just modify the file and save the result

![Branch conflict modified](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter3/ConfilctModified.png "Branch conflict modified")

### 3.3 Rebase
Sth. changed in master branch, you wish to take this change to your branch, so you should use rebase to realize this operation
- git rebase *branch name*: It will rebase your appointed branch base on active branch 
**Not use it in shared branch!**
![Branch rebase 1](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter3/4-3-1.png "Branch rebase 1")

![Branch rebase 2](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter3/4-3-2.png "Branch rebase 2")

![Branch rebase 3](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter3/4-3-3.png "Branch rebase 3")

![Branch rebase 4](https://raw.githubusercontent.com/leapoldzhu/Study-Note/master/Git/img/Chapter3/4-3-4.png "Branch rebase 4")

### 3.4 Stash
Make your job hang on, and do some other things, then continue your job before.
- git stash: Hang on your job
- git stash list: Check tasks
- git stash pop: Pop your job, now, you can continue~

# Reference
[莫烦Python - Git](https://morvanzhou.github.io/tutorials/others/git/)

[Vedio On bilibili](https://www.bilibili.com/video/av16377923)
