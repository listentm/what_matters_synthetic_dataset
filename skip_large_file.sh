#!/bin/bash

# 设置文件大小限制（单位：字节）
max_size=1000000  # 文件大小限制为 1MB

# 1. 处理当前暂存区中的大文件
echo "Checking for large files in the current working directory..."
for file in $(git ls-files); do
    # 检查文件是否存在，且大小超过限制
    if [ -f "$file" ] && [ $(stat -c%s "$file") -gt $max_size ]; then
        echo "Skipping large file: $file"
        # 从暂存区移除大文件
        git rm --cached "$file"
        # 添加到 .gitignore 防止后续被跟踪
        echo "$file" >> .gitignore
    fi
done

# 提交 .gitignore 更新
git add .gitignore
git commit -m "Automatically ignore large files"


