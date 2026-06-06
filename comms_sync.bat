@echo off
chcp 65001 >nul
cd /d C:\work\2026\Claude\SheDance
echo ===== 同步 comms (SHEDANCE) =====
git add comms/
git diff --cached --quiet || git commit -m "comms sync"
echo --- 拉取对方消息 ---
git pull --rebase
echo --- 推送我方消息 ---
git push
echo.
echo ===== 完成。看 comms\msgs\ 里 to-SHEDANCE 的新消息 =====
pause
