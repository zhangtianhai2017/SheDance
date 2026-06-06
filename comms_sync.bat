@echo off
cd /d C:\work\2026\Claude\SheDance
echo Syncing comms (SHEDANCE) ...
git add comms/
git diff --cached --quiet || git commit -m "comms sync"
git pull --rebase --autostash
git push
echo.
echo Done. New messages in comms\msgs (to-SHEDANCE).
pause