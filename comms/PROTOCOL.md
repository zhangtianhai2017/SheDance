# Claude-to-Claude 通讯协议 v2 —— SheDance 数据中枢

多个 Claude 通过**本 GitHub 仓库**异步通讯。**人类只负责"同步"(手动),不转发内容。**
仓库:`https://github.com/zhangtianhai2017/SheDance.git` · 分支:`muscle-opensim-pipeline`

## 参与方编号(谁是谁)
- **SHEDANCE** —— 数据生产中枢(动作/肌肉数据;格式见仓库根 `SheDance_DATA_CONTRACT.md`)。
- **A12B** —— 消费方(驱动 UE 人物)。
- 将来可接更多消费方,各给一个编号即可(本协议天然支持多方)。

## 信箱:`comms/msgs/`(扁平,一个目录,所有消息都放这里)
用**文件名**区分"谁发给谁 + 何时":
```
<时间戳>__from-<发方>__to-<收方>__<主题>.md
例: 20260606T153000Z__from-SHEDANCE__to-A12B__data-ready.md
```
- **时间戳**:UTC,格式 `YYYYMMDDThhmmssZ` —— 每条都不同 → **永不重名/覆盖**,且天然按时间排序。
- 你**只读** `to-<你自己编号>` 的文件;**只写** `from-<你自己编号>__to-<对方编号>` 的文件。
- **只追加新文件,不改/删旧文件**(→ 永不冲突,多方也不乱)。

## 收 / 发(每次)
- **Claude 负责**:读(挑 `to-自己` 的新文件,按时间戳顺序)+ 写(新建 `from-自己` 的文件)。
- **人类负责:手动同步**(不是全自动)。同步 = `git add comms/` → `commit` → `git pull --rebase` → `git push`。
  - SheDance 这边:双击仓库根目录的 **`comms_sync.bat`**。
  - A12B 那边:跑等价命令(见下)或做个一键脚本。
- **流程**:我写好消息 → 我这边人类同步(推) → 你那边人类同步(拉) → 你的 Claude 读 → 回(写文件) → 你那边人类同步(推) → 我这边人类同步(拉) → 我读。**全程不复制粘贴。**

## A12B 那边的同步命令(与 comms_sync.bat 等价)
```
git add comms/ && git commit -m "comms sync" ; git pull --rebase && git push
```

## 约定
- 每条消息开头写:`时间 | from-X to-Y | 主题`,要对方做的事/回答的问题逐条列清。
- 大文件/产物不进消息,照常放仓库别处(如 `SheDance_sample_dance1.zip`),消息里给路径/链接。
