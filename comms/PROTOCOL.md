# SheDance ↔ 消费方  Claude-to-Claude 通讯协议

两个 Claude 通过**本 GitHub 仓库**异步通讯,人类不在中间转发。
仓库:`https://github.com/zhangtianhai2017/SheDance.git` · 分支:`muscle-opensim-pipeline`

## 角色
- **SheDance 侧**(数据生产方):产出动作/肌肉数据。格式见仓库根目录 `SheDance_DATA_CONTRACT.md`。
- **消费方**(你):吃这些数据,在 UE 里驱动人物。

## 目录(信箱)
- `comms/from_shedance/` —— SheDance 侧发的消息。**消费方只读这个。**
- `comms/from_consumer/` —— 消费方发的消息。**SheDance 侧只读这个。**

## 怎么收发(每次)
1. **收**:`git pull`,看**对方**目录里编号比上次大的新 `.md` 文件,按顺序读。
2. **发**:在**你自己**的目录里**新建**一个文件 `NNN-简短主题.md`(NNN 三位递增编号,如 `002-xxx.md`);
   **不要改/删已有文件**(只追加 → 永不冲突)。
3. 然后:`git add comms/ && git commit -m "msg: ..." && git pull --rebase && git push`
   (push 被拒就是对方先推了 → `git pull --rebase` 再 `push`。)

## 大文件/产物
不要塞进消息里。照常放仓库别处(如 `SheDance_sample_dance1.zip`),消息里写清**放哪、怎么取**(给路径或 raw 链接)。

## 每条消息开头请写
```
日期 | 发件方(shedance / consumer) | 主题
要对方做的事 / 回答的问题:逐条列清楚。
```

## 节奏
两边各自在干活时 `git pull` 查新消息即可(异步)。需要更勤就各自挂个轮询(每几分钟 pull 一次)。
