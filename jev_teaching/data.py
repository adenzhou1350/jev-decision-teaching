"""原创手写中文教学数据；不是自然流量或生产验收集。"""

QUESTION = "根据用户当前明确意图选择动作；闲聊、引用、含糊或未要求操作时选 other。"
CANDIDATES = {
    "pause": "暂时暂停当前任务，保留进度，之后可以恢复",
    "cancel": "取消并结束当前任务，不再执行",
    "continue": "继续或恢复执行当前任务",
    "other": "没有明确的任务控制要求，不执行控制动作",
}

# 先写不同表达的分组，再固定 split；不要先扩写模板再随机拆散。
EXAMPLES = {
    "train": [
        ("暂停一下", "pause"), ("先停一会，等我回来", "pause"),
        ("暂时别做了，保留进度", "pause"), ("我接个电话，先暂停", "pause"),
        ("等一等，待会再继续", "pause"), ("先挂起这个任务", "pause"),
        ("取消这个任务", "cancel"), ("不用做了，结束吧", "cancel"),
        ("把现在的任务终止掉", "cancel"), ("这件事不需要了", "cancel"),
        ("撤销当前任务", "cancel"), ("别继续了，直接结束", "cancel"),
        ("继续执行", "continue"), ("恢复刚刚的任务", "continue"),
        ("接着做吧", "continue"), ("现在可以继续了", "continue"),
        ("我回来了，恢复运行", "continue"), ("不要停，继续干", "continue"),
        ("你好呀", "other"), ("今天真热", "other"),
        ("暂停是什么意思", "other"), ("你支持取消任务吗", "other"),
        ("讲个笑话", "other"), ("我在想该怎么办", "other"),
    ],
    "dev": [
        ("稍等，先别往下做", "pause"), ("临时停一下，我确认一下", "pause"),
        ("当前工作作废，不用再处理", "cancel"), ("彻底结束这次任务", "cancel"),
        ("接上之前的进度继续", "continue"), ("按原计划往下执行", "continue"),
        ("任务进度怎么样", "other"), ("我喜欢这首歌", "other"),
    ],
    "test": [
        ("先缓缓，过会再干", "pause"), ("手头的事先搁着，别丢进度", "pause"),
        ("停片刻等我的消息", "pause"),
        ("这单不做了，直接撤掉", "cancel"), ("到此为止，后面全都不用做", "cancel"),
        ("结束目前这项工作，不再恢复", "cancel"),
        ("从上次停下的地方接着干", "continue"), ("解除暂停，往后执行", "continue"),
        ("不用等我了，现在继续", "continue"),
        ("还需要多少时间", "other"), ("我不太明白你的意思", "other"),
        ("下午好，辛苦了", "other"),
    ],
    "challenge": [
        ("不要取消，只是暂停", "pause"), ("不是结束，先等五分钟", "pause"),
        ("暂停，不要把任务删掉", "pause"),
        ("不用暂停了，彻底取消", "cancel"), ("别恢复，这项任务已经作废", "cancel"),
        ("这次不是临时停，是永久结束", "cancel"),
        ("不要暂停，继续执行", "continue"), ("取消刚才的暂停要求，接着做", "continue"),
        ("别结束，恢复运行", "continue"),
        ("他说了“取消任务”这四个字", "other"),
        ("如果我说暂停，你会怎么做", "other"), ("算了……我还没决定", "other"),
    ],
}


def records():
    return [
        {"id": f"{split}-{i:02}", "group": f"{split}-{i:02}", "split": split,
         "state": text, "question": QUESTION, "candidates": dict(CANDIDATES), "target": target}
        for split, examples in EXAMPLES.items()
        for i, (text, target) in enumerate(examples)
    ]
