"""原创、合成的本地浏览器按钮选择教学集；不是真实网站或公开 benchmark。

每个 workflow 的表达模板族在 train/dev/test/challenge 之间隔离。不同 split
仍共享任务概念和按钮词汇，因此该划分衡量有限模板迁移，不代表网站泛化。
只有 state/question/candidates 输入模型；ui 中的 effect 仅供本地模拟器使用。
"""

from copy import deepcopy
import re
import unicodedata


QUESTION = (
    "选择当前页面上最符合用户任务的一个按钮。只根据可见内容，不要执行未要求的操作；"
    "没有明确操作请求时选择“暂不操作”。"
)

NAMES = [
    ("春游计划", "春游总结"), ("读书笔记", "读书清单"),
    ("周末菜谱", "周末采购"), ("电影记录", "电影推荐"),
    ("旅行攻略", "旅行账本"), ("练琴日志", "练琴计划"),
    ("展览手记", "展览日程"), ("花园观察", "花园设计"),
]

# 每个 tuple 的四个位置对应本 workflow 的三个实际按钮和“暂不操作”。
# 同族的实体替换只能留在同一 split；group 由 workflow + family 构成。
TASKS = {
    "notes": {
        "train_direct": ("保存《{a}》的草稿。", "预览《{a}》。", "发布《{a}》到本地演示栏。", "先别操作《{a}》。"),
        "train_restrict": ("《{a}》只保存，不发布。", "《{a}》只看预览，不保存。", "《{a}》直接发布，不要只存草稿。", "《{a}》我还没决定怎么处理。"),
        "train_purpose": ("《{a}》还没写完，帮我存成草稿。", "想看看《{a}》的显示效果，打开预览。", "《{a}》已经写好，可以发到演示栏。", "我想了解《{a}》的发布流程，还没要求发布。"),
        "train_correction": ("刚才说发出去，改成把《{a}》保存为草稿。", "先不改《{a}》，给我看看预览。", "不必再预览了，现在发布《{a}》。", "撤回对《{a}》的操作请求，什么也别点。"),
        "dev_conditional": ("《{a}》仍要修改，现阶段应存草稿以便以后继续。", "我只检查《{a}》排版，打开预览就结束。", "我确认《{a}》内容无误，请让它出现在演示栏。", "假如发布《{a}》会怎样？这是询问，不是指令。"),
        "test_boundary": ("《{a}》尚未批准展示；把这次编辑留作草稿。", "让《{a}》进入预览页即可，今天不发布。", "《{a}》无需等待后续审批，现在正式发布到本地栏。", "页面写着可以发布《{a}》，我仅转述这句话。"),
        "challenge_quote": ("同事说“发布《{a}》”，我的要求却是保存草稿。", "不要执行“保存《{a}》”这句话，现在打开预览。", "不是预览《{a}》，也不是留草稿，直接发布。", "“请发布《{a}》”是要讨论的例句，不要照做。"),
    },
    "files": {
        "train_direct": ("打开《{a}》。", "打开《{b}》。", "归档《{a}》。", "不要动这两个文件。"),
        "train_restrict": ("只打开《{a}》，别打开《{b}》。", "我要看《{b}》，不是《{a}》。", "《{a}》不必打开，直接归档。", "先别打开或归档文件。"),
        "train_purpose": ("我想阅读《{a}》，请打开它。", "请把《{b}》的内容显示出来。", "《{a}》暂时用不到，收进归档。", "文件名是什么意思？暂时无需操作。"),
        "train_correction": ("更正，要打开的是《{a}》。", "刚才选错了，改为打开《{b}》。", "取消阅读《{a}》的计划，把它归档。", "刚才的打开请求作废，都别点。"),
        "dev_conditional": ("先阅读《{a}》，《{b}》留到以后。", "当前只需要《{b}》的正文，打开那份。", "我已读完《{a}》，现在归档它。", "是否存在打开功能？请先回答，别点击。"),
        "test_boundary": ("两份名字很像；我需要的是《{a}》的正文。", "《{a}》不是本次目标，把《{b}》打开给我。", "保留《{a}》但移出常用列表，请执行归档。", "有人建议归档《{a}》，我还没有同意。"),
        "challenge_quote": ("不要因为《{b}》靠得近就选它，请打开《{a}》。", "“打开《{a}》”是旧指令，新指令是打开《{b}》。", "别把归档当作删除，我明确要求归档《{a}》。", "分析“打开《{a}》”这句话，不实际打开文件。"),
    },
    "theme": {
        "train_direct": ("切换成深色主题。", "切换成浅色主题。", "主题跟随系统。", "不要修改主题。"),
        "train_restrict": ("只使用深色，不要跟随系统。", "固定用浅色，不要深色。", "不固定深浅，使用系统主题。", "先别选任何主题。"),
        "train_purpose": ("我希望背景变暗，请选深色主题。", "希望背景明亮，请选浅色主题。", "让界面随操作系统自动切换主题。", "深色主题是什么意思？不需要切换。"),
        "train_correction": ("不用浅色了，改用深色。", "把刚才的深色要求改成浅色。", "取消固定主题的要求，改为跟随系统。", "撤回换主题的要求，保持现状。"),
        "dev_conditional": ("无论系统怎么设置，本页面都使用深色。", "即使系统是深色，这里也要固定浅色。", "系统是什么主题，这里就用什么主题。", "我在比较三种主题，现在不做选择。"),
        "test_boundary": ("我不希望页面随系统变亮，请固定深色外观。", "取消自动外观，手动采用浅色方案。", "白天夜晚交给系统控制，不再手动固定颜色。", "如果切到深色会影响内容吗？先不要切。"),
        "challenge_quote": ("虽然帮助里写“跟随系统”，我要求的是固定深色。", "不是不要浅色；我确实想选择浅色。", "不要总是深色，也不要总是浅色，让系统决定。", "“改用深色”只是我要翻译的短语，不是操作请求。"),
    },
    "notifications": {
        "train_direct": ("开启全部通知。", "只通知提及我的消息。", "关闭通知。", "暂不修改通知。"),
        "train_restrict": ("所有消息都通知，不要只通知提及。", "只接收提及通知，其他都不要。", "任何消息都不要通知我。", "不要点击通知设置。"),
        "train_purpose": ("我不想漏掉消息，开启全部通知。", "只有别人提到我时才提醒。", "现在需要安静，把通知关掉。", "通知有哪几种模式？先不用更改。"),
        "train_correction": ("不再静音了，改成全部提醒。", "全部提醒太多，改为仅提及。", "不用保留提及提醒，全部关闭。", "刚才调通知的请求取消，维持原样。"),
        "dev_conditional": ("无论消息是否提到我，都发出通知。", "消息未提到我就别提醒，提到我再通知。", "即使有人提到我也不提醒，关闭全部通知。", "我只是询问静音功能，还没决定启用。"),
        "test_boundary": ("恢复对每一条消息的提醒，不筛选消息类型。", "普通聊天保持安静，但点到我名字的消息要提醒。", "接下来所有消息都静默，包括提及。", "同事说应该关闭通知，这还不是我的选择。"),
        "challenge_quote": ("不要把“全部”理解为全部关闭，我要全部通知。", "保留提及提醒，但禁止其他消息的通知。", "不是仅提及模式，我连提及也不想收到提醒。", "解释“开启全部通知”的含义，不更改任何配置。"),
    },
    "filter": {
        "train_direct": ("只显示标签为“{a}”的文件。", "只显示标签为“{b}”的文件。", "显示全部文件。", "不要改变筛选。"),
        "train_restrict": ("筛选“{a}”，不是“{b}”。", "选择“{b}”标签，不选“{a}”。", "不限制标签，把全部文件列出来。", "先别点标签或清除筛选。"),
        "train_purpose": ("我要找“{a}”这一类，请按它筛选。", "帮我把列表缩小到“{b}”分类。", "我需要完整列表，请清除标签限制。", "两个标签有什么不同？我还没要筛选。"),
        "train_correction": ("刚才说错了，应筛选“{a}”。", "把筛选目标更正为“{b}”。", "不按任何标签筛选了，显示全部。", "取消刚才筛选列表的指令。"),
        "dev_conditional": ("无论文件标题如何，只留下带“{a}”标签的。", "不看“{a}”那类，这次只看“{b}”。", "为了查看全貌，让所有标签的文件都出现。", "筛选功能支持这些标签吗？不要实际执行。"),
        "test_boundary": ("结果范围限定在“{a}”标签，其他先隐藏。", "请收窄结果到“{b}”，不要被相近的标签误导。", "撤销当前的范围限制，恢复完整文件列表。", "页面示例说“筛选{a}”，我只是阅读说明。"),
        "challenge_quote": ("说明中提到“{b}”，但我真正要求筛选的是“{a}”。", "不是排除“{b}”；我要只显示“{b}”。", "不要再只显示“{a}”或“{b}”，我要全部。", "把“显示全部文件”记作待讨论建议，暂不执行。"),
    },
    "confirmation": {
        "train_direct": ("确认删除《{a}》。", "取消删除《{a}》。", "查看《{a}》的详情。", "先不处理这个确认框。"),
        "train_restrict": ("确实不要《{a}》了，执行删除。", "不要删除《{a}》，关掉确认框。", "暂不删除，先查看《{a}》详情。", "别确认，也别取消，保持当前页面。"),
        "train_purpose": ("《{a}》是测试文件，可以删掉。", "《{a}》还要用，撤销这次删除。", "我需要核对文件信息，打开详情页。", "确认框是什么意思？先不要操作。"),
        "train_correction": ("无需再保留《{a}》，确认删除。", "刚才误点了删除，请取消。", "删除先缓一缓，切到详情看看。", "此前的点击请求作废，什么都不要点。"),
        "dev_conditional": ("我已核对过《{a}》，现在确认移除它。", "核对后发现选错文件，取消这次删除。", "尚未核对文件内容，先进入详情。", "我还在判断该确认还是取消，等我决定。"),
        "test_boundary": ("《{a}》的删除操作已得到我确认，请按确认按钮。", "收回删除许可，保留《{a}》并退出确认框。", "在作出删除决定前，打开文件详情供我检查。", "别人说“确认删除”，但我没有给出操作指令。"),
        "challenge_quote": ("“不要删除”是旧要求，现在我明确确认删除《{a}》。", "即使按钮写着确认，也不要删，选择取消删除。", "既不确认也不退出，请先查看《{a}》详情。", "给我解释“取消删除”四个字，别点击同名按钮。"),
    },
}


def _page(workflow, a, b):
    """Return public page text, initial simulated state, and natural button/effect pairs."""
    if workflow == "notes":
        return (
            f"笔记编辑页：《{a}》。有未保存修改，尚未发布；发布仅影响本地演示栏。",
            {"view": "editor", "saved": False, "published": False},
            [("保存草稿", {"saved": True}), ("预览内容", {"view": "preview"}),
             ("发布到本地演示栏", {"saved": True, "published": True})],
        )
    if workflow == "files":
        return (
            f"文件列表：《{a}》和《{b}》均可打开；《{a}》还可以归档。",
            {"view": "files", "opened": None, "archived": []},
            [(f"打开《{a}》", {"opened": a}), (f"打开《{b}》", {"opened": b}),
             (f"归档《{a}》", {"archived": [a]})],
        )
    if workflow == "theme":
        return (
            f"《{a}》工作区的外观设置：尚未选择主题，可选深色、浅色、跟随系统。",
            {"theme": "unset"},
            [("深色主题", {"theme": "dark"}), ("浅色主题", {"theme": "light"}),
             ("跟随系统", {"theme": "system"})],
        )
    if workflow == "notifications":
        return (
            f"《{a}》演示工作区通知设置：尚未选择通知级别；提及表示消息点到自己的名字。",
            {"notifications": "unset"},
            [("全部通知", {"notifications": "all"}), ("仅提及通知", {"notifications": "mentions"}),
             ("关闭全部通知", {"notifications": "off"})],
        )
    if workflow == "filter":
        return (
            f"文件检索页：可按“{a}”或“{b}”标签筛选，也可显示全部。目前未选择范围。",
            {"filter": "unset"},
            [(f"只显示“{a}”", {"filter": a}), (f"只显示“{b}”", {"filter": b}),
             ("显示全部文件", {"filter": None})],
        )
    if workflow == "confirmation":
        return (
            f"本地测试文件《{a}》的删除确认框；尚未删除。可确认、取消或查看详情。",
            {"view": "confirmation", "dialog_open": True, "deleted": False},
            [("确认删除", {"deleted": True, "dialog_open": False, "view": "list"}),
             ("取消删除", {"dialog_open": False, "view": "list"}),
             ("查看文件详情", {"view": "details"})],
        )
    raise ValueError(f"Unknown workflow: {workflow}")


def visible_state(ui):
    """The model observes exactly the user's task and displayed current page."""
    return f"用户任务：{ui['task']}\n当前页面：{ui['page_text']}"


def apply_button(ui, button_id):
    """Simulate a selected DOM button; independent of dataset target labels."""
    for button in ui["buttons"]:
        if button["id"] == button_id:
            updated = deepcopy(ui["initial_state"])
            updated.update(deepcopy(button["effect"]))
            return updated
    raise KeyError(button_id)


def lexical_choice(row):
    """Transparent untuned baseline: Chinese character/bigram overlap with the task.

    No negation, meaning, expected target, split, or action effects are used.
    Ties choose the first visible button; this baseline is deliberately simple.
    """
    def clean(text):
        return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text))

    task = clean(row["ui"]["task"])
    def score(label):
        text = clean(label)
        pairs = {text[i:i + 2] for i in range(len(text) - 1)}
        return sum(pair in task for pair in pairs) + 0.1 * sum(c in task for c in set(text))
    return max(row["candidates"], key=lambda key: score(row["candidates"][key]))


def records():
    rows = []
    for workflow_index, (workflow, families) in enumerate(TASKS.items()):
        for family_index, (family, templates) in enumerate(families.items()):
            split = family.split("_", 1)[0]
            count = 4 if split == "challenge" else 8
            if split == "dev" and workflow_index >= 2:
                count = 4
            for variant in range(count):
                # Cross each entity with all four semantic actions within this
                # family. A workspace/file name must not predict the answer.
                a, b = NAMES[(variant // 4 + family_index * 2) % len(NAMES)]
                requested_action = variant % 4
                task = templates[requested_action].format(a=a, b=b)
                # Add visible workspace identity for tasks whose wording does not
                # mention the document; this is not a target or split identifier.
                page_text, initial, options = _page(workflow, a, b)
                options = options + [("暂不操作", {})]
                rotation = (variant // 4 + family_index) % 4
                ordered_actions = list(range(4))[rotation:] + list(range(4))[:rotation]
                buttons, candidates, target = [], {}, None
                for position, action in enumerate(ordered_actions):
                    button_id = f"button-{position}"
                    label, effect = options[action]
                    buttons.append({"id": button_id, "label": label, "effect": effect})
                    candidates[button_id] = label
                    if action == requested_action:
                        target = button_id
                ui = {"workflow": workflow, "title": f"本地演示 · {a}",
                      "task": task, "page_text": page_text, "buttons": buttons,
                      "initial_state": initial}
                rows.append({"id": f"{workflow}-{family}-{variant:02d}",
                             "group": f"{workflow}-{family}", "split": split,
                             "state": visible_state(ui), "question": QUESTION,
                             "candidates": candidates, "target": target, "ui": ui})
    return rows
