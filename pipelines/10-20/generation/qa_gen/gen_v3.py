"""7 类题型的出题线（Omni 版 10-16：逐秒上下文带音频转写，要点分 [音频]/[视觉]，每题至少一条音频要点；其余沿用 v3 协议，10-11 用户认可的"每类一个具体例子"；10-13 v3b 加硬规则 + t0 闸门）：
temporal_update / selective_attention / constraint_update / context_correction / task_resume / monitor_cancel / proactive_clarify。
historical_reference 见 gen_hr.py，proactive_guidance 见 gen_pg.py。

每类的独有考点和例子见 SPECS[*]['rule']；每类有自己的分支/槽位结构和看帧核验条件，不再共用一个模板。
流程（复用 refresh_hr 的调用/取帧/标注函数）：
  候选窗口（每类关键词给 50s 窗口打分，每视频取前 N 个不重叠窗口）
  → screen：只看逐秒摘要+OCR，选时刻（代码检查间隔，最多重选 3 次）
  → author：看各时刻真实帧出题 → 代码检查（结构、时刻、台词相同/不同、要点条数、答案差异）
  → 泄露核查 → 看帧核验（台词成立、要点成立、本类考点条件）→ 不过则带累积原因重写，最多 3 次。
用法: python3 gen_v3.py --batch 50|316 --caps temporal_update,... --per-video 1 [--videos 10] [--budget 2] [--workers 10]
输出: runs/<batch>/runs_v3/<cap>_<src>_<start>/{log.json, task.json, candidate_task.json, screen*.response.txt, a*_*.response.txt}
     log.json 里 status ∈ pass / needs_revision / infeasible / screen_rejected / error。
"""
import argparse
import json
import re
import sys
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as R  # noqa: E402
import config as CFG  # noqa: E402
LEAK_SYSTEM = R.LEAK_SYSTEM_FACTS

OUT = None  # runs/<batch>/runs_v3，由 main() 按 --batch 设定
windows_g = {}  # 逐秒窗口（verify 里取转写用），main() 里赋值
WIN = 50  # 候选窗口长度（秒）
WATERMARK = re.compile(r'bilibili|@|水印|抖音|小红书|UID', re.I)

# ---------------------------------------------------------------- 每类规格
# events: (event_id, 时刻名, op)；slots: (slot_id, 时刻名, 动作, 触发方式 user|visual)
# same_text: 两个分支必须相同的事件；diff_text: 必须不同的事件；distinct: 必须答案不同的 (分支, 槽位) 对
SPECS = {
    'temporal_update': {
        'family': 'v3_state_change',
        'rule': '考点：同一个对象的状态在会话中发生了变化，模型要回答**最新状态**，不能沿用旧记忆。'
                '例子：讲解员先在左边讲青铜鼎，20 秒后走到右边指着另一件，用户问"她现在在讲哪件？"。'
                '设计：t0 用户让模型留意对象 X（人/动物/可动的展项/屏幕内容等会变化的对象）；t1 时 X 处于状态 S1；t2 时 X 变成明显不同的状态 S2（位置、动作、正在看/讲的展品、屏幕画面等），X 在 t1 和 t2 都在画面里。'
                '**硬性要求：用户在 t0 开口时 X 必须已经清楚地在画面里（t0 前 1 秒就已可见、可辨认），不能选尚未出现、刚入画一角或看不清的对象；如果对象是稍后才出现，就把 t0 往后挪到它清楚可见之后。**'
                '分支 ask_twice：t1 和 t2 各问一次同一句"它/他/她现在在干嘛/怎么样了？"，t1 答 S1，t2 答 S2（forbidden 写 S1）；分支 ask_late：只在 t2 问同一句，答 S2。',
        'keywords': r'讲解|观众|游客|孩子|男子|女子|人员|走|转身|指向|举起|抬|蹲|坐|拿|屏幕|画面切换|播放',
        'times': ['t0', 't1', 't2'], 'cons': [('t0', 't1', 3, 20), ('t1', 't2', 8, 30), ('t0', 't2', 12, 45)],
        'branches': {
            'ask_twice': {'events': [('ue0', 't0', 'bind'), ('ue1', 't1', 'query'), ('ue2', 't2', 'query')],
                          'slots': [('sl_early', 't1', 'SPEAK', 'user'), ('sl_late', 't2', 'SPEAK', 'user')]},
            'ask_late': {'events': [('ue0', 't0', 'bind'), ('ue2', 't2', 'query')],
                         'slots': [('sl_late', 't2', 'SPEAK', 'user')]},
        },
        'same_text': ['ue0', 'ue2'], 'diff_text': [], 'same_text_pairs': [('ask_twice', 'ue1', 'ask_twice', 'ue2')],
        'distinct': [('ask_twice', 'sl_early', 'ask_twice', 'sl_late')],
        'diagnostic': ['sl_late'],
        'extra_frames': [('t1', 't2', 2)],
        'checks': {'visible_at_t0': 'X 在 t0 组帧里（包括 t0−1s 那一帧）已经清楚可见、可辨认，不是尚未出现、刚露出一角或模糊不清',
                   'same_target': 'X 在 t1 组和 t2 组帧里都可见，而且是同一个对象（不是换了一个长得像的）',
                   'state_changed': 'X 在 t2 的状态与 t1 的状态确实明显不同（不是镜头角度变化造成的错觉）'},
    },
    'selective_attention': {
        'family': 'v3_distractor_filter',
        'rule': '考点：有显眼干扰时只报告用户指定的目标。'
                '例子："只告诉我那个穿红衣服的小孩在干嘛"，期间旅行团走过、讲解员大声说话，回答不能提这些。'
                '设计：t0 用户说"接下来只跟我说 X 的情况，别的不用管"。X 必须是 t0 到 t2 一直在画面里的对象：可以是人、一群人、屏幕/投影内容，也可以是镜头一直对着的展品/展柜；'
                '**硬性要求：用户在 t0 开口时 X 必须已经清楚地在画面里（t0 前 1 秒就已可见、可辨认），不能选尚未出现、刚入画一角或看不清的对象；对象稍后才出现的话就把 t0 往后挪。**'
                'td 时画面里出现与 X 无关的显眼干扰 Z（有人走入或穿过画面、旁边观众的明显动作、新物体进入画面等），t2 用户提问。'
                '分支 focus：t2 问 X 现在的情况（人："他现在在干嘛？"；展品/屏幕："它现在怎么样/上面显示什么？"），只答 X 在 t2 的情况，forbidden 写 Z 的情况；分支 ask_distractor：t2 用户主动问 Z（"刚才从旁边过去的那些人是干嘛的？"），此时应答 Z。'
                '两个分支在 td 都有一个 WAIT 槽位：干扰出现时不要主动播报。',
        'keywords': r'走入|走进|经过|穿过|走过|进入画面|入画|人群|一群|旅行团|多名|讲解员|孩子|游客',
        'times': ['t0', 'td', 't2'], 'cons': [('t0', 'td', 3, 20), ('td', 't2', 2, 15), ('t0', 't2', 8, 30)],
        'branches': {
            'focus': {'events': [('ue0', 't0', 'bind'), ('ue1', 't2', 'query')],
                      'slots': [('sl_quiet', 'td', 'WAIT', 'visual'), ('sl_main', 't2', 'SPEAK', 'user')]},
            'ask_distractor': {'events': [('ue0', 't0', 'bind'), ('ue1', 't2', 'query_distractor')],
                               'slots': [('sl_quiet', 'td', 'WAIT', 'visual'), ('sl_main', 't2', 'SPEAK', 'user')]},
        },
        'same_text': ['ue0'], 'diff_text': ['ue1'], 'distinct': [('focus', 'sl_main', 'ask_distractor', 'sl_main')],
        'diagnostic': ['sl_quiet', 'sl_main'],
        'extra_frames': [('td', 't2', 1)],
        'checks': {'visible_at_t0': 'X 在 t0 组帧里（包括 t0−1s 那一帧）已经清楚可见、可辨认，不是尚未出现、刚露出一角或模糊不清',
                   'distractor_salient': 'td 组帧里确实出现了显眼、与 X 无关的干扰 Z（t0 组帧里还没有）',
                   'target_visible': 'X 在 t2 组帧里可见，focus 分支的答案说的是 X 而不是 Z'},
    },
    'constraint_update': {
        'family': 'v3_constraint_update',
        'rule': '考点：同一个场景/同一个信息源，换了筛选条件，答案就跟着变。'
                '例子："只读展签上的英文" → "改成只告诉我年代"，同一块展签给出不同答案。'
                '设计：t0 用户对画面里同一个信息源（展签、展板、屏幕、一组展品、一群人）设定筛选条件 C1；t1 分支 keep 确认 C1，分支 update 改成 C2；t2 问"上面写了什么/现在符合要求的是什么？"。'
                'C1 和 C2 作用在**同一个信息源**上，在 t2 帧里都能给出**非空、具体**的答案（不能是"没有/找不到/没列出"），两个答案不同、可核对；forbidden 写按另一个条件会给出的答案。',
        'keywords': r'展签|说明牌|展板|文字|字样|英文|标题|屏幕|标牌|题签',
        'ocr_weight': 3,
        'times': ['t0', 't1', 't2'], 'cons': [('t0', 't1', 4, 20), ('t1', 't2', 6, 25), ('t0', 't2', 12, 40)],
        'branches': {
            'keep_constraint': {'events': [('ue0', 't0', 'set_constraint'), ('ue1', 't1', 'keep_constraint'), ('ue2', 't2', 'query')],
                                'slots': [('sl_main', 't2', 'SPEAK', 'user')]},
            'update_constraint': {'events': [('ue0', 't0', 'set_constraint'), ('ue1', 't1', 'update_constraint'), ('ue2', 't2', 'query')],
                                  'slots': [('sl_main', 't2', 'SPEAK', 'user')]},
        },
        'same_text': ['ue0', 'ue2'], 'diff_text': ['ue1'], 'distinct': [('keep_constraint', 'sl_main', 'update_constraint', 'sl_main')],
        'diagnostic': ['sl_main'],
        'extra_frames': [],
        'checks': {'same_source': '两个约束作用在 t2 帧里同一个信息源上（同一块展签/展板/同一组对象）',
                   'constraints_differ': '按 C1 和 C2 在 t2 帧里确实得到不同的答案，且两者都是非空、具体、能从帧里读出/看出的内容（任一答案是"没有/找不到"即为 false）'},
    },
    'context_correction': {
        'family': 'v3_user_self_correction',
        'rule': '考点：用户自己先说错了，随后根据画面证据更正，模型要放弃旧认知、按更正后的信息回答。'
                '例子："这是唐代的吧"……"哦不对，展签写的是宋代"，之后问相关问题要按宋代回答。'
                '设计：t0 用户对画面里的对象说了一个**错误**判断 W（朝代、名称、作者、哪一件是某物、颜色材质等），画面里有证据（展签文字、近景）能看出正确信息 T；'
                '分支 correction：t1 用户自己更正为 T（"哦不对，牌子上写的是……"），t2 用户问一个答案取决于该判断的问题，模型按 T 回答，forbidden 写沿用 W 的说法；'
                '分支 no_correction：t1 用户说一句无关的话（不更正），t2 问同一句，模型应指出画面证据显示是 T、不是 W（主动纠正用户），forbidden 写附和 W。'
                't2 的答案必须能在 t1/t2 帧里核对（例如展签上与 T 相关的其他文字、该对象的外观）。',
        'keywords': r'展签|说明牌|题签|文字|字样|标题|名称|年代|朝代|作者|代|窑',
        'ocr_weight': 3,
        'times': ['t0', 't1', 't2'], 'cons': [('t0', 't1', 4, 20), ('t1', 't2', 6, 25), ('t0', 't2', 12, 40)],
        'branches': {
            'correction': {'events': [('ue0', 't0', 'claim'), ('ue1', 't1', 'correct'), ('ue2', 't2', 'query')],
                           'slots': [('sl_main', 't2', 'SPEAK', 'user')]},
            'no_correction': {'events': [('ue0', 't0', 'claim'), ('ue1', 't1', 'chat'), ('ue2', 't2', 'query')],
                              'slots': [('sl_main', 't2', 'SPEAK', 'user')]},
        },
        'same_text': ['ue0', 'ue2'], 'diff_text': ['ue1'], 'distinct': [],
        'diagnostic': ['sl_main'],
        'extra_frames': [],
        'checks': {'claim_wrong': 't0 台词里用户的判断 W 与画面证据不符，画面证据（展签/近景）显示的是 T',
                   'evidence_visible': '证据 T 在 t1 或 t2 组帧里清楚可读/可见',
                   'branches_behave': 'correction 分支答案按 T 回答且不提 W；no_correction 分支答案明确指出是 T 不是 W'},
    },
    'task_resume': {
        'family': 'v3_resume_progress',
        'rule': '考点：被打断的多步（累积型）任务，恢复后要接着原来的进度做。'
                '例子："帮我数这面墙有几幅画" → 插问别的 → "继续数，刚才数到几了？"。'
                '设计：t0 用户布置一个随镜头移动逐步累积的任务（数经过的画/展柜/雕像，按顺序记下看到的展签标题，比较先后两件展品）；'
                't1 用户插问一个无关的、t1 画面里能直接回答的问题；t2 用户恢复。'
                '分支 resume：t2 说"好，继续刚才的——到现在一共几个/都有哪些？"，答案要把 t0–t1 的进度和 t1–t2 新看到的合起来；'
                '分支 restart：t1 插问同一句后再加一句"刚才那个先不算了"，t2 说"从现在开始重新数/记，画面里现在有几个/有哪些？"，答案只算 t2 当前画面。'
                '两个分支的 t2 答案必须不同。resume 的答案必须**同时包含插问前（t0–t1）看到的和插问后（t1–t2）新看到的**条目，插问后没有新条目的时刻不能选；'
                'restart 必须重新做**同一个**任务（任务是记名字就重新记名字，是数个数就重新数），t2 句不点名对象。数量要能从给你的帧里数清；数不清就换成"记下展签标题/展品名称"之类的任务。',
        'keywords': r'一排|多幅|多件|依次|并排|陈列|若干|几件|两件|三件|一组|成排|多个|多尊|平移',
        'times': ['t0', 't1', 't2'], 'cons': [('t0', 't1', 6, 25), ('t1', 't2', 5, 20), ('t0', 't2', 15, 45)],
        'branches': {
            'resume': {'events': [('ue0', 't0', 'bind_task'), ('ue1', 't1', 'side_question'), ('ue2', 't2', 'resume')],
                       'slots': [('sl_side', 't1', 'SPEAK', 'user'), ('sl_main', 't2', 'SPEAK', 'user')]},
            'restart': {'events': [('ue0', 't0', 'bind_task'), ('ue1', 't1', 'side_question_and_cancel'), ('ue2', 't2', 'restart')],
                        'slots': [('sl_side', 't1', 'SPEAK', 'user'), ('sl_main', 't2', 'SPEAK', 'user')]},
        },
        'same_text': ['ue0'], 'diff_text': ['ue2'], 'distinct': [('resume', 'sl_main', 'restart', 'sl_main')],
        'diagnostic': ['sl_main'],
        'extra_frames': [('t0', 't2', 6)],
        'checks': {'progress_correct': 'resume 分支答案与 t0 到 t2 的帧里累积看到的内容一致（数量/列表正确）',
                   'progress_spans': 'resume 答案里既有 t1 之前看到的条目，也有 t1 之后才新出现的条目（插问后任务确实有进展）',
                   'restart_correct': 'restart 分支答案只对应 t2 当前画面，且与 resume 不同',
                   'side_ok': '插问在 t1 画面里能回答，和累积任务无关'},
    },
    'monitor_cancel': {
        'family': 'v3_monitor_cancel',
        'rule': '考点：用户让模型盯着某个条件、出现时提醒；取消后即使条件出现也要保持沉默。'
                '例子："看到出口标志提醒我" → "算了不用了" → 标志出现，不能说话。'
                '设计：t0 用户说"看到 X 就提醒我"（X 在 t0 到 tt 之前**都不在画面里**，tt 时第一次清楚出现：标志、某类展品、某个人走入、某段文字等）；'
                't1 分支 keep 确认继续盯（"好，继续帮我留意"），分支 cancel 取消（"算了，不用提醒了"）；'
                'tt 时 keep 分支应主动提醒（SPEAK，说清看到了什么、在哪），cancel 分支保持沉默（WAIT）。',
        'keywords': r'进入画面|入画|出现|走入|走进|露出|显现|映入|开始出现',
        'times': ['t0', 't1', 'tt'], 'cons': [('t0', 't1', 3, 15), ('t1', 'tt', 6, 30), ('t0', 'tt', 12, 45)],
        'branches': {
            'keep_monitor': {'events': [('ue0', 't0', 'bind'), ('ue1', 't1', 'keep')],
                             'slots': [('sl_alert', 'tt', 'SPEAK', 'visual')]},
            'cancel_monitor': {'events': [('ue0', 't0', 'bind'), ('ue1', 't1', 'cancel')],
                               'slots': [('sl_alert', 'tt', 'WAIT', 'visual')]},
        },
        'same_text': ['ue0'], 'diff_text': ['ue1'], 'distinct': [],
        'diagnostic': ['sl_alert'],
        'extra_frames': [('t0', 'tt', 4)],
        'checks': {'absent_before': 'X 在 tt 之前的所有帧（A、B、M 组）里都看不到',
                   'appears_at_tt': 'X 在 tt 组帧里清楚出现，keep 分支提醒的内容与画面一致'},
    },
    'proactive_clarify': {
        'family': 'v3_ambiguity_clarify',
        'rule': '考点：用户的指代有歧义时，模型先问清楚，不要猜。'
                '例子：画面里有三个花瓶，用户问"这个是什么年代的？"，应该先问是哪一个。'
                '设计：t0 画面里同时可见 ≥2 个同类候选（花瓶、画、雕像、展签、人…），彼此能用位置/外观区分；'
                '分支 ambiguous：t0 用户问一个关于其中一件的问题（答案能从画面看出：颜色、图案、上面的字、在做什么），模型应澄清（CLARIFY，列出候选让用户选），'
                '**硬性要求：ambiguous 分支的问句必须点出物品的类别名称（如"这幅画是什么年代的""这里面的服饰是什么颜色的""这两个陶瓷哪个更大"），歧义只能出在"指的是哪一个"上；不能只说"这个是什么颜色的""这是什么"这种连物品是什么都不说的问法。**'
                't1 用户说明是哪一个（"左边那个"），模型回答；分支 specific：t0 用户直接说清是哪一个再问同样的问题，模型应直接回答（SPEAK，不要多余澄清）。'
                '两个分支最终的答案是同一个对象的同一个信息。',
        'keywords': r'两件|三件|两个|三个|两尊|三尊|一排|并排|多件|几件|两幅|三幅|两块|多个|成对|一对|并列',
        'times': ['t0', 't1'], 'cons': [('t0', 't1', 3, 12)],
        'branches': {
            'ambiguous': {'events': [('ue0', 't0', 'ambiguous_query'), ('ue1', 't1', 'clarify_reply')],
                          'slots': [('sl_clarify', 't0', 'CLARIFY', 'user'), ('sl_answer', 't1', 'SPEAK', 'user')]},
            'specific': {'events': [('ue0', 't0', 'specific_query')],
                         'slots': [('sl_direct', 't0', 'SPEAK', 'user')]},
        },
        'same_text': [], 'diff_text': ['ue0'], 'distinct': [],
        'diagnostic': ['sl_clarify', 'sl_direct'],
        'extra_frames': [],
        'checks': {'multiple_candidates': 't0 组帧里确实同时可见 ≥2 个同类候选，ambiguous 分支的问句在当时确实指代不明',
                   'names_category': 'ambiguous 分支的问句点出了物品的类别名称（画/佛像/服饰/陶瓷/展签…），而不是只用"这个/那个/这是"指物',
                   'answer_correct': 'sl_answer 与 sl_direct 回答的是同一个对象，内容在帧里成立'},
    },
}

AUDIO_HINT = {
    'temporal_update': '音频用法：对象的“状态”可以是讲解员/主播**正在讲的内容**（S1 时在讲 A 展品/朝代，S2 时已转到 B），S1/S2 的要点可以引用她说的话；也可以是画面状态变化加上她当时说的话作为 [音频] 要点。',
    'selective_attention': '音频用法：干扰可以是另一个人的说话、广播或背景音乐，模型只关注目标时要忽略无关的声音；提问可以问“她刚才说的是什么”，答案要点引用目标对象说的话。',
    'constraint_update': '音频用法：约束可以针对音频，例如“以后只把讲解员提到的年代告诉我”“她说的数字都帮我记着”，t2 的回答要点引用在约束下听到的内容。',
    'context_correction': '音频用法：用户在 t0 的说法可以和讲解员/广播说的内容矛盾（用户说错了年代或名称），t1 用户纠正或不纠正，t2 回答以纠正后或听到的事实为准，并把听到的原话作为 [音频] 要点。',
    'task_resume': '音频用法：任务可以是“把讲解员接下来讲的要点记下来/等她说到价格就告诉我”，插问后恢复时要复述这期间听到的内容作为 [音频] 要点。',
    'monitor_cancel': '音频用法：监测的触发条件可以是声音事件（广播响起、讲解员说到某个词、音乐开始），keep 分支在触发时说出听到的内容，cancel 分支保持沉默。',
    'proactive_clarify': '音频用法：歧义可以来自音频——讲解员刚提到两件展品，用户问“她刚说的那个是哪年的”，模型应先澄清是哪一件；specific 分支点名其中一件直接回答，要点引用她说的话。',
}
for _cap, _spec in SPECS.items():
    _spec['rule'] = _spec['rule'] + '\n' + AUDIO_HINT.get(_cap, '') + '\n' + R.OMNI_RULE

SCREEN_SYSTEM = """你在为第一视角博物馆流式交互 benchmark 挑选出题时刻（题型 {cap}）。给你一段 {win} 秒视频的逐秒摘要（含 🔊 音频）、OCR 和这段的音频转写。
题型要求：{rule}
需要选出这些时刻（秒，可带一位小数）：{times}；间隔要求：{cons}。所用对象在相应时刻必须在画面中（看摘要确认）。
""" + R.TIMING_RULE + """
只有当这段视频真的能按题型设计出题时才 ok=true；否则 ok=false 并说明原因。
只输出 JSON：{{"ok": bool, "reason": "...", "times": {{{time_keys}}}, "plan": "一句话设计，点名用到的对象", "audio_use": "打算把转写里的哪句话/哪个声音事件作为音频要点（引用原句和时间）"}}"""

AUTHOR_SYSTEM = """你在为第一视角博物馆流式交互 benchmark 出题（题型 {cap}）。用户戴着摄像头参观，模型实时看画面、记住对话状态。
题型要求：{rule}
已选定时刻：{times}。按下面的骨架写题，事件和槽位的 id、时刻、op、动作都不能改：
{skeleton}
文本要求：{text_rules}
每个 SPEAK 槽位写 target_text（该时刻的自然口语标准回答）、key_facts 2–4 条（每条一个短要点，以“[音频]”或“[视觉]”开头：[视觉] 要点要能在该时刻帧里核对，[音频] 要点要能在音频转写里逐字核对且说出时间早于该槽位）、audio_refs（该槽位引用的转写原句，含 start/end/text；没有就空列表）、forbidden_claims 2–3 条；
CLARIFY 槽位写 target_text（澄清句，点出候选），key_facts 写 1–2 条（候选是什么），forbidden_claims 写直接猜一个答案；WAIT 槽位 target_text 为空，forbidden_claims 写不该说的话。
台词只用位置/指示词/外观最少的说法指物体，不能提前说出评分时刻的答案；看不清的不要写。做不出来就 feasible=false 并说明原因。
只输出 JSON：{{"feasible": bool, "infeasible_reason": "",
 "branches": [{{"branch_id": "...", "user_events": [{{"event_id": "...", "time_s": 数字, "text": "...", "state_op": {{"op": "...", "id": "...", "target": "内部描述"}}}}],
   "slots": [{{"slot_id": "...", "target_text": "...", "key_facts": ["[音频] ...", "[视觉] ..."], "audio_refs": [{{"start": 秒, "end": 秒, "text": "转写原句"}}], "forbidden_claims": ["..."]}}]}}]}}"""

VERIFY_SYSTEM = """你是严格的全模态核验员。给你一道题（题型 {cap}）、这段视频的音频转写（带时间）和真实帧，帧按时刻分组标注（组名就是时刻名，M 是中间时刻）。题型要求：{rule}
逐项判断：
1. lines_ok：每句用户台词在它说出的时刻画面里成立（所指对象可见、指代能对上）；
2. 每个槽位的 key_facts 逐条判断：以 [视觉] 开头的看该槽位时刻的帧；以 [音频] 开头的对照音频转写——转写里确有这句话（逐字或明确同义）、说出时间早于槽位时刻、且没有被后面的话否定才算成立；转写里找不到的一律 false；
3. 题型考点条件（逐条）：{checks}
只输出 JSON：{{"lines_ok": bool, "lines_note": "...", "facts": [{{"branch_id": "...", "slot_id": "...", "fact": "...", "holds": bool, "why": "..."}}],
"checks": {{{check_keys}}}}}"""


# ---------------------------------------------------------------- 候选窗口
_ocr_cache = {}
_ocr_lock = threading.Lock()


def ocr_of(zf, src, sec):
    key = (src, sec)
    with _ocr_lock:
        if key in _ocr_cache:
            return _ocr_cache[key]
    a = R.annotation(zf, src, sec)
    # 316 视频的部分标注里 OCR 项是 dict（10-12），取 text 字段或拼接其中的字符串值
    raw = [t.get('text') or ' '.join(v for v in t.values() if isinstance(v, str)) if isinstance(t, dict) else t
           for e in (a or {}).get('events', []) for t in (e.get('ocr') or [])]
    texts = [t for t in raw if isinstance(t, str) and t and not WATERMARK.search(t)]
    with _ocr_lock:
        _ocr_cache[key] = texts
    return texts


def _tried_v3(cap):
    """--expand 用：本批该题型已试过的窗口起点（全部 / 通过的），按视频分组。"""
    tried, passed = defaultdict(list), defaultdict(list)
    if OUT is None or not OUT.exists():
        return tried, passed
    for d in OUT.glob(f'{cap}_*'):
        f = d / 'log.json'
        if not f.exists():
            continue
        src, a = d.name[len(cap) + 1:].rsplit('_', 1)
        tried[src].append(int(a))
        try:
            if json.load(open(f)).get('status') == 'pass':
                passed[src].append(int(a))
        except Exception:
            pass
    return tried, passed


def candidates(cap, windows, zf, per_video, videos, expand=False):
    """expand=True（10-16 补题）：起点改为每 5 秒一个，跳过离已试窗口 <20 秒、离已通过窗口 <30 秒的起点；筛选/出题/核验规则不变。"""
    spec = SPECS[cap]
    tried, passed = _tried_v3(cap) if expand else ({}, {})
    kw = re.compile(spec['keywords'])
    out = []
    blocked = R._cfg.get('content_blocked', set())
    for src in [x for x in sorted(windows) if x not in blocked][:videos]:
        w = windows[src]
        secs = sorted(w)
        score = {s: len(kw.findall(w[s]['summary'])) + 3 * bool(w[s].get('speech')) + (spec.get('ocr_weight', 0) * bool(ocr_of(zf, src, s)) if spec.get('ocr_weight') else 0)
                 for s in secs}
        starts = [s for s in secs[::5 if expand else 10] if s + WIN in w and R.informative_speech(w, s, s + WIN)]
        if expand:
            starts = [s for s in starts if all(abs(s - b) >= 20 for b in tried.get(src, [])) and all(abs(s - b) >= 30 for b in passed.get(src, []))]  # Omni：窗口内要有有信息量的人声（≥80 字且有整句）
        ranked = sorted(starts, key=lambda a: -sum(score.get(s, 0) for s in range(a, a + WIN)))
        picked = []
        for a in ranked:
            if len(picked) >= per_video:
                break
            if all(abs(a - b) >= WIN for b in picked):
                picked.append(a)
        out += [{'cap': cap, 'source_id': src, 'seg': [a, a + WIN]} for a in sorted(picked)]
    return out


# ---------------------------------------------------------------- 各阶段
def times_bad(cap, t, duration):
    spec = SPECS[cap]
    try:
        t = {k: float(t[k]) for k in spec['times']}
    except (KeyError, TypeError, ValueError):
        return '缺少时刻 ' + '/'.join(spec['times'])
    for a, b, lo, hi in spec['cons']:
        if not lo <= t[b] - t[a] <= hi:
            return f'{b}−{a}={t[b] - t[a]:.1f}s，要求在 [{lo},{hi}]'
    if max(t.values()) + 3 > duration or min(t.values()) < 0:
        return '时刻超出视频范围'
    return ''


def cons_text(spec):
    return '；'.join(f'{b}−{a} 在 {lo}–{hi}s' for a, b, lo, hi in spec['cons'])


def screen(c, windows, zf, log_dir, budget, duration):
    spec, w = SPECS[c['cap']], windows[c['source_id']]
    a, b = c['seg']
    lines = []
    for s in range(a, b):
        if s in w:
            o = ocr_of(zf, c['source_id'], s)
            lines.append(R.line(w, s) + (f"  OCR: {' | '.join(o)[:120]}" if o else ''))
    sys_ = SCREEN_SYSTEM.format(cap=c['cap'], win=WIN, rule=spec['rule'], times='、'.join(spec['times']), cons=cons_text(spec),
                                time_keys=', '.join(f'"{k}": 数字' for k in spec['times']))
    msgs = [{'role': 'system', 'content': sys_}, {'role': 'user', 'content': R.lang_rule(R.video_lang(w)) + f'\n片段 {a}–{b}s：\n' + '\n'.join(lines) + f'\n\n音频转写（{a}–{b}s）：\n' + R.transcript(w, a, b)}]
    for k in range(3):
        sc = R.call(msgs, f'screen{k}', log_dir, budget, max_tokens=900)
        bad = times_bad(c['cap'], sc.get('times') or {}, duration) if sc.get('ok') else ''
        if not bad and sc.get('ok'):
            bad = R.speech_clash(w, {k: float(v) for k, v in (sc.get('times') or {}).items()})
        if not bad:
            return sc
        msgs = msgs + [{'role': 'assistant', 'content': json.dumps(sc, ensure_ascii=False)},
                       {'role': 'user', 'content': bad + '，请重新选择；做不到就 ok=false。'}]
    return {**sc, 'ok': False, 'reason': 'screen 时间三次不满足：' + bad}


def skeleton(cap, t):
    spec = SPECS[cap]
    rows = []
    for bid, br in spec['branches'].items():
        ev = '，'.join(f"{e}@{tn}={t[tn]}s op={op}" for e, tn, op in br['events'])
        sl = '，'.join(f"{s}@{tn}={t[tn]}s {act}" for s, tn, act, _ in br['slots'])
        rows.append(f'- 分支 {bid}：事件 {ev}；槽位 {sl}')
    return '\n'.join(rows)


def text_rules(cap):
    spec = SPECS[cap]
    rules = []
    if spec['same_text']:
        rules.append('两个分支的 ' + '、'.join(spec['same_text']) + ' 文本完全相同')
    for b1, e1, b2, e2 in spec.get('same_text_pairs', []):
        rules.append(f'{b1} 的 {e1} 与 {b2} 的 {e2} 文本完全相同')
    if spec['diff_text']:
        rules.append('两个分支的 ' + '、'.join(spec['diff_text']) + ' 文本必须不同')
    for b1, s1, b2, s2 in spec['distinct']:
        rules.append(f'{b1}/{s1} 与 {b2}/{s2} 的答案必须实质不同')
    return '；'.join(rules) or '无'


def frame_plan(cap, t):
    """(组名, 时刻) 列表：每个时刻取 t、t+1，再加上 extra_frames 指定的中间帧。"""
    plan = []
    for k in SPECS[cap]['times']:
        plan += [(k, t[k]), (k, t[k] + 1)]
    k0 = SPECS[cap]['times'][0]
    if t[k0] >= 1:
        plan.append((f'{k0}−1s(开口前1秒)', t[k0] - 1))  # 10-13 v3b：开口前 1 秒的帧，核对对象是否已经在画面里
    for a, b, n in SPECS[cap]['extra_frames']:
        step = (t[b] - t[a]) / (n + 1)
        plan += [('M', round(t[a] + step * (i + 1), 1)) for i in range(n)]
    return sorted(plan, key=lambda x: x[1])


def author(c, sc, zf, media, windows, log_dir, attempt, budget, feedback):
    cap, src, t = c['cap'], c['source_id'], {k: float(v) for k, v in sc['times'].items()}
    w = windows[src]
    lo, hi = int(min(t.values())) - 1, int(max(t.values())) + 3
    summ = '\n'.join(R.line(w, s) for s in range(lo, hi) if s in w)
    ann = [a for s in sorted({int(v) for v in t.values()}) if (a := R.annotation(zf, src, s))]
    tr = R.transcript(w, lo - 10, hi)
    text = R.lang_rule(R.video_lang(w)) + f"\n初步设计：{json.dumps(sc, ensure_ascii=False)}\n逐秒摘要：\n{summ}\n音频转写（{lo - 10}–{hi}s）：\n{tr}\n关键时刻详细标注：{json.dumps(ann, ensure_ascii=False)}"
    if feedback:
        text += '\n\n之前的版本没有通过检查，以下问题全部要避免：\n' + '\n'.join(f'- {x}' for x in feedback)
    content = [{'type': 'text', 'text': text}]
    for lab, tt in frame_plan(cap, t):
        content += [{'type': 'text', 'text': f'[{lab} 帧 {tt:.1f}s]'}, R.img_part(R.frame(src, media, tt))]
    sys_ = AUTHOR_SYSTEM.format(cap=cap, rule=SPECS[cap]['rule'], times=json.dumps(t), skeleton=skeleton(cap, t), text_rules=text_rules(cap))
    return R.call([{'role': 'system', 'content': sys_}, {'role': 'user', 'content': content}], f'a{attempt}_author', log_dir, budget, max_tokens=5000)


VAGUE_Q = re.compile(r'^\s*(这|那)(是|个|些|块|件|边|里)\s*(是|有|什么|啥|叫|在|用|几|多|长|属于|放|来自|做|为|的|怎么|哪)')


def vague_question(text):
    """10-13 v3b：只说"这个是什么颜色的"而不点出物品类别的问法。"""
    return bool(VAGUE_Q.match(text or ''))


def det_check(c, sc, r, duration):
    if not r.get('feasible'):
        return ['infeasible']
    cap, spec = c['cap'], SPECS[c['cap']]
    t = {k: float(v) for k, v in sc['times'].items()}
    fails = []
    if cap == 'proactive_clarify':
        for b in r['branches']:
            for e in b.get('user_events', []):
                if e.get('event_id') == 'ue0' and vague_question(e.get('text')):
                    fails.append(f"{b['branch_id']}: 问句“{e.get('text')}”没有点出物品类别名称，不能只用“这个/这是”指物")
    bad = times_bad(cap, t, duration)
    if bad:
        fails.append('时间不满足：' + bad)
    bs = {b.get('branch_id'): b for b in r.get('branches', [])}
    if set(bs) != set(spec['branches']):
        return fails + [f"分支应为 {sorted(spec['branches'])}，现为 {sorted(bs)}"]
    ev = {bid: {e.get('event_id'): e for e in bs[bid].get('user_events', [])} for bid in bs}
    sl = {bid: {s.get('slot_id'): s for s in bs[bid].get('slots', [])} for bid in bs}
    for bid, br in spec['branches'].items():
        want = [(e, round(t[tn], 1), op) for e, tn, op in br['events']]
        got = sorted(((e.get('event_id'), round(float(e.get('time_s', -1)), 1), (e.get('state_op') or {}).get('op'))
                      for e in bs[bid].get('user_events', [])), key=lambda x: [w[0] for w in want].index(x[0]) if x[0] in [w[0] for w in want] else 99)
        if got != want:
            fails.append(f'{bid}: 事件应为 {want}（event_id, 时刻, op），现为 {got}')
        if sorted(sl[bid]) != sorted(s for s, *_ in br['slots']):
            fails.append(f"{bid}: 槽位应为 {[s for s, *_ in br['slots']]}")
            continue
        for s, tn, act, _ in br['slots']:
            slot = sl[bid][s]
            n = len(slot.get('key_facts') or [])
            if act == 'SPEAK' and not 2 <= n <= 4:
                fails.append(f'{bid}/{s}: key_facts 需要 2–4 条')
            if act in ('SPEAK', 'CLARIFY') and not (slot.get('target_text') or '').strip():
                fails.append(f'{bid}/{s}: 缺少 target_text')
    lang = R.video_lang(windows_g.get(c['source_id'], {}))
    for bid in bs:
        for e in bs[bid].get('user_events', []):
            m = R.lang_bad(e.get('text'), lang)
            if m: fails.append(f"{bid}/{e.get('event_id')}: 台词语言不符：{m}")
        for slot in bs[bid].get('slots', []):
            m = R.lang_bad(slot.get('target_text'), lang)
            if m: fails.append(f"{bid}/{slot.get('slot_id')}: target_text 语言不符：{m}")
    slot_t = {s_: (tn, trig) for br in spec['branches'].values() for s_, tn, _act, trig in br['slots']}
    for bid in bs:
        for slot in bs[bid].get('slots', []):
            tn, trig = slot_t.get(slot.get('slot_id'), (None, None))
            if tn is None: continue
            limit = t[tn] + 3 if trig == 'visual' else t[tn] - 0.3  # 画面/声音触发的槽位允许引用窗口内听到的话
            for a in slot.get('audio_refs') or []:
                try: st = float(a.get('start'))
                except (TypeError, ValueError): continue
                if st > limit:
                    fails.append(f"{bid}/{slot.get('slot_id')}: 引用的转写“{str(a.get('text'))[:20]}”在 {st}s 才说出，晚于该槽位时刻 {t[tn]}s——[音频] 要点只能用槽位之前听到的话，否则换要点或把时刻往后挪")
    n_audio = 0
    for bid in bs:
        for slot in bs[bid].get('slots', []):
            for f in slot.get('key_facts') or []:
                f = (f or '').strip()
                if f.startswith('[音频]'):
                    n_audio += 1
                elif not f.startswith('[视觉]'):
                    fails.append(f"{bid}/{slot.get('slot_id')}: 要点“{f}”必须以 [音频] 或 [视觉] 开头")
            if any((f or '').strip().startswith('[音频]') for f in slot.get('key_facts') or []) and not slot.get('audio_refs'):
                fails.append(f"{bid}/{slot.get('slot_id')}: 有 [音频] 要点但 audio_refs 为空，要引用转写原句")
    if n_audio == 0 and not any(f == 'infeasible' for f in fails):
        fails.append('全模态要求：整道题至少要有一条 [音频] 要点（只有听了音轨才能知道的事实），现在一条都没有')
    for bid in bs:
        for e in bs[bid].get('user_events', []):
            if '/' in (e.get('text') or '') or '／' in (e.get('text') or ''):
                fails.append(f"{bid}/{e.get('event_id')}: 台词里不能出现“/”这类模板残留，只写一种说法")
    names = list(spec['branches'])
    for e in spec['same_text']:
        texts = {ev[b].get(e, {}).get('text') for b in names if e in ev[b]}
        if len(texts) > 1:
            fails.append(f'两个分支的 {e} 文本必须相同')
    for b1, e1, b2, e2 in spec.get('same_text_pairs', []):
        if ev[b1].get(e1, {}).get('text') != ev[b2].get(e2, {}).get('text'):
            fails.append(f'{b1}/{e1} 与 {b2}/{e2} 文本必须相同')
    for e in spec['diff_text']:
        texts = [ev[b].get(e, {}).get('text') for b in names]
        if len(set(texts)) < len(texts):
            fails.append(f'两个分支的 {e} 文本必须不同')
    for b1, s1, b2, s2 in spec['distinct']:
        x, y = (R.bigrams(sl[b].get(s, {}).get('target_text', '')) for b, s in ((b1, s1), (b2, s2)))
        if x and y and len(x & y) / len(x | y) >= 0.5:
            fails.append(f'{b1}/{s1} 与 {b2}/{s2} 的答案太相似，必须实质不同')
    return fails


def leak(r, log_dir, attempt, budget):
    fails = []
    for b in r['branches']:
        facts = [f for s in b.get('slots', []) for f in (s.get('key_facts') or []) if s.get('target_text')]
        if not facts:
            continue
        lines = [f"{e['time_s']}s：{e['text']}" for e in b['user_events']]
        v = R.call([{'role': 'system', 'content': LEAK_SYSTEM},
                    {'role': 'user', 'content': f"用户台词：{json.dumps(lines, ensure_ascii=False)}\n评分时刻答案要点：{json.dumps(facts, ensure_ascii=False)}"}],
                   f'a{attempt}_leak_{b["branch_id"]}', log_dir, budget, max_tokens=800)
        fails += [f"{b['branch_id']}: 台词“{x.get('line')}”里的“{x.get('phrase')}”透露了答案" for x in v.get('leaks', [])]
    return fails


def verify(c, sc, r, media, log_dir, attempt, budget):
    cap, src, spec = c['cap'], c['source_id'], SPECS[c['cap']]
    t = {k: float(v) for k, v in sc['times'].items()}
    slot_time = {s: tn for br in spec['branches'].values() for s, tn, *_ in br['slots']}
    brief = {'times': t, 'branches': [{'branch_id': b['branch_id'], 'user_events': [[e['time_s'], e['text']] for e in b['user_events']],
                                       'slots': [{'slot_id': s['slot_id'], 'time': slot_time.get(s['slot_id']), 'target_text': s.get('target_text'),
                                                  'key_facts': s.get('key_facts')} for s in b['slots']]} for b in r['branches']]}
    lo, hi = int(min(t.values())) - 12, int(max(t.values())) + 4
    content = [{'type': 'text', 'text': '题目：' + json.dumps(brief, ensure_ascii=False) + f'\n\n音频转写（{lo}–{hi}s）：\n' + R.transcript(windows_g[src], lo, hi)}]
    for lab, tt in frame_plan(cap, t):
        content += [{'type': 'text', 'text': f'[{lab} {tt:.1f}s]'}, R.img_part(R.frame(src, media, tt))]
    checks = '；'.join(f'{k}：{v}' for k, v in spec['checks'].items())
    sys_ = VERIFY_SYSTEM.format(cap=cap, rule=spec['rule'], checks=checks,
                                check_keys=', '.join(f'"{k}": {{"ok": bool, "note": "..."}}' for k in spec['checks']))
    v = R.call([{'role': 'system', 'content': sys_}, {'role': 'user', 'content': content}], f'a{attempt}_verify', log_dir, budget, max_tokens=2500)
    fails = []
    if not v.get('lines_ok'):
        fails.append(f"用户台词在说出时刻的画面里不成立（{v.get('lines_note', '')}）")
    fails += [f"{f.get('branch_id')}/{f.get('slot_id')}: 要点“{f.get('fact')}”在帧里不成立（{f.get('why', '')}）"
              for f in v.get('facts', []) if not f.get('holds')]
    for k, desc in spec['checks'].items():
        x = (v.get('checks') or {}).get(k) or {}
        if not x.get('ok'):
            fails.append(f"考点条件不成立：{desc}（{x.get('note', '')}）")
    return fails, v


T0_GATE_SYSTEM = """你是严格的视觉核验员。第一视角博物馆视频里，用户在 t0 时刻对模型说了一句话，让模型留意/只关注某个对象 X。
给你 t0−1s、t0、t0+1s 的真实帧。判断：用户在 t0 开口时，X 是否**已经清楚地出现在画面里**（在 t0−1s 和 t0 的帧里都能看到并辨认出它，不是尚未出现、刚入画一角、被遮挡或模糊到无法确认）。
台词里的描述（位置、衣着、外观）必须能在 t0 帧里对上。只输出 JSON：{"visible_at_t0": bool, "note": "一句话说明在哪帧看到/没看到什么"}"""


def t0_gate(c, sc, r, media, log_dir, attempt, budget):
    """10-13 v3b：temporal/selective 的独立闸门——t0 开口时对象必须已经清楚在画面里（合并在 verify 里的同名检查经常放水）。"""
    if c['cap'] not in ('temporal_update', 'selective_attention'):
        return []
    b0 = r['branches'][0]
    e0 = min(b0['user_events'], key=lambda e: e['time_s'])
    t0 = float(e0['time_s'])
    content = [{'type': 'text', 'text': f"t0={t0}s 用户说：“{e0['text']}”\n对象 X 的内部描述：{json.dumps((e0.get('state_op') or {}).get('target', ''), ensure_ascii=False)}"}]
    for lab, tt in (('t0−1s', max(0.0, t0 - 1)), ('t0', t0), ('t0+1s', t0 + 1)):
        content += [{'type': 'text', 'text': f'[{lab} 帧 {tt:.1f}s]'}, R.img_part(R.frame(c['source_id'], media, tt))]
    v = R.call([{'role': 'system', 'content': T0_GATE_SYSTEM}, {'role': 'user', 'content': content}], f'a{attempt}_t0gate', log_dir, budget, max_tokens=400)
    if v.get('visible_at_t0'):
        return []
    return [f"t0 开口时对象还没清楚出现在画面里（{v.get('note', '')}）——要么换成 t0 时已经在画面里的对象，要么做不出来就 feasible=false"]


def to_task(c, sc, r):
    cap, spec = c['cap'], SPECS[c['cap']]
    t = {k: float(v) for k, v in sc['times'].items()}
    by_id = {b['branch_id']: b for b in r['branches']}
    branches = []
    for bid, br in spec['branches'].items():
        b = by_id[bid]
        events = sorted(b['user_events'], key=lambda e: e['time_s'])
        for e in events:  # 记录用户开口时是否与人声重叠（eval 做 TTS 混音时据此压低原音轨）
            seg = R.speech_at(windows_g.get(c['source_id'], {}), float(e['time_s']))
            e['speech_overlap'] = bool(seg)
        by_eid = {e['event_id']: e for e in events}
        at = {tn: by_eid[eid] for eid, tn, _ in br['events'] if eid in by_eid}
        given = {s['slot_id']: s for s in b['slots']}
        slots = []
        for s, tn, act, trig in br['slots']:
            g = given[s]
            trigger = ({'kind': 'user_query', 'event_id': at[tn]['event_id']} if trig == 'user' and tn in at
                       else {'kind': 'visual_event', 'event_id': f've_{s}'})
            slots.append({'slot_id': s, 'trigger': trigger, 'window': {'start_s': t[tn], 'end_s': t[tn] + 3},
                          'depends_on': [e['event_id'] for e in events if e['time_s'] <= t[tn]], 'expected_action': act,
                          'target_text': g.get('target_text', '') if act != 'WAIT' else '',
                          'key_attributes': g.get('key_facts') or [], 'forbidden_claims': g.get('forbidden_claims', []),
                          'audio_refs': g.get('audio_refs') or [],
                          'modality_facts': {'audio': [f for f in (g.get('key_facts') or []) if f.startswith('[音频]')],
                                             'visual': [f for f in (g.get('key_facts') or []) if f.startswith('[视觉]')]}})
        branches.append({'branch_id': bid, 'user_events': events, 'slots': slots})
    t0 = min(t.values())
    return {'source_id': c['source_id'], 'capability': cap, 'interaction_family': spec['family'],
            'task_id': f"OM_V3_{cap[:4].upper()}_{c['source_id']}_{int(t0):05d}", 'status': 'omni_candidate_pending_human_review', 'modality': 'omni', 'language': R.video_lang(windows_g.get(c['source_id'], {})),
            'session': {'start_s': max(0.0, t0 - 2), 'end_s': max(t.values()) + 3, 'observation_hz': 5, 'history_policy': 'causal_prefix'},
            'shared_future_from_s': t0, 'diagnostic_slot_ids': spec['diagnostic'], 'branches': branches,
            'v3': {'version': 'omni_v1_20261016', 'model': R.MODEL, 'segment': c['seg'], 'times': t, 'plan': sc.get('plan', ''), 'audio_use': sc.get('audio_use', '')}}


def run_one(c, windows, media_map, args):
    src = c['source_id']
    media, duration = media_map[src]
    log_dir = OUT / f"{c['cap']}_{src}_{c['seg'][0]:05d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    rec, status = {'candidate': c, 'attempts': []}, 'screen_rejected'
    try:
        with zipfile.ZipFile(R.ARCHIVE) as zf:
            sc = screen(c, windows, zf, log_dir, args.budget, duration)
            rec['screen'] = sc
            if sc.get('ok'):
                status, feedback = 'needs_revision', []
                for attempt in range(1, args.max_attempts + 1):
                    a = {'attempt': attempt}
                    r = author(c, sc, zf, media, windows, log_dir, attempt, args.budget, feedback)
                    a['author_output'] = r
                    fails = det_check(c, sc, r, duration)
                    if fails == ['infeasible']:
                        status = 'infeasible'
                        rec['attempts'].append(a)
                        break
                    if not fails:
                        fails = leak(r, log_dir, attempt, args.budget)
                    if not fails:
                        fails, a['visual_check'] = verify(c, sc, r, media, log_dir, attempt, args.budget)
                    if not fails:
                        fails = t0_gate(c, sc, r, media, log_dir, attempt, args.budget)
                    a['fails'] = fails
                    rec['attempts'].append(a)
                    if not fails:
                        status = 'pass'
                        task = to_task(c, sc, r)
                        for name in ('task.json', 'candidate_task.json'):
                            json.dump(task, open(log_dir / name, 'w'), ensure_ascii=False, indent=1)
                        break
                    feedback = list(dict.fromkeys(feedback + fails))
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
        status = 'error'
    rec['status'] = status
    json.dump(rec, open(log_dir / 'log.json', 'w'), ensure_ascii=False, indent=1)
    print(f"[{status}] {c['cap']} {src}@{c['seg']} spent=${R._spent['usd']:.3f}", flush=True)
    return c['cap'], status


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, choices=list(CFG.BATCHES))
    ap.add_argument('--caps', default=','.join(SPECS))
    ap.add_argument('--per-video', type=int, default=1)
    ap.add_argument('--videos', type=int, default=999)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--budget', type=float, default=2.0)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--expand', action='store_true', help='10-16 补题：错开起点扩候选（见 candidates）')
    args = ap.parse_args()
    R.use_batch(args.batch)
    OUT = CFG.RUNS / args.batch / 'runs_v3'
    windows, media_map = R.load_windows(), R.media_paths()
    windows_g.update(windows)
    with zipfile.ZipFile(R.ARCHIVE) as zf:
        cands = [c for cap in args.caps.split(',') for c in candidates(cap, windows, zf, args.per_video, args.videos, args.expand)]
    cands = [c for c in cands if not (OUT / f"{c['cap']}_{c['source_id']}_{c['seg'][0]:05d}" / 'log.json').exists()]
    print('candidates', len(cands), flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda c: run_one(c, windows, media_map, args), cands))
    summary = defaultdict(lambda: defaultdict(int))
    for cap, s in results:
        summary[cap][s] += 1
    out = {'by_cap': {k: dict(v) for k, v in summary.items()}, 'spent_usd': round(R._spent['usd'], 4), 'calls': R._spent['calls']}
    json.dump(out, open(CFG.RUNS / args.batch / 'summary_v3_last_run.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
