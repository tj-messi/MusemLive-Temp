# MuseumLive 0–5 分 Judge 与案例展示

对现有 10 段视频、240 条 MiniCPM-o-4.5 已保存回答进行文本语义整体评分，并生成用于组内展示的「原始截图 + QA + 得分 + 评分理由」独立 HTML。展示精选 4 个 0 分与 4 个 5 分案例，包含 13 张原评测截图，支持图片放大、分数筛选和逐例演示。

关联 #1。本目录独立于标注管线；这次交付没有修改原始问题、参考答案或候选模型回答。

## 已运行的结果

239 条请求全部成功；1 条已知题意不清的 QA 在请求前列为待复核，不记 0 分。纳入已确认的 17 条人工复核分数后：

| 口径 | 有效评分 | 均分 / 5 | 0–1 分 | 2–3 分 | 4–5 分 |
|---|---:|---:|---:|---:|---:|
| 纳入已确认人工复核 | 239 | 4.025105 | 24 | 35 | 180 |
| 统一自动评分 | 239 | 4.029289 | 23 | 35 | 181 |

人工复核口径总分为 962/1195，5 分 148 条，0 分 15 条。分母排除待复核与 API 失败，不把空值转换成零分。自动结果与人工覆盖层分别保存。

这里的「全量」只指当前 240 条保存的回答。参考答案仍由机器编写、待全面人审；这些结果不是最终独立人标 Test Pool 的主结果。Full 与 Sparse 的 QA 不同，组间均分不能当作同题对照。17 条人审样本用于开发校准，14 条与新 Judge 完全一致、3 条相差 1 分，这不是独立验证集上的 Judge 准确率。

## 评分协议

- Judge：商业模型 `x-ai/grok-4.6`，temperature 0，最多 900 输出 tokens。
- 借鉴 OmniMMI 官方文本问答语义比较，输出独立的 yes/no 与 0–5 整数分数；没有逐题加权得分点。
- v2 明确按问题所问判断完整性：短但完整准确的答案可以得 5 分；不要求复述未被问到的参考细节。描述题仍考虑相关信息是否完整、准确。
- 请求只包含问题、参考答案和模型回答；不包含旧评分细则、人工分数、候选模型身份、图片或对话历史。
- 明确的 QA 歧义可以返回空分数并等待复核；API 失败单独记录，不计 0 分。
- 所有 239 条使用同一固定 v2 提示词重新评估。历史 v1 自动分数不混入本次自动结果。

完整提示词在 [prompts](prompts)。这不是对 OmniMMI 的原样复现：Judge 模型、问题范围澄清、结构化输出及待复核字段经过适配。

OmniMMI 官方评估脚本固定来源：

- URL：<https://raw.githubusercontent.com/OmniMMI/OmniMMI/2b93f6ad63126581769151df635b2234f25c2c33/evaluations/evaluate.py>
- SHA256：`390028e506ea4f383322853b073b5469310ab8d0887b7c2aa3861a6974b0a12c`
- 输入 `omni_review.json` SHA256：`ab1d65437cc87ea8091548d44ed4b21fef822eaf906d82fee808bda9e5bfeea5`

## 运行与复现

要求 Python 3.10+。从仓库根目录运行：

```sh
python -m pip install -r pipelines/benchmark_judge/requirements.txt
python -m unittest discover -s pipelines/benchmark_judge/tests -v
```

将既有数据与运行产物放在仓库外的 `ARTIFACT_ROOT` 下，所需结构：

```text
ARTIFACT_ROOT/
  qa10_ab_20260926/
    omni_review.json
    frames/                         # 原始证据帧；不能用占位图替代
  official_evaluate.py               # 上述固定来源；读取前校验 SHA256
  museumlive_review_summary_20260928/
    评分汇总.json
    你的10题反馈_原件.json
    你的8题反馈_原件.json
  museumlive_benchmark_full_20260928/
    configuration.json, sample.json, human_review.json
    results.json, summary.json, manifest.json
    统计核验.json
    requests/                       # 每次请求和响应
    reproducibility/                # 执行时的源码原始字节快照
```

这些数据和含截图的 HTML 是外部产物，不放入源码仓库。本仓库保存代码、模板、统计、选择清单与校验值。现有产物由本次执行者维护在 `museumlive_benchmark_full_20260928`；取得它和证据帧后无需 API 即可重建展示。

准备新的评分运行（将示例路径替换为实际产物路径）：

```sh
python pipelines/benchmark_judge/code/run_holistic_benchmark.py \
  --input-root ARTIFACT_ROOT/qa10_ab_20260926 \
  --official-source ARTIFACT_ROOT/official_evaluate.py \
  --feedback-dir ARTIFACT_ROOT/museumlive_review_summary_20260928 \
  --output-dir ARTIFACT_ROOT/new_judge_run --prepare-only
```

准备后，将 `configuration.json` 的 `code_sha256` 所列三个源码文件按原始字节复制到运行目录的 `reproducibility/`，避免换行符或后续编辑改变冻结版本。通过安全环境变量提供 `OPENROUTER_KEY1`，去掉 `--prepare-only` 才会调用 API。如网络需要代理，再设置 `OPENROUTER_PROXY`；未设置时直接连接 OpenRouter。每条最多 3 次尝试，成功响应不按分数高低重跑。

核验已有评分并生成统计，再生成展示页：

```sh
python pipelines/benchmark_judge/code/summarize_holistic_benchmark.py \
  --input-root ARTIFACT_ROOT/qa10_ab_20260926 \
  --output-dir ARTIFACT_ROOT/museumlive_benchmark_full_20260928

python pipelines/benchmark_judge/code/build_holistic_visualization.py \
  --input-root ARTIFACT_ROOT/qa10_ab_20260926 \
  --run-dir ARTIFACT_ROOT/museumlive_benchmark_full_20260928
```

生成 `Judge_展示版_截图与QA.html`，直接用浏览器打开。统计数字、全部案例与原图在生成时直接写入 HTML；即使预览器禁用或删除脚本，仍能阅读内容。脚本只增强分数筛选、图片放大与逐例展示，不支持的交互自动隐藏。原图以原始字节内置，不需要联网；案例图都先校验 SHA256 和提问前时间截止。展示案例选择固定在代码与 `showcase_selection.json`，属于讲解样例，不代表总体分布。原问题、参考答案、实际回答和评分理由保持原文；标题只是展示说明。

这次兼容修复的外部交付文件另名为 `Judge_展示版_手机兼容修复版.html`，方便区分已转发的旧副本；默认文件名也已替换为相同内容。生成指定文件名、运行浏览器检查并导出 PDF：

```sh
python pipelines/benchmark_judge/code/build_holistic_visualization.py \
  --input-root ARTIFACT_ROOT/qa10_ab_20260926 \
  --run-dir ARTIFACT_ROOT/museumlive_benchmark_full_20260928 \
  --output ARTIFACT_ROOT/museumlive_benchmark_full_20260928/Judge_展示版_手机兼容修复版.html

python -m pip install -r pipelines/benchmark_judge/requirements-browser.txt
python pipelines/benchmark_judge/tests/check_showcase_browser.py \
  --run-dir ARTIFACT_ROOT/museumlive_benchmark_full_20260928 \
  --html ARTIFACT_ROOT/museumlive_benchmark_full_20260928/Judge_展示版_手机兼容修复版.html \
  --work-dir BROWSER_CHECK_WORK_DIR \
  --pdf ARTIFACT_ROOT/museumlive_benchmark_full_20260928/Judge_展示版_直接转发.pdf
```

浏览器检查需要本地 Chrome，可用 `--chrome` 指定路径。PDF 在禁用脚本时导出，A4 横向，共 9 页：一页统计和每例一页；可直接转发。`--pdf` 可省略。浏览器截图、HTML、PDF 和原始数据保存在仓库外，仓库仅收录验证记录与校验值。

## 验证证据与边界

[results/holistic_v2_20260928](results/holistic_v2_20260928) 保存了数值统计、执行与展示清单和验证记录。已运行的检查包括：240 条覆盖去重、输入不变、每条有效响应的模型和分数、原始请求对应关系、人工与自动结果分离、缺失分母、分数段加总、分组统计与凭据检查。

7 项单元测试通过，包含删除所有脚本后内容仍保留、QA 文本正确转义的回归测试。Chrome 实际加载展示页，确认 8 个案例、13 张截图、全量统计、0/5 分筛选、截图放大和键盘逐例导航；分别检查正常运行、禁用脚本、删除脚本、缺少对话框支持。1360/390/360 像素视口没有横向溢出、JavaScript 错误或外部网络请求。原始 QA 文本和全部图片字节哈希均与数据源一致。检查范围是 Chrome 中模拟的视口与预览限制，未直接测试收件人的手机应用。

已查看手机禁用脚本时的实际截图。PDF 经 Poppler 渲染并查看全部 9 页，各页 QA、评分理由和 13 张截图的数量已核验，没有观察到截断。完整记录保存在 `showcase_browser_validation.json` 与 `showcase_pdf_validation.json`。评分输入、原始自动结果与统计均未改变。

既有 240 条模型推理没有重新运行；本次重新执行的是文本 Judge。截图供展示读者查看，没有传给 Judge。这些分数不单独验证视频事实，也不构成反事实或因果推断能力的证据。
