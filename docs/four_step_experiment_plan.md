# 四步推理链实验项目说明文档

> 项目背景：在现有多轮强化学习分子生成框架（M4olGen + GRPO）基础上，引入
> `Analyse → Plan → Edit → Verify` 四步推理链，验证其对多属性约束精度的提升效果，
> 并与论文 §4.2 的因果推理理论对应。

---

## 目录

1. [整体实验设计](#1-整体实验设计)
2. [前置工作：合成四步SFT数据集](#2-前置工作合成四步sft数据集)
3. [代码改动清单](#3-代码改动清单)
4. [三级消融实验](#4-三级消融实验)
5. [评估指标与脚本](#5-评估指标与脚本)
6. [已知问题修复备忘](#6-已知问题修复备忘)
7. [实验运行顺序](#7-实验运行顺序)
8. [论文对应关系](#8-论文对应关系)

---

## 1. 整体实验设计

### 实验条件

| 条件 | 名称 | 说明 |
|------|------|------|
| A | Baseline（无四步） | 现有 `run_multistep_rl_training_fixed` |
| B1 | 结构奖励 | 检查 `<PLAN> + <EDIT>` 是否稳定输出 |
| B2 | 结构 + Plan-Edit 一致性 | 检查 Plan 最终决策是否与 Edit 一致 |
| B3 | 结果主导联合奖励 | 结构 + 一致性 + 原有属性 outcome reward |

### 固定超参（所有条件一致）

```
模型：       Qwen3-8B（从 SFT checkpoint 起步）
hop 数：     3
trajectories_per_prompt： 4
prompts_per_iter：        4
learning_rate：           1e-5
gamma：                   0.85
评估协议：   100 target tuples，best-of-10
数据集：     Alan123/qa_dataset_with_qed_logp_mw_grpo
```

---

## 2. 前置工作：合成四步SFT数据集

### 为什么需要先合成数据

Qwen3-8B 从未见过 `<ANALYSE>/<PLAN>/<EDIT>/<VERIFY>` 格式，纯 RL 探索收敛极慢（需要约 10x 迭代）。SFT 冷启动只需合成 **3000–5000 条**，成本低效果好。

### 合成策略

原数据集每条已有：`prompt`（分子 + 属性目标）+ `output`（目标 SMILES）。

- **ANALYSE / VERIFY 的数字**：直接用 RDKit 计算，精确填入，不依赖大模型
- **EDIT 的金标准**：直接复用原始 `qa_dataset_with_qed_logp_mw_grpo` 中已有的 `output`
- **PLAN 的推理文字**：调用更强的推理模型补写，但其角色是对 gold `EDIT` 的化学解释，不要求它是“真实原始思维轨迹”

换句话说，当前数据构建任务不是“重新生成 edit”，而是：

- 保留原始数据集已有的正确 `EDIT`
- 用程序精确注入 `ANALYSE / VERIFY`
- 为每条 gold edit 补一段高质量 `PLAN`

### 合成脚本

```python
import re
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors

SYNTHESIS_SYSTEM = (
    "You are a chemistry reasoning assistant. "
    "Given an input molecule and a target molecule, write ONLY the <PLAN> block "
    "explaining why a specific fragment edit achieves the desired property change. "
    "One sentence max. Be concrete about which fragment changes which property."
)

def synthesize_fourstep_example(prompt: str, answer_smiles: str, llm_call_fn) -> str | None:
    """
    根据 (prompt, answer_smiles) 合成一条完整四步格式的训练样本。
    llm_call_fn: 接受 (system, user) → str 的函数，对接你的 API 客户端。
    返回完整的四步字符串，失败返回 None。
    """
    # ── 1. 从 prompt 解析输入分子 ──────────────────────────────────────
    smiles_match = re.search(r"<SMILES>(.*?)</SMILES>", prompt)
    if not smiles_match:
        return None
    input_smiles = smiles_match.group(1).strip()

    mol_in  = Chem.MolFromSmiles(input_smiles)
    mol_out = Chem.MolFromSmiles(answer_smiles)
    if mol_in is None or mol_out is None:
        return None

    # ── 2. 用 RDKit 计算所有属性（ANALYSE + VERIFY 数字精确） ─────────
    def get_props(mol):
        return {
            "QED":  round(QED.qed(mol), 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "MW":   round(Descriptors.ExactMolWt(mol), 2),
        }

    props_in  = get_props(mol_in)
    props_out = get_props(mol_out)

    def fmt_delta(name):
        delta = props_out[name] - props_in[name]
        sign  = "higher" if delta > 0 else "lower"
        return f"Δ={abs(delta):.3f}(need {sign})"

    def remaining_gap(name, target_match):
        # 从 prompt 解析绝对目标（如果有），否则用 delta 方向判断
        t_m = re.search(rf"<{name}>(-?\d+\.?\d*)</{name}>\s*(higher|lower)", prompt)
        if t_m:
            base    = props_in[name]
            delta   = float(t_m.group(1))
            direction = 1 if t_m.group(2) == "higher" else -1
            target  = base + direction * delta
            gap     = target - props_out[name]
            if abs(gap) < 1e-3:
                return f"{name}: on target ✓"
            sign = "higher" if gap > 0 else "lower"
            return f"{name} Δ={abs(gap):.3f}(still need {sign})"
        return f"{name}: Δ={props_out[name] - props_in[name]:+.3f}"

    # ── 3. 生成 PLAN 块（唯一需要 LLM 的部分） ────────────────────────
    plan_user = (
        f"Input SMILES: {input_smiles}\n"
        f"Input props:  QED={props_in['QED']}, LogP={props_in['LogP']}, MW={props_in['MW']}\n"
        f"Output SMILES: {answer_smiles}\n"
        f"Output props:  QED={props_out['QED']}, LogP={props_out['LogP']}, MW={props_out['MW']}\n\n"
        "Write the <PLAN> block ONLY (no other tags). "
        "Include: Proposed edit, Expected effect (direction + magnitude), Reasoning (1 sentence)."
    )
    plan_raw = llm_call_fn(SYNTHESIS_SYSTEM, plan_user)
    if not plan_raw:
        return None
    # 清理多余标签（防止 LLM 自作主张加了其他块）
    plan_content = re.sub(r"</?PLAN>", "", plan_raw).strip()

    # ── 4. 从 prompt 或 answer 推断 fragment edit 动作 ─────────────────
    # 简单策略：直接描述 "Replace X with Y"（如果 prompt 中有 fragment 信息则复用）
    frag_match = re.search(
        r"(Replace|Add|Remove)\s+(.+?)(?:\s+with\s+(.+?))?\s+to\s+form",
        prompt
    )
    if frag_match:
        edit_sentence = frag_match.group(0) + f" <SMILES>{answer_smiles}</SMILES>."
    else:
        edit_sentence = f"Edit molecule to form <SMILES>{answer_smiles}</SMILES>."

    # ── 5. 拼装完整四步输出 ────────────────────────────────────────────
    analyse_block = (
        f"<ANALYSE>\n"
        f"Current molecule: <SMILES>{input_smiles}</SMILES>\n"
        f"Property gaps: QED {fmt_delta('QED')}, "
        f"LogP {fmt_delta('LogP')}, MW {fmt_delta('MW')}\n"
        f"</ANALYSE>"
    )
    plan_block = f"<PLAN>\n{plan_content}\n</PLAN>"
    edit_block = f"<EDIT>\n{edit_sentence}\n</EDIT>"

    all_met = all(abs(props_out[k] - props_in[k]) < 0.05 for k in ["QED", "LogP"])
    verify_block = (
        f"<VERIFY>\n"
        f"New QED={props_out['QED']}, LogP={props_out['LogP']}, MW={props_out['MW']}\n"
        f"Remaining gaps: {remaining_gap('QED', None)}; "
        f"{remaining_gap('LogP', None)}; {remaining_gap('MW', None)}\n"
        f"All targets met: {'Yes' if all_met else 'No'}\n"
        f"</VERIFY>"
    )

    return "\n\n".join([analyse_block, plan_block, edit_block, verify_block])


def build_sft_dataset(raw_data, llm_call_fn, n_samples: int = 4000, output_path: str = "fourstep_sft.jsonl"):
    """
    批量合成，写入 JSONL 文件供 SFT 使用。
    """
    import json, random, time
    indices = random.sample(range(len(raw_data)), min(n_samples * 2, len(raw_data)))
    collected = 0

    with open(output_path, "w") as f:
        for idx in indices:
            if collected >= n_samples:
                break
            example = raw_data[int(idx)]
            prompt  = str(example["input"])
            answer  = str(example["output"])

            result = synthesize_fourstep_example(prompt, answer, llm_call_fn)
            if result is None:
                continue

            record = {"prompt": prompt, "completion": result}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            collected += 1

            if collected % 100 == 0:
                print(f"  Synthesized {collected}/{n_samples}")
            time.sleep(0.1)  # 防止 API rate limit

    print(f"Done. {collected} examples saved to {output_path}")
```

### 数量建议

| 用途 | 数量 |
|------|------|
| SFT 冷启动 | 3000–5000 条 |
| 验证集（不参与训练） | 200 条 |
| 预期 SFT 步数 | 3000–5000 steps |

### 2.1 两阶段 SFT 策略

这里需要明确区分两类四步数据的用途：

- **阶段一：格式型四步数据**  
  目标是让模型稳定学会 `<ANALYSE>/<PLAN>/<EDIT>/<VERIFY>` 的输出骨架。
  这一步可以使用当前已经合成好的 4000/200 四步母数据，即使 `PLAN` 偏模板化也没有关系，因为此时关注点是：
  - 四个 block 是否稳定输出
  - 每个 block 是否包含固定字段
  - `<EDIT>` 和 `<VERIFY>` 的结构是否可靠

  但在进入当前正式方案训练前，仍建议把它导出成与推理链路一致的 `PLAN + EDIT` 两步 supervision。

- **阶段二：高质量 reasoning 四步数据**  
  目标是让模型真正学会 `PLAN` 的规划内容，而不是只会填三行模板。
  这一步需要后续额外构建更高质量的四步样本，重点提升：
  - `ANALYSE` 是否真的识别出属性方向和关键矛盾
  - `PLAN` 是否明确指出要改哪个 fragment、为什么这会影响 QED/LogP/MW
  - `VERIFY` 是否与真实 RDKit 结果自洽

推荐训练顺序如下：

1. **SFT Stage 1：格式冷启动**
   - 数据：由四步母数据集导出的 `PLAN + EDIT` 两步训练集（约 4000 train + 200 val）
   - 目标：学会当前正式推理链路下的 `PLAN + EDIT` 输出结构
   - 结果：得到一个能稳定输出 `PLAN + EDIT` 的起始 checkpoint

2. **SFT Stage 2：高质量 reasoning 蒸馏**
   - 数据：后续构建的小而精的高质量四步数据集
   - 目标：强化 `PLAN` 的化学解释能力与 `VERIFY` 的自洽性
   - 结果：得到一个真正适合作为 RL 起点的四步 checkpoint

换句话说，第一阶段主要教模型“长成什么样”，第二阶段再教模型“里面该说什么”。

### 2.2 当前正式采用的运行方案：程序注入 `ANALYSE` / `VERIFY`

经过对第一阶段 SFT 结果的检查，当前项目正式采用如下方案：

- **训练时注入 `ANALYSE`**
- **推理时也注入 `ANALYSE`**
- **模型只负责生成 `PLAN + EDIT`**
- **程序根据 `EDIT` 的结果计算并回填 `VERIFY`**

最终交付给评估器或下游模块的仍然是完整四步：

1. 程序先根据输入分子计算 `ANALYSE`
2. 将 `ANALYSE + 原始任务 prompt` 一起喂给模型
3. 模型续写 `PLAN + EDIT`
4. 程序解析 `<EDIT>` 里的目标分子 SMILES
5. 程序基于 RDKit 计算真实属性并拼出 `VERIFY`
6. 最终组合成完整的 `<ANALYSE>/<PLAN>/<EDIT>/<VERIFY>` 输出

这样做的直接原因是：

- `QED/LogP/MW` 是可计算量，不适合完全交给语言模型“心算”
- 第一阶段 SFT 已经证明，模型可以学会四步结构，但不能稳定生成精确属性数值
- 因此应把**可验证数值部分交给程序**，把**规划和编辑决策交给模型**

这个方案兼顾了两点：

- **保留完整四步实验框架**
- **避免模型在 `ANALYSE/VERIFY` 中产生不可靠数值**

从训练目标看，这等价于把四步任务重新拆成：

- 程序负责：`ANALYSE`、`VERIFY`
- 模型负责：`PLAN`、`EDIT`

因此后续 SFT、RL、评估都应围绕这个职责划分来设计，而不再默认“模型需要独立生成所有四步中的精确数值”。

### 2.3 当前对 `PLAN` 的重新定义

经过后续讨论，当前项目对 `PLAN` 的定位进一步明确为：

- `PLAN` 的主要作用是**帮助模型形成更好的 edit 决策**，而不是成为一个复杂的、完全可 regex 打分的中间变量
- `PLAN` 可以是 **post-hoc chemically grounded explanation**
- 也就是说：先给出 gold `EDIT`，再让强模型补写“为什么这个 edit 合理”，这是允许且合理的

这里需要明确区分两个概念：

1. **trace（原始思维轨迹）**
   - 指模型在真正做题时一步一步产生的原始中间思考
   - 当前数据集并不追求严格恢复这种内容

2. **explanation（事后化学解释）**
   - 指在已知 gold edit 正确的前提下，补写一段化学上站得住脚的中间理由
   - 当前项目正式采用的就是这一类 `PLAN`

因此，后续文档中提到的高质量 `PLAN`，默认含义是：

- 与 gold `EDIT` 严格一致
- 与真实 RDKit 属性变化不冲突
- 能解释 fragment change 与 `QED / LogP / MW` 变化的因果关系
- 不要求它是模型真实未见答案时的原始草稿纸

### 2.4 当前正式采用的数据形态

为了兼顾可审计性和训练一致性，推荐把数据分成两层：

1. **母数据集（完整四步）**
   - 保留 `<ANALYSE>/<PLAN>/<EDIT>/<VERIFY>` 四步
   - 其中 `ANALYSE / VERIFY` 为程序注入，`EDIT` 为原始 gold answer，`PLAN` 为后补解释
   - 主要用于审阅、抽查和离线分析

2. **训练导出数据（两步输出）**
   - `user = 原始 prompt + 程序注入的 ANALYSE`
   - `assistant = PLAN + EDIT`
   - `VERIFY` 不作为模型自由生成目标，只作为程序后处理结果

这意味着：

- 四步完整数据可以保留
- 但真正喂给当前正式方案的主训练数据，推荐是 `PLAN + EDIT` 两步输出

### 2.5 当前正式采用的 `PLAN` 质量标准

高质量 `PLAN` 不再追求固定三行模板，而是追求“自由但可靠”的化学解释。建议至少满足：

- 明确指出具体修改的 fragment / scaffold
- 明确说明 `QED / LogP / MW` 的变化方向
- 提供结构到性质的因果解释，例如 polarity、halogen、hydrogen-bonding、aromaticity、bulk、rigidity、charge 等
- 不出现模板化空话，如 “moves the properties in the requested direction”
- 不出现元讨论，如 `Thinking Process`、`Task`、`Constraints`、对 prompt/tag 的解释
- 与 gold `EDIT`、真实 `VERIFY` 结果保持一致

### 2.6 当前推荐的数据生成质量控制

在大规模补写 `PLAN` 时，推荐优先保证数据纯度，而不是强行保留每一条样本：

- 可以使用更强的模型来生成 `PLAN`
- 可以采用“生成 → 审稿 → 最多一次修订”的轻量多模型流程
- 如果 `PLAN` 与 gold `EDIT` 不一致、出现元讨论、或解释明显空泛，则应直接丢弃样本

原则上：

- **宁可少一些，也不要把低质量 `PLAN` 混进 SFT 数据**

---

## 3. 代码改动清单

### 3.1 新增：四步 System Prompt

在 notebook 中新增一个 prompt 常量，**不覆盖原有的 `SYSTEM_PROMPT_RL`**。  
注意：这里的四步 prompt 不再要求模型独立生成完整四步全部内容，而是要求模型在给定程序注入的 `ANALYSE` 基础上，输出 `PLAN + EDIT`，`VERIFY` 由程序回填。`PLAN` 采用自由解释风格，而不是固定三行模板。

```python
SYSTEM_PROMPT_FOURSTEP = """
You are a chemistry assistant for multi-turn molecule optimization.
You will be given a program-computed <ANALYSE> block.
Do not rewrite or alter the analysis numbers.

Your job is to output ONLY the following two blocks:

<PLAN>
Write 2-4 sentences of chemically grounded reasoning.
Explain which fragment or scaffold should be changed and why that should move
QED / LogP / MW in the desired direction.
End with one decision sentence:
Therefore I will [Replace/Add/Remove] ...
</PLAN>

<EDIT>
Replace <old_fragment> with <new_fragment> to form <SMILES>new_smiles_here</SMILES>.
</EDIT>

The <VERIFY> block will be computed by the program after your edit is parsed.
"""
```

### 3.2 新增：`PLAN + EDIT` 结构奖励函数

```python
def plan_edit_format_reward(completion: str) -> float:
    """
    检查 <PLAN> 和 <EDIT> 是否存在且顺序正确。
    满分 1.0；缺失标签扣分；顺序错误额外扣分。
    """
    tags = ["<PLAN>", "<EDIT>"]
    positions = [completion.find(t) for t in tags]

    if any(p == -1 for p in positions):
        missing = sum(1 for p in positions if p == -1)
        return -0.5 * missing

    if positions != sorted(positions):
        return -0.3

    return 1.0
```

### 3.3 更新：`PLAN` 的 reward 角色

这里需要明确一个关键设计决策：

- **`PLAN` 的首要角色是推理草稿 / 化学解释**
- **不是一个需要被强结构化打分的主目标**

因此，当前方案不再把“`PLAN` 是否写出了固定格式的方向预测句”作为核心 reward 目标。  
原因很简单：

- 格式正确 ≠ 推理有效
- rigid template 很容易训出“会填空的 Plan”，但不一定真的帮助 `EDIT`

当前正式采用的原则是：

- `PLAN` 尽量自由
- `EDIT` 的真实结果由 outcome reward 负责约束
- `PLAN` 只保留一个**最小可验证信号**

### 3.4 新增：`PLAN-EDIT` 一致性奖励（最小可验证信号）

```python
def plan_edit_consistency_reward(completion: str) -> float:
    """
    只检查一件事：
    PLAN 末尾的决策句是否和 <EDIT> 块中的实际动作一致。

    推荐要求 PLAN 末尾包含类似：
    Therefore I will Replace X with Y.

    然后用 regex 提取：
    - 动作类型（Replace / Add / Remove）
    - fragment 对象
    - 与 <EDIT> 中真正执行的动作是否一致
    """
    ...
```

相比旧版 `plan_direction_reward`，这个奖励的好处是：

- 对自由文本更友好
- 不要求整个 `PLAN` 都结构化
- 仍然能保证 `PLAN` 和 `EDIT` 不脱节

### 3.5 新增：程序化 outcome / `VERIFY` 奖励

```python
def verify_consistency_reward(predicted_smiles: str, prompt: str) -> float:
    """
    程序根据 <EDIT> 产生的 SMILES 重新计算真实属性，
    用于判断该 edit 是否真的朝目标方向移动，以及是否满足 target tolerance。
    """
    ...
```

在当前方案下，真正决定训练方向的主信号仍然是：

- `EDIT` 产生的结果是否有效
- 程序回填的 `VERIFY` 是否显示目标达成
- 最终属性误差（MAE / NTE）是否下降

### 3.6 新增：联合奖励函数（当前正式推荐）

```python
def compute_reward_fourstep_injected(completion: str, prompt: str, ground_truth: str) -> float:
    """
    当前正式方案下的联合奖励：
    - outcome reward 仍是主信号
    - PLAN 只保留最小一致性奖励
    - 最终 outcome 用程序化 VERIFY 结果来判断
    """
    outcome = compute_reward(completion, prompt, ground_truth)
    fmt = plan_edit_format_reward(completion)
    consistency = plan_edit_consistency_reward(completion)
    return (
        0.70 * outcome
        + 0.15 * fmt
        + 0.15 * consistency
    )
```

### 3.7 更新：reward 设计原则

在当前方案中，reward 不再默认“模型应独立写准 ANALYSE/VERIFY 的数值”，因为这些数值由程序负责。  
因此 reward 的重点应转移到：

- **结构奖励**：保证 `PLAN + EDIT` 输出稳定
- **一致性奖励**：保证 `PLAN` 最终决策与 `EDIT` 一致
- **结果奖励**：程序根据 `EDIT` 生成的 `VERIFY` 是否满足目标

建议把 reward 分解为以下几个子项：

#### (a) `r_format`

作用：检查 `<PLAN>` 和 `<EDIT>` 是否存在且顺序正确。

用途：
- 防止 RL 训练后模型连 `PLAN + EDIT` 两步都输出不稳定
- 作为最基本的 process reward

注意：
- 权重不能过大，否则模型只会在意“标签齐不齐”

#### (b) `r_analyse`

在当前方案中，`ANALYSE` 由程序前注入，因此**不再作为模型直接学习和直接奖励的核心项**。

如果仍然需要保留一个轻量分析奖励，可改为：
- 检查模型在 `PLAN` 中是否正确承接了程序注入的目标方向
- 而不是检查模型是否重新算对 `QED/LogP/MW`

#### (c) `r_plan`

当前正式方案下，**不建议再给 `PLAN` 设计复杂的文本主奖励**。  
原因是：

- 一段自由推理文本很难被稳定地 regex 打分
- 过强的格式奖励会把 `PLAN` 重新推回模板填空
- 如果 `PLAN` 真的有帮助，它的价值最终会反映在 `EDIT` 和属性结果上

因此当前推荐做法是：

- 让 `PLAN` 保持自由
- 不对整段 `PLAN` 做重文本打分
- 只保留 `PLAN-EDIT consistency` 这一项最小可验证信号

#### (d) `r_edit`

这是最硬的 outcome reward 部分，建议保持最高权重。

可检查内容：
- `<EDIT>` 格式是否合法
- 生成的 SMILES 是否合法
- 是否满足单步 fragment-level edit 的约束
- 最终属性方向是否朝目标移动
- 与 ground truth 或 target molecule 是否有一定相似性

#### (e) `r_verify`

在当前方案中，`VERIFY` 由程序回填，因此它的作用从“让模型别胡写数值”转成了“让 reward 使用真实程序结果”。

可检查内容：
- 程序回填后的 `VERIFY` 是否显示目标达成
- `EDIT` 造成的真实属性变化是否与目标方向一致
- target tolerance 是否满足

这一步仍然很重要，因为它提供了最可靠的结果监督信号。

### 3.8 推荐的 reward 聚合方式

在当前“程序注入 ANALYSE / VERIFY，模型负责 PLAN + EDIT”的方案下，建议总 reward 采用 outcome 主导形式：

```python
total_reward = (
    0.15 * r_format
    + 0.15 * r_plan_edit_consistency
    + 0.70 * r_edit_outcome
)
```

设计理由：

- `r_edit_outcome` 最高，因为真正决定分子结果的是编辑动作本身
- `r_plan_edit_consistency` 单独存在，是为了防止 `PLAN` 和 `EDIT` 脱节
- `r_format` 保底，防止 RL 把结构训坏

### 3.9 不推荐的 reward 设计

以下几种 reward 设计应避免：

1. **只奖励最终属性命中**
   - 模型会绕过过程，直接把 `PLAN` 退化成空模板

2. **只奖励 `PLAN` 是否像人话**
   - 模型可能学会“会解释但不会做”

3. **对 `PLAN` 做过强的关键词/方向硬匹配**
   - 模型容易学会堆术语或填模板，而不是做真实规划

因此，后续 RL 的核心原则应是：

- `PLAN` 不必写死
- `ANALYSE/VERIFY` 不再强行交给模型自由生成精确数值
- `PLAN` 允许是自由解释，但它必须和 `EDIT`、程序回填的 `VERIFY` 结论不冲突
- reward 重点奖励“最小一致性 + 真实结果正确”，而不是表面文风

### 3.10 新增：四步输出解析器（评估用）

```python
def parse_fourstep(completion: str) -> dict:
    """
    从四步输出中解析各块，用于计算 step_format_acc 和 plan_edit_consistency_acc。
    """
    result = {}
    for key, tag in [("analyse","ANALYSE"), ("plan","PLAN"),
                     ("edit","EDIT"),       ("verify","VERIFY")]:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", completion, re.DOTALL)
        result[key] = m.group(1).strip() if m else None

    result["format_ok"]    = all(v is not None for v in result.values())
    result["edit_smiles"]  = extract_smiles(result["edit"]) if result["edit"] else None
    result["verify_smiles"] = extract_smiles(result["verify"]) if result["verify"] else None
    return result
```

### 3.11 修改：`run_mode_b` 支持四步模式

在 `run_mode_b` 的函数签名中新增参数 `use_fourstep: bool = False`，据此切换 system prompt 和 reward 函数：

```python
def run_mode_b(
    example, renderer, sampling_client, tokenizer, ema_baseline,
    num_trajectories=2, max_turns=3, gamma=0.85,
    max_tokens=256, temperature=0.7,
    use_fourstep: bool = False,          # ← 新增
    debug=False, debug_prefix="",
):
    system_prompt  = SYSTEM_PROMPT_FOURSTEP if use_fourstep else SYSTEM_PROMPT_RL
    reward_fn      = compute_reward_fourstep_injected if use_fourstep else compute_reward
    # ...其余逻辑不变，只把这两处替换即可
```

### 3.12 advantage 计算修复（已在上一版本完成，确认保留）

```python
# 确认 run_mode_b 里的这段是新版（不再对 group_normalized 减 EMA）
if G_std < 1e-8:
    relative_advs = [G - ema_baseline.value for G in G_values]
    for G in G_values:
        ema_baseline.update_and_get(G)
else:
    relative_advs = ((G_arr - G_mean) / G_std).tolist()
    for G in G_values:
        ema_baseline.update_and_get(G)
```

---

## 4. 三级消融实验

### Abl-1（B1）：仅格式奖励

- `use_fourstep=True`
- `reward_fn = lambda c, p, gt: compute_reward(c, p, gt) * 0.7 + plan_edit_format_reward(c) * 0.3`
- 目标：`step_format_acc > 80%`
- 训练步数：5000 steps（在 SFT 四步 checkpoint 上继续）

### Abl-2（B2）：结构 + Plan-Edit 一致性奖励

- 在 B1 checkpoint 基础上继续
- `reward_fn = outcome * 0.70 + format * 0.15 + consistency * 0.15`
- 目标：`plan_edit_consistency_acc > 70%`
- 训练步数：5000 steps

### Abl-3（B3）：结果主导联合奖励

- 在 B2 checkpoint 基础上继续
- `reward_fn = compute_reward_fourstep_injected`
- 目标：NTE 低于 Baseline（Condition A）
- 训练步数：10000–20000 steps

---

## 5. 评估指标与脚本

### 指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| `step_format_acc` | `mean(parse_fourstep(c)["format_ok"])` | 四个块全部存在且顺序正确的比例 |
| `plan_edit_consistency_acc` | `mean(plan_edit_consistency_reward(c) > 0)` | Plan 末尾决策句与 Edit 实际动作一致的比例 |
| `MAE_QED/LogP/MW` | `mean(abs(pred - target))` | 各属性平均绝对误差 |
| `NTE` | `MAE_QED/1 + MAE_LogP/10 + MAE_MW/700` | 归一化总误差（越低越好） |
| `Validity` | `mean(Chem.MolFromSmiles(pred) is not None)` | 有效 SMILES 比例 |
| `Uniqueness` | 生成分子去重比例 | 衡量多样性 |

说明：

- `plan_direction_acc` 可作为离线人工分析或辅助诊断指标保留
- 但在当前正式方案中，它**不再作为主 reward 和主对比指标**

### 评估脚本

```python
def evaluate_fourstep(
    sampling_client, raw_data, renderer, tokenizer,
    n_tuples: int = 100, n_trials: int = 10,
    use_fourstep: bool = True,
) -> dict:
    """
    跑 n_tuples 个 target tuple，每个 best-of-n_trials，返回所有指标。
    """
    all_format, all_plan_cons, all_nte, all_valid = [], [], [], []
    rng = np.random.default_rng(0)
    indices = rng.choice(len(raw_data), size=n_tuples, replace=False)

    for idx in indices:
        example = raw_data[int(idx)]
        prompt  = str(example["input"])
        gt      = str(example["output"])

        best_nte = float("inf")
        best_completion = None

        for _ in range(n_trials):
            system = SYSTEM_PROMPT_FOURSTEP if use_fourstep else SYSTEM_PROMPT_RL
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ]
            prompt_input = renderer.build_generation_prompt(messages)
            result = sampling_client.sample(
                prompt=prompt_input,
                sampling_params=types.SamplingParams(
                    max_tokens=512 if use_fourstep else 256,
                    temperature=0.7,
                    stop=renderer.get_stop_sequences(),
                ),
                num_samples=1,
            ).result()
            comp = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
            comp = _normalize_completion_text(comp)

            rc = compute_reward_components(comp, prompt, gt)
            nte = rc.get("nte", float("inf"))
            if nte < best_nte:
                best_nte = nte
                best_completion = comp

        if best_completion:
            parsed = parse_fourstep(best_completion)
            all_format.append(float(parsed["format_ok"]))
            cons_score = plan_edit_consistency_reward(best_completion) if use_fourstep else 0.0
            all_plan_cons.append(float(cons_score > 0))
            all_nte.append(best_nte)
            valid = Chem.MolFromSmiles(extract_smiles(best_completion) or "") is not None
            all_valid.append(float(valid))

    return {
        "step_format_acc":         round(np.mean(all_format), 4)    if all_format   else 0.0,
        "plan_edit_consistency_acc": round(np.mean(all_plan_cons), 4) if all_plan_cons else 0.0,
        "NTE":                     round(np.mean(all_nte), 4)       if all_nte      else 0.0,
        "Validity":                round(np.mean(all_valid), 4)     if all_valid    else 0.0,
        "n_evaluated":             len(all_nte),
    }
```

---

## 6. 已知问题修复备忘

### 6.1 mean_adv 趋零问题（已修复）

**原因**：组内归一化后均值≈0，再减以归一化值为输入的 EMA（也趋近0），导致二次归零。

**修复**：`G_std >= 1e-8` 分支直接用组内归一化结果作为 advantage，不再减 EMA。EMA 只用于监控。

**确认方法**：观察训练 log 中 `mean_adv` 是否在训练早期维持在 ±0.3 以上。

### 6.2 四步输出 max_tokens 需要调大

四步格式的输出比单句 edit 长 3–4 倍，B1-B3 的 `max_tokens` 应从 128 调整为 **512**。

### 6.3 重复惩罚计数器跨轨迹污染

`_completion_counter` 和 `_fragment_counter` 是全局 defaultdict，在 `run_mode_b` 开头已经有 `clear()`。确认四步版本的 `run_mode_b` 保留了这两行：

```python
if "_completion_counter" in globals():
    _completion_counter.clear()
if "_fragment_counter" in globals():
    _fragment_counter.clear()
```

---

## 7. 实验运行顺序

```
Step 0  合成四步SFT数据集
        ├── 保留完整四步母数据集（含程序注入的 ANALYSE/VERIFY）
        ├── 用强模型补写 post-hoc chemically grounded PLAN
        ├── 必要时做生成→审稿→修订或直接丢弃坏样本
        └── 产出：fourstep_sft_master.jsonl

Step 1  导出正式训练数据
        ├── 从母数据集中导出两步训练样本
        ├── user = 原始 prompt + 注入 ANALYSE
        ├── assistant = PLAN + EDIT
        └── 产出：fourstep_plan_edit_sft.jsonl

Step 2  四步SFT冷启动
        ├── 在现有 SFT checkpoint 基础上
        ├── 训练数据：fourstep_plan_edit_sft.jsonl
        ├── steps：3000–5000
        └── 产出：checkpoint "fourstep-sft-init"

Step 3  Condition A（Baseline）评估
        ├── 使用已有的无四步 GRPO checkpoint
        ├── 运行 evaluate_fourstep(..., use_fourstep=False)
        └── 记录：NTE, Validity 作为基线数字

Step 4  Condition B1（结构奖励）训练
        ├── 起点：fourstep-sft-init
        ├── reward：outcome + 轻量 format
        ├── steps：5000
        ├── 目标：step_format_acc > 80%
        └── 产出：checkpoint "fourstep-b1"

Step 5  Condition B2（+ Plan-Edit 一致性奖励）训练
        ├── 起点：fourstep-b1
        ├── reward：0.70 * outcome + 0.15 * format + 0.15 * consistency
        ├── steps：5000
        ├── 目标：plan_edit_consistency_acc > 70%
        └── 产出：checkpoint "fourstep-b2"

Step 6  Condition B3（结果主导联合奖励）训练
        ├── 起点：fourstep-b2
        ├── reward：compute_reward_fourstep_injected
        ├── steps：10000–20000
        ├── 目标：NTE < Condition A
        └── 产出：checkpoint "fourstep-b3-final"

Step 7  所有条件最终评估
        ├── 统一跑 evaluate_fourstep，n_tuples=100，n_trials=10
        └── 填入 Table 4
```

---

## 8. 论文对应关系

| 实验结果 | 对应论文位置 |
|---------|-------------|
| Table 4（四步消融） | §7 新增子节 "7.4 Ablation of Four-Step Reasoning Chain" |
| `plan_direction_acc` | §4.2 "Plan step is the most critical from a causal standpoint" 的量化验证 |
| B3 NTE vs Baseline NTE | §4.2 Table 1 的实证支撑（因果阶梯 L2/L3 的实际收益） |
| `step_format_acc` 随训练迭代上升曲线 | Figure 新增：格式学习曲线，说明 SFT 冷启动的必要性 |

### 预期结论写法（待填数字）

> We find that introducing the four-step reasoning chain yields a **X%** relative reduction
> in NTE compared to the baseline (0.163 → TBD), with `plan_direction_acc` reaching TBD%—
> significantly above the random baseline of 50%—indicating that the model has learned
> to perform genuine counterfactual predictions at the Plan step rather than merely
> mimicking surface-level edit patterns.

---

*最后更新：根据 `multiStepRL_nonsubprocess__1_.ipynb` 和论文 `causal_molgen.pdf` 整理*
