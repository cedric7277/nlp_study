
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 实验配置
# ============================================================

@dataclass
class Config:

    # 随机种子
    seed: int = 42

    # 数学题：
    # a + b
    # a,b ∈ [0,20]
    max_operand: int = 20

    # 最大答案
    max_answer: int = 40

    # GRPO：
    # 每一道题生成多少个候选答案
    group_size: int = 8

    # 每个 Batch 多少道数学题
    batch_size: int = 32

    # 训练轮数
    epochs: int = 300

    # 学习率
    learning_rate: float = 0.003

    # PPO / GRPO clipping
    clip_epsilon: float = 0.2

    # KL penalty
    beta_kl: float = 0.02

    # entropy bonus
    entropy_coef: float = 0.005

    # 每多少轮测试一次
    eval_interval: int = 20


CFG = Config()


# ============================================================
# 2. 数学题生成器
# ============================================================

def generate_problem():

    a = random.randint(0, CFG.max_operand)
    b = random.randint(0, CFG.max_operand)

    return a, b


def get_correct_answer(problem):

    a, b = problem

    return a + b


def format_problem(problem):

    a, b = problem

    return f"{a} + {b} = ?"


# ============================================================
# 3. Reward Function
# ============================================================

def reward_function(problem, model_answer):

    """
    数学 Reward：

    完全正确：
        reward = 1

    与答案相差 1：
        reward = 0.2

    其他：
        reward = 0

    在真实 LLM 数学强化学习中，可以替换为：

        SymPy verifier
        GSM8K answer checker
        Math verifier
        Judge Model
    """

    correct = get_correct_answer(problem)

    difference = abs(model_answer - correct)

    if difference == 0:

        return 1.0

    elif difference == 1:

        return 0.2

    else:

        return 0.0


# ============================================================
# 4. Policy Model
# ============================================================

class MathPolicy(nn.Module):

    """
    一个极简 Policy Network。

    输入：

        a + b

    输出：

        P(answer | problem)

    例如：

        输入：

            7 + 8

        输出：

            P(0)
            P(1)
            ...
            P(15)
            ...
            P(40)

    可以把它理解成：

        pi_theta(answer | prompt)

    在真实 GRPO 中，这里就是：

        Qwen
        Llama
        DeepSeek
        Gemma
    """

    def __init__(self):

        super().__init__()

        embedding_dim = 32

        hidden_dim = 128

        # 数字 embedding
        self.embedding = nn.Embedding(
            CFG.max_operand + 1,
            embedding_dim
        )

        # Policy Network
        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                CFG.max_answer + 1
            )
        )

    def forward(self, problems):

        # problems shape:
        # [batch,2]

        a = problems[:, 0]

        b = problems[:, 1]

        a_embedding = self.embedding(a)

        b_embedding = self.embedding(b)

        x = torch.cat(
            [a_embedding, b_embedding],
            dim=-1
        )

        logits = self.network(x)

        return logits


# ============================================================
# 5. GRPO Group Sampling
# ============================================================

@torch.no_grad()
def sample_group(policy, problems):

    """
    GRPO：

    对同一个 Prompt 生成多个回答。

    例如：

        Prompt：

            7 + 8 = ?

        Group：

            Answer1 = 13
            Answer2 = 15
            Answer3 = 17
            Answer4 = 15
            Answer5 = 14
            Answer6 = 12
            Answer7 = 15
            Answer8 = 20

    然后比较这些答案的 Reward。
    """

    logits = policy(problems)

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    distribution = torch.distributions.Categorical(
        probabilities
    )

    answers = []

    log_probs = []

    for _ in range(CFG.group_size):

        answer = distribution.sample()

        log_prob = distribution.log_prob(answer)

        answers.append(answer)

        log_probs.append(log_prob)

    answers = torch.stack(
        answers,
        dim=1
    )

    log_probs = torch.stack(
        log_probs,
        dim=1
    )

    return answers, log_probs


# ============================================================
# 6. Reward Calculation
# ============================================================

def calculate_rewards(
    problems,
    sampled_answers
):

    batch_size = problems.shape[0]

    group_size = sampled_answers.shape[1]

    rewards = torch.zeros(
        batch_size,
        group_size
    )

    for i in range(batch_size):

        problem = (
            int(problems[i][0]),
            int(problems[i][1])
        )

        for j in range(group_size):

            answer = int(
                sampled_answers[i][j]
            )

            reward = reward_function(
                problem,
                answer
            )

            rewards[i][j] = reward

    return rewards


# ============================================================
# 7. GRPO Relative Advantage
# ============================================================

def calculate_advantages(rewards):

    """
    GRPO 的核心思想之一：

    不训练 Value Model。

    而是在同一道题生成的一组答案内部比较。

    Advantage：

             reward_i - mean(reward)
        A = --------------------------
                  std(reward)

    ----------------------------------

    例如：

    同一道数学题：

        Answer      Reward

        13           0
        15           1
        17           0
        15           1

    平均 Reward：

        0.5

    那么：

        正确答案 Advantage > 0

        错误答案 Advantage < 0

    模型会：

        提高正确答案概率

        降低错误答案概率
    """

    mean_reward = rewards.mean(
        dim=1,
        keepdim=True
    )

    std_reward = rewards.std(
        dim=1,
        keepdim=True,
        unbiased=False
    )

    advantages = (

        rewards - mean_reward

    ) / (

        std_reward + 1e-6

    )

    return advantages


# ============================================================
# 8. GRPO Update
# ============================================================

def grpo_update(
    policy,
    reference_policy,
    optimizer,
    problems,
    answers,
    old_log_probs,
    advantages
):

    """
    GRPO / PPO Style Objective

    probability ratio：

                    pi_theta
        ratio = ----------------
                    pi_old


    clipped objective：

        min(

            ratio * Advantage,

            clip(ratio) * Advantage

        )


    同时加入：

        KL Penalty

    防止 Policy 与 Reference Model 偏离过大。
    """

    logits = policy(problems)

    log_probs_all = F.log_softmax(
        logits,
        dim=-1
    )

    probs_all = log_probs_all.exp()

    batch_size = problems.shape[0]

    group_size = answers.shape[1]

    # 扩展维度
    expanded_log_probs = (

        log_probs_all

        .unsqueeze(1)

        .expand(
            -1,
            group_size,
            -1
        )
    )

    # 找到采样答案对应的 log probability
    new_log_probs = torch.gather(

        expanded_log_probs,

        dim=2,

        index=answers.unsqueeze(-1)

    ).squeeze(-1)

    # PPO probability ratio
    ratio = torch.exp(

        new_log_probs - old_log_probs

    )

    # unclipped objective
    objective1 = (

        ratio * advantages

    )

    # clipped objective
    objective2 = (

        torch.clamp(

            ratio,

            1 - CFG.clip_epsilon,

            1 + CFG.clip_epsilon

        )

        * advantages

    )

    # GRPO Policy Loss
    policy_loss = (

        -torch.min(

            objective1,

            objective2

        ).mean()

    )

    # ========================================================
    # KL Divergence
    # ========================================================

    with torch.no_grad():

        reference_logits = reference_policy(
            problems
        )

        reference_log_probs = F.log_softmax(

            reference_logits,

            dim=-1

        )

    kl_divergence = (

        probs_all

        * (

            log_probs_all

            - reference_log_probs

        )

    ).sum(

        dim=-1

    ).mean()

    # ========================================================
    # Entropy
    # ========================================================

    entropy = -(

        probs_all

        * log_probs_all

    ).sum(

        dim=-1

    ).mean()

    # ========================================================
    # Total Loss
    # ========================================================

    total_loss = (

        policy_loss

        + CFG.beta_kl * kl_divergence

        - CFG.entropy_coef * entropy

    )

    optimizer.zero_grad()

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(

        policy.parameters(),

        1.0

    )

    optimizer.step()

    return {

        "loss":
            total_loss.item(),

        "policy_loss":
            policy_loss.item(),

        "kl":
            kl_divergence.item(),

        "entropy":
            entropy.item()

    }


# ============================================================
# 9. 模型评估
# ============================================================

@torch.no_grad()
def evaluate(
    policy,
    test_samples=500
):

    correct_count = 0

    for _ in range(test_samples):

        problem = generate_problem()

        input_tensor = torch.tensor(

            [[

                problem[0],

                problem[1]

            ]],

            dtype=torch.long

        )

        logits = policy(
            input_tensor
        )

        predicted_answer = int(

            logits.argmax(
                dim=-1
            ).item()

        )

        correct_answer = get_correct_answer(
            problem
        )

        if predicted_answer == correct_answer:

            correct_count += 1

    accuracy = (

        correct_count

        / test_samples

    )

    return accuracy


# ============================================================
# 10. 显示数学题案例
# ============================================================

@torch.no_grad()
def show_examples(
    policy,
    number=10
):

    print()

    print(
        "=" * 60
    )

    print(
        "数学题测试"
    )

    print(
        "=" * 60
    )

    for _ in range(number):

        problem = generate_problem()

        input_tensor = torch.tensor(

            [[

                problem[0],

                problem[1]

            ]],

            dtype=torch.long

        )

        logits = policy(
            input_tensor
        )

        prediction = int(

            logits.argmax(
                dim=-1
            ).item()

        )

        gold = get_correct_answer(
            problem
        )

        if prediction == gold:

            result = "✓"

        else:

            result = "✗"

        print(

            f"{format_problem(problem):12s}"

            f" 模型答案={prediction:2d}"

            f" 正确答案={gold:2d}"

            f" {result}"

        )


# ============================================================
# 11. 查看 GRPO Group
# ============================================================

def show_grpo_group(
    problems,
    answers,
    rewards,
    advantages
):

    """
    用于演示：

    展示一道题生成的多个答案，
    以及对应 Reward / Advantage。
    """

    problem = (

        int(problems[0][0]),

        int(problems[0][1])

    )

    print()

    print(
        "GRPO Group 示例"
    )

    print(
        "-" * 50
    )

    print(
        "题目：",
        format_problem(problem)
    )

    print()

    print(

        f"{'Sample':<10}"

        f"{'Answer':<10}"

        f"{'Reward':<10}"

        f"{'Advantage':<10}"

    )

    for i in range(
        CFG.group_size
    ):

        print(

            f"{i + 1:<10}"

            f"{int(answers[0][i]):<10}"

            f"{float(rewards[0][i]):<10.2f}"

            f"{float(advantages[0][i]):<10.2f}"

        )

    print(
        "-" * 50
    )


# ============================================================
# 12. GRPO Training
# ============================================================

def train():

    # 固定随机种子
    random.seed(
        CFG.seed
    )

    torch.manual_seed(
        CFG.seed
    )

    # ========================================================
    # 创建 Policy Model
    # ========================================================

    policy = MathPolicy()

    # ========================================================
    # Reference Model
    # ========================================================

    reference_policy = MathPolicy()

    reference_policy.load_state_dict(

        policy.state_dict()

    )

    # Reference Model 不训练
    reference_policy.eval()

    for parameter in reference_policy.parameters():

        parameter.requires_grad = False

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(

        policy.parameters(),

        lr=CFG.learning_rate

    )

    # ========================================================
    # 训练前 Accuracy
    # ========================================================

    initial_accuracy = evaluate(
        policy
    )

    print()

    print(
        "=" * 70
    )

    print(
        "GRPO Math Reinforcement Learning Demo"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "实验目标："
    )

    print(
        "使用 GRPO 强化学习提升模型数学题正确率"
    )

    print()

    print(
        f"训练前 Accuracy = "
        f"{initial_accuracy:.2%}"
    )

    print()

    # ========================================================
    # Training Loop
    # ========================================================

    for epoch in range(

        1,

        CFG.epochs + 1

    ):

        # ----------------------------------------------------
        # Step 1
        #
        # 随机生成 Batch 数学题
        # ----------------------------------------------------

        batch = [

            generate_problem()

            for _ in range(
                CFG.batch_size
            )

        ]

        problems = torch.tensor(

            batch,

            dtype=torch.long

        )

        # ----------------------------------------------------
        # Step 2
        #
        # 每一道题生成多个回答
        # ----------------------------------------------------

        answers, old_log_probs = sample_group(

            policy,

            problems

        )

        # ----------------------------------------------------
        # Step 3
        #
        # Reward Model / Verifier
        # ----------------------------------------------------

        rewards = calculate_rewards(

            problems,

            answers

        )

        # ----------------------------------------------------
        # Step 4
        #
        # Group Relative Advantage
        # ----------------------------------------------------

        advantages = calculate_advantages(

            rewards

        )

        # 第一轮展示一次 Group
        if epoch == 1:

            show_grpo_group(

                problems,

                answers,

                rewards,

                advantages

            )

        # ----------------------------------------------------
        # Step 5
        #
        # GRPO Update
        # ----------------------------------------------------

        stats = grpo_update(

            policy,

            reference_policy,

            optimizer,

            problems,

            answers,

            old_log_probs,

            advantages

        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (

            epoch == 1

            or

            epoch % CFG.eval_interval == 0

        ):

            accuracy = evaluate(
                policy
            )

            average_reward = (

                rewards.mean().item()

            )

            print(

                f"Epoch {epoch:03d}"

                f" | Reward={average_reward:.3f}"

                f" | Accuracy={accuracy:.2%}"

                f" | Loss={stats['loss']:.4f}"

                f" | KL={stats['kl']:.4f}"

                f" | Entropy={stats['entropy']:.3f}"

            )

    # ========================================================
    # Final Evaluation
    # ========================================================

    final_accuracy = evaluate(

        policy,

        test_samples=1000

    )

    print()

    print(
        "=" * 70
    )

    print(
        "训练完成"
    )

    print(
        "=" * 70
    )

    print()

    print(

        f"训练前 Accuracy: "
        f"{initial_accuracy:.2%}"

    )

    print(

        f"训练后 Accuracy: "
        f"{final_accuracy:.2%}"

    )

    improvement = (

        final_accuracy

        - initial_accuracy

    )

    print(

        f"Accuracy 提升: "
        f"{improvement:+.2%}"

    )

    # 展示最终效果
    show_examples(
        policy,
        number=10
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train()
