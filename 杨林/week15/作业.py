"""
Multi-SubAgent Parallel Execution Demo
======================================

功能：
1. MainAgent 接收复杂目标
2. 将目标拆解成多个独立子任务
3. 将任务下发给多个 SubAgent
4. 使用 asyncio 并行执行
5. 汇总所有 SubAgent 的执行结果

运行：
    python multi_subagent_demo.py

说明：
    本示例只使用 Python 标准库，不需要安装第三方依赖。
    当前通过 asyncio.sleep 模拟 LLM / API / 工具调用。
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import List


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Task:
    """MainAgent 下发给 SubAgent 的任务。"""

    id: int
    role: str
    instruction: str


@dataclass
class AgentResult:
    """SubAgent 执行完成后返回的结果。"""

    task_id: int
    role: str
    output: str
    elapsed: float


# ============================================================
# SubAgent
# ============================================================

class SubAgent:
    """
    子 Agent。

    每个 SubAgent 可以独立执行一个任务。

    实际项目中 run() 可以替换成：
    - LLM API
    - 搜索工具
    - 数据库查询
    - Python / Shell
    - MCP 服务
    - 其他业务 API
    """

    def __init__(self, name: str):
        self.name = name

    async def run(self, task: Task) -> AgentResult:
        """异步执行一个任务。"""

        start_time = time.perf_counter()

        print(
            f"[START] {self.name:<12} "
            f"| role={task.role:<10} "
            f"| {task.instruction}"
        )

        # ----------------------------------------------------
        # 模拟外部 LLM / API 调用
        #
        # 因为使用 await，所以等待期间不会阻塞其他 SubAgent。
        # ----------------------------------------------------

        await asyncio.sleep(
            random.uniform(1.5, 3.5)
        )

        # 模拟 Agent 的执行结果
        output = self._mock_execute(task)

        elapsed = time.perf_counter() - start_time

        print(
            f"[DONE ] {self.name:<12} "
            f"| role={task.role:<10} "
            f"| elapsed={elapsed:.2f}s"
        )

        return AgentResult(
            task_id=task.id,
            role=task.role,
            output=output,
            elapsed=elapsed,
        )

    @staticmethod
    def _mock_execute(task: Task) -> str:
        """
        模拟不同角色的 Agent 返回不同结果。

        实际系统中，这部分通常由 LLM 完成。
        """

        mock_results = {

            "researcher": (
                "调研结果：多 Agent 系统常见模式包括 "
                "Supervisor-Worker、Planner-Executor "
                "以及多个专家 Agent 协作。"
                "对于彼此独立的任务，可以采用并行执行，"
                "从而减少系统整体等待时间。"
            ),

            "architect": (
                "架构设计：MainAgent 负责理解目标、拆解任务、"
                "调度 SubAgent 和聚合结果；"
                "SubAgent 负责执行具体任务。"
                "各 SubAgent 之间保持相对独立，"
                "通过统一的数据结构进行任务和结果交换。"
            ),

            "coder": (
                "实现方案：使用 Python asyncio 创建异步任务，"
                "通过 asyncio.gather 同时等待多个 SubAgent 完成，"
                "实现多个任务的并发执行。"
            ),

            "reviewer": (
                "评审建议：生产环境应增加最大并发限制、"
                "超时控制、异常隔离、自动重试、日志追踪、"
                "结构化输出以及 Token / Cost 限制。"
            ),
        }

        return mock_results.get(
            task.role,
            f"{task.role} 已完成任务：{task.instruction}",
        )


# ============================================================
# MainAgent
# ============================================================

class MainAgent:
    """
    主 Agent / Supervisor。

    MainAgent 负责：

    1. 接收复杂目标
    2. 拆解任务
    3. 选择 SubAgent
    4. 下发任务
    5. 并行执行
    6. 汇总结果
    """

    def __init__(self, subagent_count: int = 4):

        self.subagents = [
            SubAgent(
                name=f"subagent-{i + 1}"
            )
            for i in range(subagent_count)
        ]

    # --------------------------------------------------------
    # Step 1：任务拆解
    # --------------------------------------------------------

    def plan(self, goal: str) -> List[Task]:
        """
        将复杂目标拆成多个独立任务。

        Demo 为了不依赖 LLM，使用固定规则拆解。

        实际 Agent 系统中，这一步可以让 LLM 动态生成：
            [
                Task(...),
                Task(...),
                ...
            ]
        """

        print("\n" + "=" * 70)
        print("STEP 1 - MainAgent 拆解任务")
        print("=" * 70)

        print(f"用户目标：{goal}\n")

        tasks = [

            Task(
                id=1,
                role="researcher",
                instruction=(
                    "调研多 Agent 并行执行的"
                    "常见设计模式"
                ),
            ),

            Task(
                id=2,
                role="architect",
                instruction=(
                    "设计 MainAgent + SubAgent "
                    "整体系统架构"
                ),
            ),

            Task(
                id=3,
                role="coder",
                instruction=(
                    "设计 Python 异步并发执行方案"
                ),
            ),

            Task(
                id=4,
                role="reviewer",
                instruction=(
                    "分析该方案在生产环境中的"
                    "风险和优化方向"
                ),
            ),
        ]

        for task in tasks:

            print(
                f"Task {task.id}: "
                f"[{task.role}] "
                f"{task.instruction}"
            )

        return tasks

    # --------------------------------------------------------
    # Step 2：并行调度
    # --------------------------------------------------------

    async def dispatch_parallel(
        self,
        tasks: List[Task],
    ) -> List[AgentResult]:
        """
        将多个任务分配给多个 SubAgent。

        这里是整个 Demo 最关键的部分：

            asyncio.gather()

        多个 SubAgent 会同时执行，
        而不是一个执行完成后再执行下一个。
        """

        print("\n" + "=" * 70)
        print("STEP 2 - MainAgent 并行下发任务")
        print("=" * 70)

        coroutines = []

        # 将任务分配给不同 SubAgent
        for index, task in enumerate(tasks):

            agent = self.subagents[
                index % len(self.subagents)
            ]

            coroutine = agent.run(task)

            coroutines.append(coroutine)

        # ====================================================
        # 核心：
        #
        # 同时执行所有 SubAgent
        # ====================================================

        results = await asyncio.gather(
            *coroutines
        )

        return results

    # --------------------------------------------------------
    # Step 3：结果聚合
    # --------------------------------------------------------

    def synthesize(
        self,
        goal: str,
        results: List[AgentResult],
    ) -> str:
        """
        汇总所有 SubAgent 返回的结果。
        """

        print("\n" + "=" * 70)
        print("STEP 3 - MainAgent 汇总结果")
        print("=" * 70)

        sections = []

        # 按 Task ID 排序，避免并行执行导致结果顺序变化
        for result in sorted(
            results,
            key=lambda item: item.task_id,
        ):

            sections.append(
                f"[{result.role}]\n"
                f"{result.output}\n"
                f"执行耗时："
                f"{result.elapsed:.2f}s"
            )

        final_answer = (
            f"目标：{goal}\n\n"
            + "\n\n".join(sections)
            + "\n\n"
            + "最终结论：该系统通过 MainAgent "
              "将复杂目标拆分为多个相互独立的子任务，"
              "再将任务下发给不同 SubAgent 并行执行，"
              "最后统一汇总执行结果。"
              "这是一种典型的 Supervisor-Worker "
              "多 Agent 协作模式。"
        )

        return final_answer

    # --------------------------------------------------------
    # 完整 Agent 工作流
    # --------------------------------------------------------

    async def run(
        self,
        goal: str,
    ) -> str:

        total_start = time.perf_counter()

        # 1. 拆任务
        tasks = self.plan(goal)

        # 2. 并行执行
        results = await self.dispatch_parallel(
            tasks
        )

        # 3. 汇总结果
        final_answer = self.synthesize(
            goal,
            results,
        )

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        # 如果采用串行执行，需要的理论耗时
        serial_elapsed = sum(
            result.elapsed
            for result in results
        )

        # ----------------------------------------------------
        # 输出最终结果
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("STEP 4 - 最终结果")
        print("=" * 70)

        print(final_answer)

        print("\n" + "-" * 70)

        print(
            f"实际并行总耗时："
            f"{total_elapsed:.2f}s"
        )

        print(
            f"若采用串行执行："
            f"约 {serial_elapsed:.2f}s"
        )

        print(
            "并行执行时，总耗时主要取决于"
            "执行最慢的 SubAgent，"
            "而不是所有 SubAgent "
            "执行时间之和。"
        )

        return final_answer


# ============================================================
# 程序入口
# ============================================================

async def main() -> None:

    # 创建 MainAgent
    main_agent = MainAgent(
        subagent_count=4
    )

    # 给 MainAgent 一个复杂目标
    goal = (
        "实现一个可以向多个 SubAgent 下发任务，"
        "并行完成多项工作的 Agent 系统"
    )

    # 执行
    await main_agent.run(goal)


if __name__ == "__main__":

    asyncio.run(
        main()
    )
