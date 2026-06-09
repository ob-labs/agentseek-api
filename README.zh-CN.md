# AgentSeek API

[English](README.md) | **中文**

> [!WARNING]
> 本项目正在积极开发中，**尚未达到生产可用状态**。
> 欢迎提交 Pull Request 来修复 Bug 或贡献增强功能！

通过 FastAPI 运行时承载 LangGraph 与 LangChain 应用，并提供独立的
`agentseek-api` CLI。

> [!NOTE]
> AgentSeek API 主要以
> [Agent Protocol](https://github.com/langchain-ai/agent-protocol) 作为对外
> 兼容性的参考。当前的运行时已经覆盖了核心的 thread、run、cron、streaming
> 以及 protocol-v2 事件流，部分协议接口仍在补齐过程中。Agent 资源通过
> `/assistants`、直接的 `/agents` 别名、Streamable HTTP MCP 以及
> LangSmith 风格的 A2A 端点对外暴露。这是对核心 agent-server 接口实用的
> OSS 对齐实现，并不等同于完整的 LangSmith Agent Server 对齐。

当前版本边界：

- 已实现：assistants、threads、runs、crons、streaming、Store API、MCP
  以及 A2A
- 明确不实现：分布式运行时对齐、assistant 子图查看，以及 assistant 版本
  晋升流程

## 🚀 快速上手

### 前置条件

- Python 3.12+
- `uv`

### 选择合适的本地开发循环

| 工作流 | 适用场景 | 推荐命令 |
| --- | --- | --- |
| `langgraph dev` | 用于图原型开发或 Studio 实验，需要最快的 mock 或内存版本本地 API 循环。 | `langgraph dev` |
| `agentseek-api dev` | 需要真实的 AgentSeek API 接口，配合真实的 MySQL 系列 / seekdb / OceanBase 风格持久化、鉴权以及 Docker/runtime 行为。 | `uv run agentseek-api dev` |

当你不需要真实后端验证时，使用 `langgraph dev`；当你希望验证本仓库实际
提供的 API 契约时，使用 `agentseek-api dev`。

### 1. 安装依赖

```bash
uv sync
```

### 将 seekdb embed 配置为默认后端（推荐）

最快获得真实后端的方式是 **seekdb embed** — 一个进程内嵌入式 SeekDB 实例，
无需 Docker 或独立进程：

```bash
uv sync --dev --extra embedded
```

然后以嵌入模式启动 API：

```bash
SEEKDB_EMBED=true uv run agentseek-api dev
```

数据默认存储在 `~/.agentseek/seekdb_data`，可通过 `SEEKDB_EMBED_DIR` 修改。

<details>
<summary>改用 seekdb Docker 容器</summary>

```bash
docker run -d --name seekdb-dev \
  -p 2881:2881 -p 2886:2886 \
  oceanbase/seekdb:latest
```

等待容器健康后，导出环境变量：

```bash
export OCEANBASE_HOST=127.0.0.1
export OCEANBASE_PORT=2881
export OCEANBASE_USER=root
export OCEANBASE_PASSWORD=
export OCEANBASE_DB_NAME=seekdb
```

</details>

<details>
<summary>改用 OceanBase 实例</summary>

以 mini 模式启动 OceanBase-CE 容器：

```bash
docker run -d --name ob-dev \
  -e MODE=mini \
  -p 2881:2881 \
  oceanbase/oceanbase-ce:latest
```

OceanBase-CE mini 模式需要几分钟完成初始化。就绪后导出环境变量：

```bash
export OCEANBASE_HOST=127.0.0.1
export OCEANBASE_PORT=2881
export OCEANBASE_USER=root@test
export OCEANBASE_PASSWORD=
export OCEANBASE_DB_NAME=seekdb
```

创建数据库：

```bash
mysql -h 127.0.0.1 -P 2881 -u root@test -e "CREATE DATABASE IF NOT EXISTS seekdb"
```

</details>

### 2. 创建配置文件

`agentseek-api` 会按以下顺序查找配置：

1. `AGENTSEEK_GRAPHS`，如果它指向一个存在的文件
2. `agentseek.json`
3. `langgraph.json`

最小化的 `langgraph.json`：

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  }
}
```

### 3. 启动本地 API

```bash
uv run agentseek-api dev
```

需要显式指定配置时：

```bash
uv run agentseek-api dev --config ./langgraph.json
```

服务启动就绪后会打印本地 API、文档以及 Studio 的 URL：

```text
> Ready!
>
> - API: http://localhost:2024
>
> - Docs: http://localhost:2024/docs
>
> - LangSmith Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 4. 验证服务是否启动

```bash
curl http://127.0.0.1:2024/health
curl http://127.0.0.1:2024/info
curl http://127.0.0.1:2024/openapi.json
```
### 5. 使用 LangGraph SDK 进行测试

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024/")

async def main():
    # 列出所有 assistant
    assistants = await client.assistants.search(graph_id="agent")

    # 我们会为你在配置中注册的每个图自动创建一个 assistant。
    agent = assistants[0]

    # 创建一个新的 thread
    thread = await client.threads.create()

    # 启动一个流式 run
    input = {"messages": [{"role": "human", "content": "hello?"}]}
    async for chunk in client.runs.stream(
        thread["thread_id"], agent["assistant_id"], input=input
    ):
        print(chunk)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

## 🧰 CLI

本包会安装名为 `agentseek-api` 的独立可执行文件，不会安装顶级的
`agentseek` 二进制，从而把这个命名空间留给上层父级 CLI 使用。

```bash
agentseek-api <command> [arguments]
```

在本仓库内运行时，请使用 `uv run agentseek-api ...`。

### 命令列表

| 命令 | 作用 |
| --- | --- |
| `dev` | 启动带热重载的本地开发 API。 |
| `serve` | 启动不带热重载的 API，适用于容器或冒烟测试。 |
| `worker` | 启动基于 Redis 的 run worker。 |
| `dockerfile` | 为当前配置生成运行时 Dockerfile。 |
| `build` | 为当前配置构建 Docker 镜像。 |
| `up` | 为当前配置启动本地 Docker 运行时。 |
| `version` | 打印已安装的包版本。 |

### 共用参数

- `-c, --config PATH`：显式指定 `agentseek.json`、`langgraph.json` 或
  manifest 路径
- `--env-file PATH`：加载到运行时环境的 dotenv 风格文件

### 常用示例

```bash
uv run agentseek-api dev
uv run agentseek-api serve --config ./langgraph.json --port 8080
uv run agentseek-api worker --config ./langgraph.json
uv run agentseek-api dockerfile --config ./langgraph.json ./Dockerfile.agentseek
uv run agentseek-api build --config ./langgraph.json -t agentseek-api:dev
uv run agentseek-api up --config ./langgraph.json --port 8123 --wait
uv run agentseek-api version
```

### 命令说明

- `dev`
  - 默认 host：`127.0.0.1`
  - 默认 port：`2024`
  - 使用 `--no-reload` 关闭热重载
  - 使用 `--no-browser` 阻止自动启动 Studio
  - 使用 `--studio-url` 指向其他 LangSmith / Studio 源
- `serve`
  - 与 `dev` 共用相同的 host 与 port 参数
- `worker`
  - 需要 `EXECUTOR_BACKEND=redis`
  - 使用 `REDIS_URL` 及下文所列的队列键
  - Redis 持久化执行当前使用单个活跃 worker lease
  - Worker 重启后，run 与 thread 的流回放会从持久化状态继续
- `scheduler`
  - 触发所有到期的持久化 cron 任务
  - 在启用 cron 时，与 API 服务及 worker 一起运行
- `build`
  - 使用 `-t, --tag` 设置镜像 tag
  - 支持 `--platform`、`--pull`、`--no-pull`
- `up`
  - 支持 `--wait`、`--image`、`--base-image`、`--postgres-uri`、
    `--recreate`、`--no-recreate`

部分仿照 LangGraph CLI 的参数会为了命令兼容性被解析，但当对应运行时
行为还未实现时会被直接拒绝。对于 mock、内存或 tunnel 化的本地工作流，
建议继续使用 `langgraph dev`。

## ✨ 功能特性

- ⚙️ 独立 CLI，同时支持作为子命令嵌入到父级 CLI 中
- 🔌 通过 `agentseek.json`、`langgraph.json` 或 `AGENTSEEK_GRAPHS`
  实现 manifest 驱动的图加载
- 🌊 基于 `message_chunk` 事件的 SSE 流式输出
- 🧰 通过 Streamable HTTP 把注册的图暴露为 MCP 工具
- 🤝 A2A assistant 端点，支持 agent-card 发现、流式以及任务查询/取消
- 🧵 Thread、run、wait、cancel、history、state 以及 protocol-v2 流式
  完整流程
- ⏰ 持久化 cron API，以及面向无状态与绑定 thread 的 run 的 scheduler 派发
- 🤖 Agent 资源同时通过 `/assistants` 和 `/agents` 暴露
- 🧑‍💻 通过 `POST /threads/{thread_id}/runs/{run_id}/resume`
  支持 Human-in-the-loop 恢复
- 🗄️ 通过 `langchain-oceanbase` 提供 seekdb / OceanBase 优先的
  checkpoint 持久化
- 📦 基于 Redis 的持久化执行，配合独立的 worker 进程
- ♻️ 持久化 run 与 thread 流回放，支持重启后恢复
- 🔐 `noop` 鉴权与自定义鉴权后端
- 🐳 Dockerfile 生成、镜像构建以及本地 Docker 运行时辅助命令
- 🧪 覆盖 MySQL、seekdb、OceanBase 与 Redis 运行时路径的真实后端 CI
- 🧪 面向真实 provider 的手动流式校验，提供端到端 SSE 证明

## 🎯 兼容性范围

可以把 AgentSeek API 看作一套实用的、OSS 兼容的 Agent Server 风格
应用核心。

- 已交付：assistant CRUD、thread/run 生命周期 API、可恢复的 SSE 流、
  cron API 与 scheduler 派发、Store API、MCP、A2A、基于 Redis 的持久化
  执行，以及 Docker/runtime 辅助命令
- 有意未实现：分布式运行时编排对齐、完整的 assistant 版本管理、
  assistant 子图查看，以及超出核心 CRUD 与 schema 流程之外的完整
  assistant 辅助能力对齐

## 🚚 部署角色

启用 cron 的部署会运行三个长期存活的角色：

- API：提供 `/assistants`、`/threads`、`/runs`、`/runs/crons`、`/info`
  以及其他 HTTP 接口
- Worker：执行 Redis 队列中的 run，并在重启后恢复持久化的流状态
- Scheduler：抢占到期的 cron 任务并把对应的 run 提交给运行时

针对真实后端进行本地开发时，请使用 `uv run agentseek-api dev`。如果只
需要 mock 或内存版本进行图迭代，请改用 `langgraph dev`。如需持久化的
cron 执行，请让 API 服务、worker 与 scheduler 共用同一套数据库与
Redis 实例同时运行。

## 🗂️ 配置

图引用可以指向：

- 模块符号，例如 `package.module:graph`
- 相对路径的 Python 文件，例如 `./graph.py:graph`
- 已编译的图对象
- 无参 builder
- `build_graph(checkpointer=...)` 形式的函数
- 接收 config 字典的工厂函数

常用配置字段：

- `dependencies`：会被安装到生成的 Docker 镜像中的本地包路径
- `graphs`：graph id 到图引用的映射
- `env`：可以是 dotenv 文件路径，或者一个标量环境变量对象
- `auth.path`：自定义鉴权后端引用
- `auth.openapi`：用于鉴权的 OpenAPI `securitySchemes` 与 `security`
  元数据
- `auth.disable_studio_auth`：关闭下文描述的 Studio 鉴权绕过
- `http.disable_mcp`：关闭 MCP 端点
- `http.disable_a2a`：关闭 A2A 端点及 agent-card 发现路由
- `base_image`、`python_version`、`image_distro`、`pip_config_file`、
  `dockerfile_lines`：Docker 构建自定义字段

CLI 层会尽量容忍 LangGraph 在端点级别使用的配置键，例如 `http` 与
`api_version`。Store 配置会被 HTTP Store API 以及注入的 LangGraph
`BaseStore` 运行时用于 TTL 与语义检索。本仓库使用 PyPI 上发布的
`langchain-oceanbase==0.5.1` 包。

配置驱动的自定义鉴权可以放在 `agentseek.json` 或 `langgraph.json` 中：

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "auth": {
    "path": "./auth.py:auth",
    "openapi": {
      "securitySchemes": {
        "apiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "X-API-Key"
        }
      },
      "security": [{ "apiKeyAuth": [] }]
    },
    "disable_studio_auth": false
  }
}
```

### Studio 与 docs 行为

- FastAPI 的文档仍在 `/docs`、`/redoc`、`/openapi.json` 提供
- Studio 通过 `agentseek-api dev` 打印的本地 API base URL 连接
- 当配置了鉴权时，`agentseek-api dev` 会接受带有
  `x-auth-scheme: langsmith` 的 loopback Studio 请求
- 若希望 `dev` 模式下 Studio 与其他客户端走同一条常规 API 鉴权路径，
  把 `auth.disable_studio_auth` 设为 `true`
- 如果只是想给 Studio 实验提供一个 mock 的本地 API 服务，请使用
  `langgraph dev`，而不是 AgentSeek

## 🔌 MCP

AgentSeek API 通过位于 `/mcp` 的无状态 Streamable HTTP 端点，把注册的图
暴露为 MCP 工具。

### 行为

- 传输方式：Streamable HTTP
- 会话模型：无状态
- 鉴权：与 API 的其他部分保持一致
- 路径：`/mcp` 与 `/mcp/` 均可
- 发现来源：来自 `agentseek.json`、`langgraph.json` 或
  `AGENTSEEK_GRAPHS` 的已注册图
- 开关：MCP 默认开启；设置 `http.disable_mcp: true` 可以关闭
- 安全性：当生效的配置文件存在但无法解析时，MCP 会保持关闭直到配置修复

图对象条目可以在 manifest 中直接携带面向 MCP 的元数据：

```json
{
  "graphs": {
    "docs_agent": {
      "graph": "./docs_agent.py:graph",
      "name": "docs_agent",
      "description": "Answers documentation questions",
      "input_schema": {
        "type": "object",
        "properties": {
          "question": { "type": "string" }
        },
        "required": ["question"]
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "answer": { "type": "string" }
        },
        "required": ["answer"]
      }
    }
  }
}
```

如果省略这些字段，AgentSeek 会回退为：

- tool name = graph id
- description = 空字符串
- input schema = `{"type": "object"}`
- output schema = `{"type": "object"}`

### 关闭 MCP

在配置文件中设置 `http.disable_mcp`：

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_mcp": true
  }
}
```

### Python 客户端示例

```python
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async with httpx.AsyncClient(
    headers={"X-API-Key": "secret"},
    trust_env=False,
) as http_client:
    async with streamable_http_client(
        url="http://127.0.0.1:2024/mcp",
        http_client=http_client,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(tools)
```

当前里程碑仅覆盖把 AgentSeek 图暴露为 MCP 工具，并不会在 AgentSeek 图
内部增加 MCP 客户端的外呼能力。

## 🤝 A2A

AgentSeek API 通过位于 `/a2a/{assistant_id}` 的 LangSmith 风格 A2A
端点暴露 assistant，并在 `/.well-known/agent-card.json?assistant_id={assistant_id}`
提供 agent-card 发现。

### 行为

- 方法：`message/send`、`message/stream`、`tasks/get`、`tasks/cancel`
- Agent card 发现：以 assistant 为粒度的
  `/.well-known/agent-card.json`
- 鉴权：与 API 的其他部分保持一致
- 路径：仅 `/a2a/{assistant_id}`
- 发现来源：基于消息兼容图的 assistant
- 线程：入参中的 `contextId` 会被转发为 LangGraph `thread_id`
- 开关：A2A 默认开启；设置 `http.disable_a2a: true` 可以同时关闭 RPC
  端点与 agent-card 发现
- 安全性：当生效的配置文件存在但无法解析时，A2A 会保持关闭直到配置修复

### 关闭 A2A

在配置文件中设置 `http.disable_a2a`：

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_a2a": true
  }
}
```

### Python 客户端示例

```python
import httpx

assistant_id = "<assistant-id>"
payload = {
    "jsonrpc": "2.0",
    "id": "send-1",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "hello"}],
            "messageId": "msg-1",
        }
    },
}

with httpx.Client(headers={"X-API-Key": "secret"}, trust_env=False) as client:
    response = client.post(
        f"http://127.0.0.1:2024/a2a/{assistant_id}",
        json=payload,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    print(response.json())
```

该端点面向 assistant 间消息传递与 SDK 互操作，目标是实用的
LangSmith 对齐。任务跟踪在进程内完成，因此任务查询与取消仅适用于
当前 API 进程内创建的任务。

## 📚 作为库使用

直接嵌入 FastAPI 应用：

```python
from agentseek_api.main import create_app

app = create_app()
```

在不放弃独立 `agentseek-api` 二进制的前提下，把 CLI 挂到父级工具下：

```python
import argparse
from agentseek_api.cli import register_subcommands, run_namespace

parser = argparse.ArgumentParser(prog="parent")
subparsers = parser.add_subparsers(dest="tool", required=True)
register_subcommands(subparsers, command_name="api")

args = parser.parse_args()
raise SystemExit(run_namespace(args))
```

这样父级 CLI 就可以暴露出这样的命令：

```bash
parent api dev --config ./langgraph.json
parent api build --config ./langgraph.json -t my-api:dev
```

## 🏗️ 运行时说明

- 元数据持久化在设置了 `METADATA_DB_URL` 时优先使用该值
- 否则元数据数据库 URL 会从 `SEEKDB_URL` 或 `OCEANBASE_*` 连接配置
  解析得到
- Run 执行默认走 `EXECUTOR_BACKEND=inline`
- 设置 `EXECUTOR_BACKEND=redis` 并启动 `agentseek-api worker`，可以把
  run 通过 Redis 转交执行
- Redis 队列相关配置：
  - `REDIS_URL=redis://127.0.0.1:6379/0`
  - `REDIS_RUN_QUEUE_KEY=agentseek:runs:pending`
  - `REDIS_RUN_PROCESSING_KEY=agentseek:runs:processing`
  - `REDIS_WORKER_LOCK_KEY=agentseek:worker:active`
  - `REDIS_WORKER_LOCK_TTL_SECONDS=30`
- `METADATA_DB_BACKEND=auto` 会对驱动进行归一化：
  - PostgreSQL：`postgresql+asyncpg://...`
  - OceanBase / MySQL：`mysql+aiomysql://...`
- Checkpoint 持久化默认使用 OceanBase / seekdb 配置
- 鉴权模式：
  - `AUTH_TYPE=noop`
  - `AUTH_TYPE=custom`，搭配 `AUTH_MODULE_PATH=module:backend_symbol`
  - `AUTH_TYPE=api_key`，搭配 `AUTH_API_KEYS=key=user_id[,key2=user2]`
  - `AUTH_TYPE=jwt`，搭配 `AUTH_JWT_SECRET`，可选
    `AUTH_JWT_ALGORITHM=HS256`，使用 `sub` 作为用户身份
- Assistant 管理、thread 与 run 端点会强制执行所配置的鉴权。

### 持久化执行

- Redis 模式会把 run 流事件与 protocol 流事件持久化到元数据数据库，
  这样流回放就不再依赖 API 进程的内存。
- 只要 Redis 与元数据数据库仍然可用，被中断的 run 可以在 worker 重启
  后恢复。
- Worker 持有一个可续期的 Redis lease，在 lease 丢失时会主动退出，从而
  避免脑裂执行。

## 🧭 示例

- `examples/minimal_agentseek/agentseek.json`：最小化的首次配置
- `examples/assistant_config/`：assistant 的 config/context/metadata 入门示例
- `examples/auth/custom_backend.py`：自定义鉴权后端
- `examples/auth/jwt.md`：JWT 鉴权环境变量约定
- `examples/custom_routes/app.py`：在 AgentSeek API 应用周围挂载自定义
  FastAPI 路由

## 🧪 贡献

本仓库采用 GitFlow-lite 工作流：

- `main`：仅保留达到生产可用状态的历史
- `develop`：持续开发的集成分支
- `feature/<topic>`：从 `develop` 拉出，PR 回 `develop`
- `release/<version>`：从 `develop` 拉出，PR 到 `main`
- `hotfix/<topic>`：从 `main` 拉出，PR 到 `main`，然后再合回
  `develop`

完整的分支与 CI 策略详见 `CONTRIBUTING.md`。

常用本地校验：

```bash
make test
make test-cov
make test-cli
make test-samples
```

Docker 与真实后端校验：

```bash
make test-cli-docker
make test-e2e
make test-seekdb
make test-redis-docker
```

GitHub Actions 还会针对 MySQL、seekdb 与 OceanBase 跑 Docker 化的后端
矩阵，其中也包含专门的 `Redis Durable Execution` workflow 任务。

如需在本地跑嵌入式 seekdb 冒烟测试，先安装可选的额外依赖：

```bash
uv sync --dev --extra embedded
```

本仓库有意维护两套 GitHub Actions workflow 用于 CI：

- `.github/workflows/ci.yml`
  - 面向 pull request 与常规分支推送的常驻仓库 CI
  - 默认情况下足够快，不会消耗任何外部模型 provider 费用
  - 覆盖单元/集成测试、CLI 兼容性、示例图、Docker 运行时校验、
    MySQL 系列 checkpoint 校验、PostgreSQL 元数据校验，以及 Redis
    持久化执行
  - 使用 GitHub 可以在 job 内启动的本地依赖或 Docker 化依赖

- `.github/workflows/live-provider-streaming.yml`
  - 专门的真实模型 workflow，用于 provider 端到端验证
  - 只会在手动触发或定时夜跑时运行
  - 使用仓库的变量与 secrets，适配 OpenAI 兼容与 Anthropic 兼容的
    provider
  - 单独存在，确保默认的 PR CI 保持快速、确定，且不被外部 provider 的
    费用与限流抖动影响

设计意图是：

- `ci.yml` 用于在不依赖真实模型 provider 的情况下证明产品逻辑、
  存储/运行时集成以及后端兼容性
- `live-provider-streaming.yml` 用于证明在真实 provider 介入时，
  相同的 API 接口依然可以工作

live-provider workflow 是 provider 驱动图发出真实 SSE `message_chunk`
事件的标准证明，并且现在也覆盖了基于真实 provider 的 Store、MCP 与
HITL 流程，采用分层后端矩阵：

- seekdb：完整的 Streaming + Store + MCP + HITL 验收
- OceanBase：完整的 Streaming + Store + MCP + HITL 验收
- MySQL：Streaming + HITL 兼容性
- PostgreSQL 元数据：Streaming + MCP 兼容性，运行时的
  checkpointer/store 仍然使用 MySQL 系列后端

Workflow 行为：

- 手动触发可以指定单个 provider、单个后端层级，或者完整矩阵
- 夜跑会跑完整的 provider/后端矩阵
- 在套件运行前会先校验 provider 配置
- 后端能力按层级 gating，因此 MySQL 不会跑 Store/MCP，PostgreSQL
  元数据也不会假装替换运行时的 MySQL 系列 store/checkpointer 路径
- 每个 lane 的日志（包含失败）都会上传

日常研发信号请使用 `ci.yml`；当需要明确证明真实 provider 仍能满足
预期的 streaming、Store、MCP 与 HITL 契约时，使用
`live-provider-streaming.yml`。

## 🧱 构建基础

AgentSeek API 本质上是一层把若干上游项目串起来的胶水层。

**核心支柱**

- [LangGraph](https://github.com/langchain-ai/langgraph) — 执行已注册
  assistant 的图运行时
- [langchain-oceanbase](https://pypi.org/project/langchain-oceanbase/) —
  checkpointer 与 store 实现，是图状态的主路径
- [OceanBase](https://github.com/oceanbase/oceanbase) /
  [seekdb](https://github.com/oceanbase/seekdb) — 一等公民数据库，用于
  checkpoint 与 store 持久化
- [Redis](https://github.com/redis/redis) — 在 `EXECUTOR_BACKEND=redis`
  时承担 run 队列、worker lease 以及流事件持久化
- [FastAPI](https://github.com/tiangolo/fastapi) — 承载 `/assistants`、
  `/threads`、`/runs`、`/mcp`、`/a2a` 等全部接口的 HTTP 框架
- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/python-sdk)
  — 已注册的图通过 `/mcp` 的 Streamable HTTP 暴露为 MCP 工具
- [A2A SDK](https://github.com/a2aproject/a2a-python) — `/a2a` 下的
  assistant 间 RPC 与 agent-card 发现的形态参考

<details>
<summary>完整依赖列表</summary>

**运行时与 API**
- [Uvicorn](https://github.com/encode/uvicorn) — `agentseek-api dev` 与
  `serve` 使用的 ASGI 服务器
- [Pydantic](https://github.com/pydantic/pydantic) 与 pydantic-settings —
  请求/响应模型，以及基于环境变量的配置体系
- [scalar-fastapi](https://github.com/scalar/scalar) — 提供 `/scalar`
  替代 API 文档渲染

**LangChain / LangGraph 体系**
- [LangChain Core](https://github.com/langchain-ai/langchain) 与
  [langgraph-sdk](https://github.com/langchain-ai/langgraph) — API 所遵循的
  消息、工具与 SDK 契约
- [langchain-openai](https://github.com/langchain-ai/langchain) /
  [langchain-anthropic](https://github.com/langchain-ai/langchain) —
  示例图与 live-provider CI 使用的 provider 集成
- [Agent Protocol](https://github.com/langchain-ai/agent-protocol) —
  assistants/threads/runs 接口的对外兼容性参考

**数据库驱动**
- [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy)（异步）配合
  [asyncpg](https://github.com/MagicStack/asyncpg)（PostgreSQL）、
  [aiomysql](https://github.com/aio-libs/aiomysql) 与
  [PyMySQL](https://github.com/PyMySQL/PyMySQL)（MySQL 系列）、
  [aiosqlite](https://github.com/omnilib/aiosqlite)（SQLite）
- [redis-py](https://github.com/redis/redis-py) — 异步 Redis 客户端

**互操作**
- [LangSmith Studio](https://smith.langchain.com/) — 通过本地 API 接入
  的外部 UI，用于图检查与 run 调试

**打包与运行时交付**
- [uv](https://github.com/astral-sh/uv) — 依赖解析以及推荐的 CLI 运行方式
- [Hatchling](https://github.com/pypa/hatch) — wheel 构建后端
- [Docker](https://www.docker.com/) — `dockerfile`、`build` 与 `up`
  命令用于生成并运行容器化 API

</details>
