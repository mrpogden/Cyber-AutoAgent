# Benchmarks Directory

## Start full-stack or single-container mode

```shell
cd docker
docker build -t cyber-autoagent-tools -f Dockerfile.tools ..

docker compose --env-file ../.env up -d
# OR
docker compose --env-file ../.env up -d --no-deps cyber-autoagent
```

## XBOW Benchmarks

Checkout one of the benchmark repos:
- `git clone --depth=1 https://github.com/schniggie/validation-benchmarks.git`
- `git clone --depth=1 https://github.com/xbow-engineering/validation-benchmarks.git`

Configure necessary environment variables:

```shell
# always needed:
TARGET_HOST=192.168.1.100
PROVIDER=ollama
MODEL=qwen3-coder-30b

# for Ollama
OLLAMA_HOST=http://${TARGET_HOST}:11434
OLLAMA_TIMEOUT=1800

# if you need to rate limit model calls:
CYBER_RATE_LIMIT_TOKENS_PER_MIN=100000
CYBER_RATE_LIMIT_REQ_PER_MIN=3
CYBER_RATE_LIMIT_MAX_CONCURRENT=1

# to use MCP servers
CYBER_MCP_ENABLED=true
CYBER_MCP_CONNECTIONS="$(jq -c .mcp.connections < ~/.cyber-autoagent/config.json)"
```

Execute:
```shell
./run_xbow_benchmark.sh XBEN-001-24

./run_xbow_benchmark.sh --all
```
