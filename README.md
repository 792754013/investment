# 产品优先（Product-first）的约束×需求投资决策系统 v1

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

### 1) 选产品（生成监控清单）

```bash
python -m system.cli select --product GOLD
```

### 2) 运行单日决策

```bash
python -m system.cli run --product GOLD --date 2026-01-19
```

### 3) 回测（支持阶段注入）

```bash
python -m system.cli backtest --product GOLD --start 2025-09-01 --end 2026-01-19 --stage-overrides configs/stage_overrides.yaml
```

### 4) 回放

```bash
python -m system.cli replay --run-id <id>
```

## 新增产品

1. 在 `system/configs/products.yaml` 中新增产品与主题列表。
2. 在 `system/configs/themes.yaml` 中为主题绑定约束。
3. 在 `system/configs/constraints.yaml` 中新增人工维护的结构约束。
4. 在 `system/configs/assets.csv` 中绑定标的到产品与主题。
5. 在 `system/data/news_stub.csv` 中新增产品与主题的需求事件记录。
