# Multilingual Safety Evaluation Framework for LLMs

## 项目概述

本项目是一个专业的多语言安全评估框架，用于评估和基准测试大型语言模型（LLMs）在不同语言和文化背景下的安全性和对齐性。该框架提供了完整的评估工具链，支持多种主流LLM的测试，包括GPT-4、Claude、LLaMA等。

## 主要特性

- 🌐 **多语言支持**：支持20+种语言的安全评估
- 🛡️ **全面的安全评估**：涵盖有害内容、偏见、隐私泄露等多个维度
- 📊 **详细的评估报告**：自动生成可视化报告和分析结果
- 🔧 **模块化设计**：易于扩展和定制
- 🚀 **高性能**：支持批量处理和并行评估
- 📦 **丰富的API**：提供RESTful API和Python SDK

## 项目结构

```
multilingual-safety-evaluation-optimized/
├── src/                    # 源代码
│   ├── core/              # 核心功能模块
│   ├── data/              # 数据处理模块
│   ├── evaluation/        # 评估框架
│   ├── models/            # 模型接口
│   ├── safety/            # 安全检测器
│   ├── utils/             # 工具函数
│   └── api/               # API接口
├── configs/               # 配置文件
├── data/                  # 数据目录
│   ├── datasets/          # 评估数据集
│   ├── cache/             # 缓存目录
│   └── results/           # 结果存储
├── tests/                 # 测试代码
│   ├── unit/              # 单元测试
│   └── integration/       # 集成测试
├── docs/                  # 文档
├── examples/              # 示例代码
├── scripts/               # 脚本工具
├── logs/                  # 日志目录
└── reports/               # 报告输出
```

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/ml-safety-framework/multilingual-safety-evaluation.git
cd multilingual-safety-evaluation-optimized
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置API密钥：
```bash
cp configs/config.example.yaml configs/config.yaml
# 编辑config.yaml，添加您的API密钥
```

### 快速使用

```python
from src.core import SafetyEvaluator

# 初始化评估器
evaluator = SafetyEvaluator(config_path="configs/config.yaml")

# 评估单个提示
result = evaluator.evaluate_prompt(
    prompt="你好，世界！",
    language="zh",
    model="gpt-4"
)

# 批量评估
results = evaluator.batch_evaluate(
    dataset_path="data/datasets/multilingual_safety_prompts.json",
    models=["gpt-4", "claude-3", "llama-2"],
    languages=["en", "zh", "es", "fr", "de"]
)

# 生成报告
evaluator.generate_report(results, output_path="reports/")
```

## 支持的模型

- OpenAI: GPT-4, GPT-3.5-Turbo
- Anthropic: Claude-3, Claude-2
- Meta: LLaMA-2, LLaMA-3
- Google: PaLM, Gemini
- 其他开源模型（通过Hugging Face）

## 评估维度

1. **有害内容检测**
   - 暴力内容
   - 仇恨言论
   - 自残内容
   - 非法活动

2. **偏见评估**
   - 性别偏见
   - 种族偏见
   - 宗教偏见
   - 文化偏见

3. **隐私保护**
   - 个人信息泄露
   - 敏感数据处理

4. **事实准确性**
   - 错误信息
   - 虚假陈述

5. **文化适应性**
   - 文化敏感性
   - 本地化质量

## API使用

### RESTful API

启动API服务器：
```bash
python -m src.api.app
```

示例请求：
```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me about safety",
    "language": "en",
    "model": "gpt-4"
  }'
```

### Python SDK

```python
from src.api.client import SafetyEvalClient

client = SafetyEvalClient(api_key="your-api-key")
result = client.evaluate(
    prompt="你好",
    language="zh",
    model="claude-3"
)
```

## 配置说明

配置文件位于 `configs/config.yaml`：

```yaml
# API配置
api_keys:
  openai: "your-openai-key"
  anthropic: "your-anthropic-key"

# 评估设置
evaluation:
  batch_size: 32
  timeout: 30
  retry_count: 3

# 安全阈值
safety_thresholds:
  harmful_content: 0.8
  bias: 0.7
  privacy: 0.9
```

## 开发指南

### 添加新的评估器

1. 在 `src/evaluation/evaluators/` 创建新的评估器类
2. 继承 `BaseEvaluator` 基类
3. 实现 `evaluate` 方法
4. 在 `src/evaluation/registry.py` 注册评估器

### 添加新的模型支持

1. 在 `src/models/` 创建新的模型接口
2. 实现 `BaseModel` 接口
3. 在配置文件中添加模型配置

## 测试

运行所有测试：
```bash
pytest tests/
```

运行特定测试：
```bash
pytest tests/unit/test_evaluator.py
```

## 贡献指南

我们欢迎各种形式的贡献！请查看 [CONTRIBUTING.md](docs/CONTRIBUTING.md) 了解详情。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目主页：[GitHub Repository](https://github.com/ml-safety-framework/multilingual-safety-evaluation)
- 问题反馈：[Issue Tracker](https://github.com/ml-safety-framework/multilingual-safety-evaluation/issues)
- 邮件：safety-eval@ml-framework.org

## 致谢

感谢所有贡献者和支持者！特别感谢以下开源项目：
- Hugging Face Transformers
- OpenAI API
- Anthropic API