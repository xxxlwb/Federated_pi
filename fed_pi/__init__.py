"""fed_pi: 树莓派联邦学习课程作业.

模块概览:
- client / server: Flower 客户端与服务器入口
- models: 模型工厂 (CNN / MobileNet / SqueezeNet)
- data: 数据集加载与三种联邦划分策略 (IID/Dirichlet/Imbalance)
- train: 本地训练 / 评估 / 参数序列化
- utils: 配置 / 日志 / 系统监控 / 指标记录与绘图
"""
__version__ = "0.1.0"
