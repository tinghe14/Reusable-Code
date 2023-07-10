# Ref: https://juejin.cn/s/python%20logging%20%E4%BE%8B%E5%AD%90

# 这个例子中，我们创建了一个 logger 实例，并设置了它的日志级别。然后，我们创建了两个日志处理器：一个输出到控制台，另一个输出到文件。
# 接着，我们为每个日志处理器设置了日志级别和日志格式。最后，我们为 logger 添加记录日志

import logging

# 创建 logger 实例
logger = logging.getLogger(__name__)

# 设置 logger 的日志级别
logger.setLevel(logging.INFO)

# 创建输出到控制台的日志处理器
console_handler = logging.StreamHandler()

# 创建输出到文件的日志处理器
file_handler = logging.FileHandler('app.log')

# 设置日志处理器的日志级别
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.ERROR)

# 设置日志处理器的日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 为 logger 添加日志处理器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 使用 logger 记录日志
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
