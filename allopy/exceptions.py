class NotOptimizedError(Exception):
    def __init__(self):
        super().__init__("Optimizer has not been optimized yet")
