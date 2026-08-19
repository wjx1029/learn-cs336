from datetime import datetime, timedelta
import time


class Timer():

    def __init__(self):
        self.start_time = datetime.now()

    def get_elapsed_time(self):
        """获取并格式化经过的时间"""
        elapsed = datetime.now() - self.start_time
        total_seconds = elapsed.total_seconds()
        
        if total_seconds < 3600:
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            return f"{minutes}分{seconds}秒"
        else:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            return f"{hours}小时{minutes}分"
    
    def __call__(self):
        return self.get_elapsed_time()

    

# 使用示例
timer = Timer()

# 你的代码
time.sleep(5)

# 输出
print(f"运行时间: {timer()}")
