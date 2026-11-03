import time


class Timer:
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def timer(self):
        time.sleep(0.5)

        total_start = time.time()  # 🔥 START TIMER

        time.sleep(1)
        print("\n🧠 Starting the pipeline...\n")
        time.sleep(1)

        starter_ = time.ctime(time.time())

        time.sleep(1)
        print(f"\n🕒 START TIME : {starter_}")
        time.sleep(1)
        return total_start
