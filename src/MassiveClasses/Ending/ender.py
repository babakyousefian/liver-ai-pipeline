import time


class End:
    def __init__(self, total_start):
        self.total_start = total_start

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def ending(self):
        end_time = time.ctime(time.time())

        time.sleep(1)
        print(f"\n🕒 END TIME : {end_time}")
        time.sleep(1)

        total_end = time.time()

        elapsed = total_end - self.total_start

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        time.sleep(1)
        print(f"\n⏱️ TOTAL PIPELINE TIME = 🔥 {hours}h : {minutes}m : {seconds}s\n\n")
        time.sleep(1)

        return total_end
