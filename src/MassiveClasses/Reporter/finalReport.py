import time

from src.reporter.reportGenerator import generate_report


class FinalReporter:
    def __init__(self, CSV_PATH, FINAL_REPORTS, total_start, total_end, files):
        self.CSV_PATH = CSV_PATH
        self.FINAL_REPORTS = FINAL_REPORTS
        self.total_start = total_start
        self.total_end = total_end
        self.files = files
        # self.myMusic = myMusic

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def finalReporter(self):
        generate_report(
            csv_path=self.CSV_PATH,
            output_pdf=self.FINAL_REPORTS,
            start_time=time.ctime(self.total_start),
            end_time=time.ctime(self.total_end),
            input_count=len(self.files),
        )

        time.sleep(1)
        print("\n🔥 DONE - PERFECT PIPELINE")
        time.sleep(1)

        time.sleep(1)
        print("\n@Author by : babak yousefian")
        time.sleep(1)

        # self.myMusic.stop_music()

        time.sleep(1)
        print("\n\n                 DONE.!                  \n\n")
        time.sleep(1)
