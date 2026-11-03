# src/utils/profile_decorator.py

import atexit

import line_profiler


class Profiler:
    _instance = None
    _profiler = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._profiler = line_profiler.LineProfiler()
            atexit.register(cls._instance.output_stats)
        return cls._instance

    def add_function(self, func):
        self._profiler.add_function(func)
        return func

    def output_stats(self):
        print("\n" + "=" * 80)
        print("📊 LINE PROFILER RESULTS")
        print("=" * 80)
        self._profiler.print_stats()
        print("=" * 80)

    def get_profiler(self):
        return self._profiler


# Usage:
# @Profiler().add_function
# def my_function():
#     pass
