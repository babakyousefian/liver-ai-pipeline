# ================= SILENCE EVERYTHING =================

import os
import time

# pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# ------------------ SILENCE ------------------ #
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import pygame


class Music:
    def __init__(self, MUSIC_PATH):
        self.MUSIC_PATH = MUSIC_PATH

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def play_music(self):
        time.sleep(1)
        print("\n🎵 music start\n")
        time.sleep(1)

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(self.MUSIC_PATH)
            pygame.mixer.music.play(-1)
        except:
            pass

    def stop_music(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
