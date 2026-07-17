"""Music. Drop .mp3 files into the `music/` folder and they show up here.

Playback is entirely optional: if SDL has no audio device (or the mixer fails to
start), everything degrades to silence without touching the game loop.
"""
from __future__ import annotations

import os
import random

import pygame

EXTS = (".mp3", ".ogg", ".wav", ".flac")


class Music:
    def __init__(self, folder: str):
        self.folder = folder
        self.tracks: list[str] = []
        self.index = 0
        self.ok = False
        self.volume = 0.55
        self.muted = False
        self.shuffle = False
        self.END = pygame.USEREVENT + 7
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            self.ok = True
        except Exception:
            self.ok = False
        self.rescan()

    # ------------------------------------------------------------- library
    def rescan(self) -> None:
        self.tracks = []
        if not os.path.isdir(self.folder):
            try:
                os.makedirs(self.folder, exist_ok=True)
            except Exception:
                return
        try:
            for f in sorted(os.listdir(self.folder)):
                if f.lower().endswith(EXTS) and not f.startswith("."):
                    self.tracks.append(os.path.join(self.folder, f))
        except Exception:
            pass
        self.index = 0

    @property
    def has_music(self) -> bool:
        return self.ok and bool(self.tracks)

    def title(self) -> str:
        if not self.tracks:
            return "no tracks - drop .mp3 files in music/"
        name = os.path.basename(self.tracks[self.index % len(self.tracks)])
        return os.path.splitext(name)[0]

    # ----------------------------------------------------------- transport
    def play(self, index: int | None = None) -> None:
        if not self.has_music:
            return
        if index is not None:
            self.index = index % len(self.tracks)
        try:
            pygame.mixer.music.load(self.tracks[self.index])
            pygame.mixer.music.set_volume(0.0 if self.muted else self.volume)
            pygame.mixer.music.play()
            pygame.mixer.music.set_endevent(self.END)
        except Exception:
            # a bad file should never take the game down; skip past it
            if len(self.tracks) > 1:
                self.tracks.pop(self.index)
                self.index %= max(1, len(self.tracks))

    def start_if_idle(self) -> None:
        if self.has_music and not pygame.mixer.music.get_busy():
            self.play()

    def next(self) -> None:
        if not self.has_music:
            return
        if self.shuffle and len(self.tracks) > 1:
            nxt = self.index
            while nxt == self.index:
                nxt = random.randrange(len(self.tracks))
            self.index = nxt
        else:
            self.index = (self.index + 1) % len(self.tracks)
        self.play()

    def prev(self) -> None:
        if not self.has_music:
            return
        self.index = (self.index - 1) % len(self.tracks)
        self.play()

    def toggle_pause(self) -> None:
        if not self.has_music:
            return
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self._paused = True
        else:
            try:
                pygame.mixer.music.unpause()
            except Exception:
                self.play()

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        if self.ok:
            try:
                pygame.mixer.music.set_volume(0.0 if self.muted else self.volume)
            except Exception:
                pass

    def nudge_volume(self, d: float) -> None:
        self.volume = max(0.0, min(1.0, self.volume + d))
        self.muted = False
        if self.ok:
            try:
                pygame.mixer.music.set_volume(self.volume)
            except Exception:
                pass

    def handle(self, ev) -> None:
        if ev.type == self.END:
            self.next()

    def stop(self) -> None:
        if self.ok:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
