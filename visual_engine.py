import os

import pygame
import pygame.camera


from rendering.components import CardGameRenderer

HEIGHT = 1100
WIDTH = 1500

from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

save_video = True

names = ["Dominik", "Michal", "Simon"]

pygame.font.init()
pygame.init()

screen = pygame.display.set_mode([WIDTH, HEIGHT])
screen.fill((0, 0, 0))


rect = (0, 0, HEIGHT, WIDTH)
renderer = CardGameRenderer("some_game2.gm", rect, names)

file_num = 0
running = True
while running:

    snap = False
    # Did the user click the window close button?
    for event in pygame.event.get():
        # Check for KEYDOWN event
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                renderer.previous()
            if event.key == K_RIGHT:
                renderer.next()

            snap = True
        # Check for QUIT event. If QUIT, then set running to false.
        elif event.type == QUIT:
            running = False

    # Draw a solid blue circle in the center

    screen.fill((0, 0, 0))
    renderer.render_all(screen)

    if snap:
        filename = "snaps/%04d.png" % file_num
        pygame.image.save(screen, filename)
        file_num += 1


    pygame.display.flip()

# Done! Time to quit.

os.system(f"ffmpeg -r 1 -f image2 -i snaps/%04d.png -y -qscale 0 -s {WIDTH}x{HEIGHT} result.avi")
pygame.quit()