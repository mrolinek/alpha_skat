from pygame.locals import K_LEFT, K_RIGHT, KEYDOWN, QUIT
import os
import sys

import pygame
import pygame.camera

from rendering.components import CardGameRenderer

HEIGHT = 1100
WIDTH = 1500


save_video = True

pygame.font.init()
pygame.init()

screen = pygame.display.set_mode([WIDTH, HEIGHT])
screen.fill((0, 0, 0))


rect = (0, 0, HEIGHT, WIDTH)
renderer = CardGameRenderer(sys.argv[1], rect)

file_num = 0
running = True
record_video = (len(sys.argv) > 1 and sys.argv[2] == 'record')
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

    if snap and record_video:
        filename = "snaps/%04d.png" % file_num
        pygame.image.save(screen, filename)
        file_num += 1

    pygame.display.flip()

# Done! Time to quit.
if record_video:
    os.system(
        f"ffmpeg -r 1 -f image2 -i snaps/%04d.png -y -qscale 0 -s {WIDTH}x{HEIGHT} result.avi")
pygame.quit()
