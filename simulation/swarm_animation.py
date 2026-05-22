import pygame
import numpy as np
import imageio

pygame.init()

W, H = 600, 600
screen = pygame.display.set_mode((W, H))

frames = []

for t in range(120):
    screen.fill((10,10,20))

    for i in range(40):
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        pygame.draw.circle(screen, (0,200,255), (x,y), 3)

    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1,0,2))
    frames.append(frame)

    pygame.display.flip()

imageio.mimsave('visuals/swarm_animation.mp4', frames, fps=20)

pygame.quit()

print('Animation saved')
