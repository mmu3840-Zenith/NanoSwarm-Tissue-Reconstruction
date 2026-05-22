import pygame
import numpy as np
import imageio

pygame.init()

W, H = 600, 600
screen = pygame.display.set_mode((W, H))

frames = []

agents = np.random.rand(60,2) * 600
vel = np.zeros((60,2))

for t in range(120):
    screen.fill((10,10,20))

    # simulate swarm movement (based on your system style)
    for i in range(len(agents)):
        direction = np.random.randn(2) * 0.5
        vel[i] = vel[i]*0.8 + direction
        agents[i] += vel[i]

        x, y = agents[i].astype(int)
        x = np.clip(x,0,W-1)
        y = np.clip(y,0,H-1)

        pygame.draw.circle(screen, (0,200,255), (x,y), 3)

    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1,0,2))
    frames.append(frame)

    pygame.display.flip()

imageio.mimsave('swarm_animation.mp4', frames, fps=20)

pygame.quit()

print('Swarm animation saved')
