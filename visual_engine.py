import pickle
import random

import pygame
from statistics import median


DIM = 1000

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

class Card(pygame.sprite.Sprite):
    def __init__(self, card_number, rotation, center):
        super().__init__()
        self.number = card_number
        self.image = pygame.image.load(f"imgs/{card_number}.jpg")
        self.image = pygame.transform.rotate(self.image, rotation)
        self.surf = self.image.convert()
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(center=center)


class Hand(object):
    def __init__(self, init_cards, player_num):
        self.player_num = player_num
        self.current_cards = init_cards
        self.rotation_angle = 90 - 90 * player_num  # player 0 -> left, 1 -> center, 2 -> right
        self.card_sprites = []
        self.sync()

    def sync(self):
        self.centers = self.recompute_centers()
        for sprite in self.card_sprites:
            sprite.kill()
        self.card_sprites = [Card(card, self.rotation_angle, center) for card, center in
                              zip(self.current_cards, self.centers)]


    def recompute_centers(self):
        edge_offset = 90
        card_center_gap = 70
        centers = [card_center_gap * i for i in range(len(self.current_cards))]
        shift = median(centers) - DIM // 2
        centers = [x - shift for x in centers]

        if self.player_num == 0:
            return [(DIM - edge_offset, center) for center in centers]
        elif self.player_num == 1:
            return [(center, DIM - edge_offset) for center in centers]
        elif self.player_num == 2:
            return [(edge_offset, center) for center in centers]
        else:
            assert 0

    def add(self, card):
        self.current_cards.append(card)
        self.sync()

    def remove(self, card):
        self.current_cards.remove(card)
        self.sync()

    def render_update(self, screen):
        for sprite in self.card_sprites:
            screen.blit(sprite.surf, sprite.rect)


class Trick(object):
    def __init__(self):
        self.cards = [None, None, None]
        self.card_sprites = []
        self.centers = self.compute_centers()


    def add(self, player, card):
        self.cards[player] = card
        self.sync()

    def end(self):
        self.cards = [None, None, None]
        self.sync()

    def sync(self):
        for sprite in self.card_sprites:
            sprite.kill()
        self.card_sprites = [Card(card, rotation=0, center=center) for card, center in
                              zip(self.cards, self.centers) if card is not None]


    def compute_centers(self):
        down_offset = 50
        side_offset = 50

        return [(DIM // 2 + side_offset, DIM // 2),
                (DIM // 2, DIM // 2 + down_offset),
                (DIM // 2 - side_offset, DIM // 2)]

    def render_update(self, screen):
        for sprite in self.card_sprites:
            screen.blit(sprite.surf, sprite.rect)



class CardGameRenderer(object):
    def __init__(self, game_file):
        with open(game_file, 'rb') as f:
            game_data = pickle.load(f)

        self.hands = game_data['init_hands']
        self.actions = game_data['actions']
        self.performed_actions = 0

        self.hand_renderers = [Hand(hand, i) for i, hand in enumerate(self.hands[:3])]
        self.trick_renderer = Trick()

    def perform_action(self):
        if all([x is not None for x in self.trick_renderer.cards]):
            self.trick_renderer.end()
            print("No actions")
            return

        card, player = self.actions[self.performed_actions]
        self.hand_renderers[player].remove(card)
        self.trick_renderer.add(player, card)

        self.performed_actions += 1

    def render_all(self, screen):
        for renderer in self.hand_renderers:
            renderer.render_update(screen)
        self.trick_renderer.render_update(screen)


pygame.init()
screen = pygame.display.set_mode([DIM, DIM])
screen.fill((0, 0, 0))


renderer = CardGameRenderer("durchmarsch.gm")


running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        # Check for KEYDOWN event
        if event.type == KEYDOWN:
            renderer.perform_action()
        # Check for QUIT event. If QUIT, then set running to false.
        elif event.type == QUIT:
            running = False

    # Draw a solid blue circle in the center

    screen.fill((0, 0, 0))
    renderer.render_all(screen)

    # Flip the display

    pygame.display.flip()

# Done! Time to quit.
pygame.quit()