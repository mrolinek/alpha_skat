import pickle
import random
from collections import namedtuple
from itertools import chain

import pygame
from pygame.locals import RLEACCEL
from statistics import median


class Card(pygame.sprite.Sprite):
    def __init__(self, card_number, rotation, center):
        super().__init__()
        self.number = card_number
        self.image = pygame.image.load(f"imgs/{card_number}.jpg")
        self.image = pygame.transform.rotate(self.image, rotation)
        self.surf = self.image.convert()
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(center=center)

Rect = namedtuple('Rect', ['top', 'left', 'bottom', 'right'])

class Hand(object):
    def __init__(self, init_cards, player_num, rect):
        self.player_num = player_num
        self.current_cards = init_cards
        self.rotation_angle = 90 - 90 * player_num  # player 0 -> left, 1 -> center, 2 -> right
        self.card_sprites = []
        self.rect = rect
        self.sync()

    def sync(self):
        self.centers = self.recompute_centers()
        for sprite in self.card_sprites:
            sprite.kill()
        self.card_sprites = [Card(card, self.rotation_angle, center) for card, center in
                              zip(self.current_cards, self.centers)]


    def recompute_centers(self):
        if not self.current_cards:
            return []
        edge_offset = 90
        card_center_gap = 70
        centers = [card_center_gap * i for i in range(len(self.current_cards))]
        if self.player_num == 1:
            middle = (self.rect.left + self.rect.right) // 2
        else:
            middle = (self.rect.top + self.rect.bottom) // 2

        shift = median(centers) - middle
        centers = [x - shift for x in centers]

        if self.player_num == 0:
            return [(self.rect.right - edge_offset, center) for center in centers]
        elif self.player_num == 1:
            return [(center, self.rect.bottom - edge_offset) for center in centers]
        elif self.player_num == 2:
            return [(self.rect.left + edge_offset, center) for center in centers]
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
    def __init__(self, rect):
        self.cards = [None, None, None]
        self.card_sprites = []
        self.rect = rect
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

        x_middle = (self.rect.left + self.rect.right) // 2
        y_middle = (self.rect.top + self.rect.bottom) // 2
        return [(x_middle + side_offset, y_middle),
                (x_middle, y_middle + down_offset),
                (x_middle - side_offset, y_middle)]

    def render_update(self, screen):
        for sprite in self.card_sprites:
            screen.blit(sprite.surf, sprite.rect)


class History(object):
    def __init__(self, rect, num_tricks):
        self.rect = rect
        self.cards = [[None, None, None] for i in range(num_tricks)]
        self.card_sprites = []
        self.num_tricks = num_tricks
        self.centers = self.compute_centers()

    def add(self, player, trick, card):
        self.cards[trick][2-player] = card
        self.sync()

    def sync(self):
        for sprite in self.card_sprites:
            sprite.kill()

        self.card_sprites = [Card(card, rotation=0, center=center) for card, center in
                             zip(chain(*self.cards), chain(*self.centers)) if card is not None]

    def compute_centers(self):
        top_offset = 70
        left_offset = 50
        x_per_card = 80
        y_per_card = 105
        middle_lower = 10

        y_centers = [self.rect.top + top_offset + y_per_card * i for i in range(self.num_tricks)]
        x_centers = [self.rect.left + left_offset + x_per_card * i for i in range(3)]


        def centers_for_k(k):
            return [(x_centers[0], y_centers[k]),
                    (x_centers[1], y_centers[k]+middle_lower),
                    (x_centers[2], y_centers[k])]

        centers = [centers_for_k(k) for k in range(self.num_tricks)]
        return centers

    def render_update(self, screen):
        for sprite in self.card_sprites:
            screen.blit(sprite.surf, sprite.rect)


class StatusBar(object):
    def __init__(self, rect, skat, names):
        self.rect = rect
        self.cards = [Card(skat[0], rotation=0, center=(self.rect.left + 140, 60)),
                      Card(skat[1], rotation=0, center=(self.rect.left + 205, 60))]
        self.scores = [0, 0, 0]
        self.score_texts = []

        self.names = names

        title_font = pygame.font.SysFont('Comic Sans MS', 60)
        self.title_text = title_font.render('Ramsch', False, (255, 255, 255))

        smaller_font = pygame.font.SysFont('Comic Sans MS', 30)
        self.skat_text = smaller_font.render('Skat:', False, (255, 255, 255))
        self.sync()


    def update_scores(self, new_scores):
        self.scores = new_scores
        self.sync()

    def sync(self):
        smaller_font = pygame.font.SysFont('Comic Sans MS', 30)

        texts = [f"{name}: {score}" for name, score in zip(self.names, self.scores)]
        self.score_texts = [smaller_font.render(text, False, (255, 255, 255)) for text in texts]

    def render_update(self, screen):
        screen.blit(self.title_text, ((self.rect.left + self.rect.right) // 2, 30))

        screen.blit(self.skat_text, (self.rect.left + 20, 50))
        screen.blit(self.cards[0].surf, self.cards[0].rect)
        screen.blit(self.cards[1].surf, self.cards[1].rect)

        score_centers = [(self.rect.right - 150, 150),
                         ((self.rect.left + self.rect.right) // 2, 150),
                         (self.rect.left + 150, 150)]


        for text, center in zip(self.score_texts, score_centers):
            screen.blit(text, center)


class CardGameRenderer(object):
    def __init__(self, game_file, rect, names):
        with open(game_file, 'rb') as f:
            game_data = pickle.load(f)

        self.hands = game_data['init_hands']
        self.actions = game_data['actions']
        self.all_scores = game_data['historical_scores']
        self.final_scores = game_data['final_scores']
        self.performed_actions = 0
        self.done = False
        self.rect = Rect(*rect)

        game_rect = Rect(top=self.rect.top + 200,
                         bottom=self.rect.bottom,
                         left=self.rect.left,
                         right=int(0.8*self.rect.right+0.2*self.rect.left))
        history_rect = Rect(top=self.rect.top,
                         bottom=self.rect.bottom,
                         left=game_rect.right,
                         right=self.rect.right)

        status_rect = Rect(top=self.rect.top,
                           bottom=self.rect.top + 200,
                           left=self.rect.left,
                           right=int(0.8 * self.rect.right + 0.2 * self.rect.left))

        self.hand_renderers = [Hand(hand, i, game_rect) for i, hand in enumerate(self.hands[:3])]
        self.trick_renderer = Trick(game_rect)
        self.history_renderer = History(history_rect, num_tricks=10)
        self.status_bar_renderer = StatusBar(status_rect, skat=self.hands[3], names=names)

    def perform_action(self):
        if all([x is not None for x in self.trick_renderer.cards]):
            self.trick_renderer.end()
            self.status_bar_renderer.update_scores(self.all_scores[-1 + self.performed_actions // 3])

            if self.performed_actions == 30:
                self.status_bar_renderer.update_scores(self.final_scores)
                self.done = True

            return

        card, player = self.actions[self.performed_actions]
        self.hand_renderers[player].remove(card)
        self.history_renderer.add(player, self.performed_actions // 3, card)
        self.trick_renderer.add(player, card)

        self.performed_actions += 1

    def render_all(self, screen):
        for renderer in self.hand_renderers:
            renderer.render_update(screen)
        self.trick_renderer.render_update(screen)
        self.history_renderer.render_update(screen)
        self.status_bar_renderer.render_update(screen)
