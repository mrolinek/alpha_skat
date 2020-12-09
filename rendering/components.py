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
    def __init__(self, player_num, rect):
        self.player_num = player_num
        self.current_cards = []
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
        if not len(self.current_cards):
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

    def set_cards(self, new_cards):
        self.current_cards = new_cards
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

    def set_cards(self, current_trick, starting_player):

        self.cards = [None, None, None]
        for i, card in enumerate(current_trick):
            self.cards[(starting_player + i) % 3] = card
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

    def set_history(self, played_cards, players):
        self.cards = [[None, None, None] for i in range(self.num_tricks)]
        for i, (card, player) in enumerate(zip(played_cards, players)):
            self.cards[i // 3][player] = card
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
            return [(x_centers[2], y_centers[k]),
                    (x_centers[1], y_centers[k]+middle_lower),
                    (x_centers[0], y_centers[k])
                    ]

        centers = [centers_for_k(k) for k in range(self.num_tricks)]
        return centers

    def render_update(self, screen):
        for sprite in self.card_sprites:
            screen.blit(sprite.surf, sprite.rect)


class StatusBar(object):
    def __init__(self, rect, names):
        self.rect = rect
        self.cards = []
        self.scores = [0, 0, 0]
        self.score_texts = []

        self.names = names

        title_font = pygame.font.SysFont('Comic Sans MS', 60)
        self.title_text = title_font.render('Ramsch', False, (255, 255, 255))

        smaller_font = pygame.font.SysFont('Comic Sans MS', 30)
        self.skat_text = smaller_font.render('Skat:', False, (255, 255, 255))
        self.sync()


    def set_status(self, current_status):
        skat = current_status.skat_as_ints
        self.cards = [Card(skat[0], rotation=0, center=(self.rect.left + 140, 60)),
                      Card(skat[1], rotation=0, center=(self.rect.left + 205, 60))]
        self.scores = current_status.current_scores
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
    def __init__(self, game_file, rect):
        with open(game_file, 'rb') as f:
            self.status_list, self.names = pickle.load(f)

        self.current_index = 0

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

        self.hand_renderers = [Hand(player_num=i, rect=game_rect) for i in range(3)]
        self.trick_renderer = Trick(game_rect)
        self.history_renderer = History(history_rect, num_tricks=10)
        self.status_bar_renderer = StatusBar(status_rect, names=self.names)

        self.update_all()


    def next(self):
        if self.current_index + 1 < len(self.status_list):
            self.current_index += 1
        self.update_all()

    def previous(self):
        if self.current_index > 0:
            self.current_index -= 1
        self.update_all()

    def update_all(self):
        current_status = self.status_list[self.current_index]

        hands_as_ints = current_status.hands_as_ints
        for hand, hand_renderer in zip(hands_as_ints, self.hand_renderers):
            hand_renderer.set_cards(hand)

        current_trick = current_status.current_trick_as_ints
        starter_of_trick = current_status.starter_of_current_trick
        self.trick_renderer.set_cards(current_trick, starter_of_trick)

        all_played_cards = current_status.played_cards_as_ints
        player_ids = current_status.players_of_all_played_cards
        self.history_renderer.set_history(all_played_cards, player_ids)

        self.status_bar_renderer.set_status(current_status)


    def render_all(self, screen):
        for renderer in self.hand_renderers:
            renderer.render_update(screen)
        self.trick_renderer.render_update(screen)
        self.history_renderer.render_update(screen)
        self.status_bar_renderer.render_update(screen)
