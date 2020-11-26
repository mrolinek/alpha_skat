from enum import IntEnum
from functools import total_ordering



class Suit(IntEnum):
    KULE = 0
    HERZ = 1
    BLATT = 2
    EICHEL = 3


class Figure(IntEnum):
    SEVEN = 0
    EIGHT= 1
    NINE = 2
    TEN = 3
    UNTER = 4
    OBER = 5
    KING = 6
    ACE = 7


class Card(IntEnum):
    SEVEN_KULE = 0
    EIGHT_KULE = 1
    NINE_KULE = 2
    TEN_KULE = 3
    UNTER_KULE = 4
    OBER_KULE = 5
    KING_KULE = 6
    ACE_KULE = 7
    SEVEN_HERZ = 8
    EIGHT_HERZ = 9
    NINE_HERZ = 10
    TEN_HERZ = 11
    UNTER_HERZ = 12
    OBER_HERZ = 13
    KING_HERZ = 14
    ACE_HERZ = 15
    SEVEN_BLATT = 16
    EIGHT_BLATT = 17
    NINE_BLATT = 18
    TEN_BLATT = 19
    UNTER_BLATT = 20
    OBER_BLATT = 21
    KING_BLATT = 22
    ACE_BLATT = 23
    SEVEN_EICHEL = 24
    EIGHT_EICHEL = 25
    NINE_EICHEL = 26
    TEN_EICHEL = 27
    UNTER_EICHEL = 28
    OBER_EICHEL = 29
    KING_EICHEL = 30
    ACE_EICHEL = 31



def same_suit(a, b):
    pass


def higher(a, b, trumf):
    pass
