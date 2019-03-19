from cshogi import *
import math

MAX_HISTORY = 8
FEATURES_PER_HISTORY = 45
MAX_FEATURES = FEATURES_PER_HISTORY * MAX_HISTORY + 2
MAX_ACTION_PLANES = 64 + 2 + 64 + 2 + 7
MAX_ACTION_LABELS = MAX_ACTION_PLANES * 81

def make_position_features(board, repetition, features, hist):
    # Input features
    #   P1 piece 14
    #   P2 piece 14
    #   Repetitions 3
    #   P1 prisoner count 7
    #   P2 prisoner count 7

    hfeatures = features[FEATURES_PER_HISTORY * hist:FEATURES_PER_HISTORY * (hist + 1)]
    # piece
    board.piece_planes(hfeatures)
    # repetition
    if repetition == 1:
        hfeatures[28].fill(1)
    elif repetition == 2:
        hfeatures[29].fill(1)
    elif repetition == 3:
        hfeatures[30].fill(1)
    # prisoner count
    pieces_in_hand = board.pieces_in_hand
    for c, hands in enumerate(pieces_in_hand):
        for hp, num in enumerate(hands):
            if hp == HPAWN:
                max_hp_num = 8
            elif hp == HBISHOP or hp == HROOK:
                max_hp_num = 2
            else:
                max_hp_num = 4

            hfeatures[31 + c * 7 + hp].fill(num / max_hp_num)

def make_color_totalmovecout_features(color, totalmovecout, features):
    #   Colour 1
    #   Total move count 1

    # Colour
    if color == WHITE:
        features[FEATURES_PER_HISTORY * MAX_HISTORY].fill(1)
    # Total move count
    features[FEATURES_PER_HISTORY * MAX_HISTORY + 1].fill(math.tanh(totalmovecout/150))

def make_action_label(move):
    # Action representation
    #   Queen moves 64
    #   Knight moves 2
    #   Promoting queen moves 64
    #   Promoting knight moves 2
    #   Drop 7
    if not move_is_drop(move):
        from_sq = move_from(move)
        to_sq = move_to(move)
        from_file, from_rank = divmod(from_sq, 9)
        to_file, to_rank = divmod(to_sq, 9)
        diff_file = to_file - from_file
        diff_rank = to_rank - from_rank
        if abs(diff_file) != 1 or abs(diff_rank) != 2:
            # Queen moves
            if diff_file < 0:
                if diff_rank < 0:
                    move_dd = -diff_file - 1 # NE
                elif diff_rank > 0:
                    move_dd = 8 - diff_file - 1 # SE
                else:
                    move_dd = 16 - diff_file - 1 # E
            elif diff_file > 0:
                if diff_rank < 0:
                    move_dd = 24 + diff_file - 1 # NW
                elif diff_rank > 0:
                    move_dd = 32 + diff_file - 1 # SW
                else:
                    move_dd = 40 + diff_file - 1 # W
            else:
                if diff_rank < 0:
                    move_dd = 48 - diff_rank - 1 # N
                else:
                    move_dd = 56 + diff_rank - 1 # S
        else:
            # Knight moves
            if diff_file < 0:
                move_dd = 64 # E
            else:
                move_dd = 65 # W

        promotion = 1 if move_is_promotion(move) else 0

        return (promotion * 66 + move_dd) * 81 + from_sq
    else:
        # drop
        to_sq = move_to(move)
        hp = move_drop_hand_piece(move)
        return (132 + hp) * 81 + to_sq

def make_outcome(color, game_result):
    # game outcome
    #   z: −1 for a loss, 0 for a draw, and +1 for a win
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1
        if game_result == WHITE_WIN:
            return -1
        else:
            return 0
    else:
        if game_result == BLACK_WIN:
            return -1
        if game_result == WHITE_WIN:
            return 1
        else:
            return 0
