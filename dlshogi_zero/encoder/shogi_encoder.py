from cshogi import *

def encode_position(board, features):
    # Input features
    #   P1 piece 14
    #   P2 piece 14
    #   Repetitions 1
    #   P1 prisoner count 7
    #   P2 prisoner count 7
    #   Colour 1
    #   Total move count 1
    # piece
    pieces = board.pieces
    for sq in SQUARES:
        piece = pieces[sq]
        if piece != NONE:
            if piece >= WPAWN:
                piece = piece - 2
            features[piece - 1][sq] = 1
    # repetition
    if board.is_draw() == REPETITION_DRAW:
        features[28].fill(1)
    # prisoner count
    pieces_in_hand = board.pieces_in_hand
    for c, hands in enumerate(pieces_in_hand):
        for hp, num in enumerate(hands):
            features[29 + c * 7 + hp].fill(num)
    # Colour
    if board.turn == WHITE:
        features[43].fill(1)
    # Total move count
    # not implement for learning from hcpe

def encode_action(move):
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
                    move_dd = -diff_file - 1
                elif diff_rank > 0:
                    move_dd = 8 - diff_file - 1
                else:
                    move_dd = 16 - diff_file - 1
            elif diff_file > 0:
                if diff_rank < 0:
                    move_dd = 24 + diff_file - 1
                elif diff_rank > 0:
                    move_dd = 32 + diff_file - 1
                else:
                    move_dd = 40 + diff_file - 1
            else:
                if diff_rank < 0:
                    move_dd = 48 - diff_rank - 1
                else:
                    move_dd = 56 + diff_rank - 1
        else:
            # Knight moves
            if diff_file < 0:
                move_dd = 64
            else:
                move_dd = 65

        promotion = 1 if move_is_promotion(move) else 0

        return (promotion * 66 + move_dd) * 81 + from_sq
    else:
        # drop
        to_sq = move_to(move)
        hp = move_drop_hand_piece(move)
        return (132 + hp) * 81 + to_sq

def encode_outcome(board, game_result):
    # game outcome
    #   z: −1 for a loss, 0 for a draw, and +1 for a win
    if board.turn == BLACK:
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
