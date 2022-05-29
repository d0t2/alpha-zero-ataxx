package ataxx;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;

public class AtaxxGame implements Game {

    static final int SIDE = 7;
    static final int JUMP_LIMIT = 25;
    static final Shape SHAPE = new Shape(2, SIDE, SIDE);
    static final ArrayList<int[]> MOVES = new ArrayList<>();
    static {
        MOVES.add(null);
        for (int r0 = 0; r0 < SIDE; r0 += 1) {
            for (int c0 = 0; c0 < SIDE; c0 += 1) {
                for (int dr = -2; dr <= 2; dr += 1) {
                    for (int dc = -2; dc <= 2; dc += 1) {
                        int c1 = c0 + dc, r1 = r0 + dr;
                        if ((c1 != c0 || r1 != r0)
                                && (c1 >= 0 && c1 < SIDE)
                                && (r1 >= 0 && r1 < SIDE)) {
                            MOVES.add(new int[]{c0, r0, c1, r1});
                        }
                    }
                }
            }
        }
    }

    @Override
    public NDArray initial(NDManager ndmanager) {
        NDArray board = ndmanager.create(SHAPE, DataType.INT32);
        int end = SIDE - 1;
        set(board, 0, 0, -1);
        set(board, 0, end, 1);
        set(board, end, 0, 1);
        set(board, end, end, -1);
        board.set(new NDIndex(1), 0);
        return board;
    }

    @Override
    public int actionSize() {
        return MOVES.size();
    }

    @Override
    public Shape boardShape() {
        return SHAPE;
    }

    @Override
    public float value(NDArray board, int player) {
        float val;
        int[] numPieces = numPieces(board);
        if (numPieces[0] == 0) {
            val = -1;
        } else if (numPieces[1] == 0) {
            val = 1;
        } else if (!canMove(board, 1)
                && !canMove(board, -1)
                || board.getInt(1, 0, 0) == JUMP_LIMIT) {
            if (numPieces[0] > numPieces[1]) {
                val = 1;
            } else if (numPieces[1] > numPieces[0]) {
                val = -1;
            } else {
                val = Main.EPS;
            }
        } else {
            val = 0;
        }
        return val * player;
    }

    @Override
    public NDArray next(NDArray board, int player, int action) {
        NDArray newBoard = board.duplicate();
        if (action == 0) {
            return newBoard;
        }
        int[] m = MOVES.get(action);
        set(newBoard, m[2], m[3], player);
        int distance = distance(m);
        if (distance == 1) {
            newBoard.set(new NDIndex(1), 0);
        } else {
            set(newBoard, m[0], m[1], 0);
            newBoard.get(1).addi(1);
        }
        flipAdjacent(newBoard, m[2], m[3]);
        return newBoard;
    }

    @Override
    public NDArray legalActions(NDArray board, int player) {
        NDManager manager = board.getManager();
        if (value(board, player) != 0) {
            return manager.zeros(new Shape(1, actionSize()));
        }
        int[] moves = new int[MOVES.size()];
        if (!canMove(board, player)) {
            moves[0] = 1;
        } else {
            for (int i = 1; i < moves.length; i += 1) {
                int[] m = MOVES.get(i);
                if (get(board, m[0], m[1]) == player
                        && get(board, m[2], m[3]) == 0
                        && distance(m) <= 2) {
                    moves[i] = 1;
                }
            }
        }
        return manager.create(moves);
    }

    @Override
    public NDArray canonical(NDArray board, int player) {
        return board.get(0).mul(player).stack(board.get(1));
    }

    @Override
    public String repr(NDArray board) {
        return new String(board.encode());
    }

    @Override
    public String str(NDArray board) {
        StringBuilder str = new StringBuilder();
        for(int r = SIDE - 1; r >= 0; r -= 1) {
            for(int c = 0; c < SIDE; c += 1) {
                int piece = get(board, c, r);
                if (piece == 1) {
                    str.append('R');
                } else if (piece == -1) {
                    str.append('B');
                } else {
                    str.append('.');
                }
                str.append(' ');
            }
            str.append('\n');
        }
        return str.toString();
    }

    private int get(NDArray board, int c, int r) {
        return board.getInt(0, r, c);
    }

    private void set(NDArray board, int c, int r, int piece) {
        board.setScalar(new NDIndex(0, r, c), piece);
    }

    private boolean canMove(NDArray board, int player) {
        int[] numPieces = numPieces(board);
        int index = player > 0 ? 0 : 1;
        if (numPieces[index] == 0) {
            return false;
        }
        for (int r = 0; r < SIDE; r += 1) {
            for (int c = 0; c < SIDE; c += 1) {
                if (get(board, c, r) == player
                    && nearEmpty(board, c, r)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean nearEmpty(NDArray board, int c, int r) {
        for (int dr = -2; dr <= 2; dr += 1) {
            for (int dc = -2; dc <= 2; dc += 1) {
                int ac = c + dc, ar = r + dr;
                if ((ac != c || ar != r)
                        && (ac > 0 && ac < SIDE)
                        && (ar > 0 && ar < SIDE)
                        && get(board, ac, ar) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    private void flipAdjacent(NDArray board, int c, int r) {
        int player = get(board, c, r);
        for (int dr = -1; dr <= 1; dr += 1) {
            for (int dc = -1; dc <= 1; dc += 1) {
                int ar = r + dr, ac = c + dc;
                if ((ar != r || ac != c)
                        && (ar > 0 && ar < SIDE)
                        && (ac > 0 && ac < SIDE)
                        && get(board, ac, ar) == -player) {
                    set(board, ac, ar, player);
                }
            }
        }
    }

    private int[] numPieces(NDArray board) {
        long numRed = board.get(0).eq(1).countNonzero().getLong();
        long numBlue = board.get(0).eq(-1).countNonzero().getLong();
        return new int[]{(int) numRed, (int) numBlue};
    }

    int distance(int[] move) {
        int dc = Math.abs(move[0] - move[2]);
        int dr = Math.abs(move[1] - move[3]);
        return Math.max(dc, dr);
    }
}
