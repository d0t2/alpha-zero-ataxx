
package ataxx;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;

interface Game {

    /** starting state of the board */
    NDArray initial(NDManager ndmanager);

    /** returns total number of possible actions */
    int actionSize();

    /** returns shape of game board */
    Shape boardShape();

    /** 1 if player won on board, -1 if lost, 0 if in progress,
     *  small positive value if drawn */
    float value(NDArray board, int player);

    /** player takes action on board (in place) */
    NDArray next(NDArray board, int player, int action);

    /** flips board if player is opponent (in place) */
    NDArray canonical(NDArray board, int player);

    /** returns all moves available to player on board */
    NDArray legalActions(NDArray board, int player);

    /** returns string representation of board */
    String repr(NDArray board);

    /** returns readable display string of board */
    String str(NDArray board);
}
