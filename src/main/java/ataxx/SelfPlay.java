package ataxx;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;

public class SelfPlay {

    static final File TEMP_SAVE = new File("temp-nnet");
    static final File BEST_SAVE = new File("best-nnet");
    static final int NUM_LEARN = 1000;
    static final int NUM_EXAMPLE_GAMES = 100;
    static final int NUM_COMPARE_GAMES = 40;
    static final int NUM_EXAMPLES_HISTORY = 20;
    static final int TEMP_THRESHOLD = 100;
    static final float WIN_RATE_THRESHOLD = 0.6f;

    SelfPlay(Game game, NeuralNet nnet) {
        _game = game;
        _nnet = nnet;
        _exampleHistory = new LinkedList<>();
    }

    void learn() {
        for (int i = 0; i < NUM_LEARN; i += 1) {
            System.out.println("========");
            System.out.println("learning session " + i);

            ArrayList<NDList> examples = new ArrayList<>();
            System.out.print("\texample self play games:");
            for (int j = 0; j < NUM_EXAMPLE_GAMES; j += 1) {
                _mcts = new MCTS(_game, _nnet);
                examples.addAll(examplesFromGame());
                System.out.print(" " + j);
            }
            System.out.println();

            _exampleHistory.add(examples);
            if (_exampleHistory.size() > NUM_EXAMPLES_HISTORY) {
                _exampleHistory.poll();
            }
            ArrayList<NDList> allExamples = new ArrayList<>();
            for (ArrayList<NDList> exs : _exampleHistory) {
                allExamples.addAll(exs);
            }
            Collections.shuffle(allExamples, Main.RAND);

            _nnet.save(TEMP_SAVE);
            NeuralNet oldNNet = new NeuralNet(_game);
            oldNNet.load(TEMP_SAVE);
            MCTS oldMCTS = new MCTS(_game, oldNNet);
            _nnet.train(allExamples);
            MCTS newMCTS = new MCTS(_game, _nnet);

            int[] numWins = compareNumWins(oldMCTS, newMCTS);
            int totalWins = numWins[0] + numWins[1];
            float winRate = (float) numWins[1] / (totalWins);
            if ( totalWins == 0 || winRate < WIN_RATE_THRESHOLD) {
                System.out.println("\trejected, winrate: " + winRate);
                _nnet.load(TEMP_SAVE);
            } else {
                System.out.println("\taccepted, winrate: " + winRate);
                _nnet.save(BEST_SAVE);
            }
        }
    }

    int[] compareNumWins(MCTS p1, MCTS p2) {
        System.out.println("\tcomparison games: ");
        int wins1 = 0, wins2 = 0, half = NUM_COMPARE_GAMES / 2;

        System.out.print("\t\told nnet starts:");
        for (int i = 0; i < half; i += 1) {
            float result = resultFromGame(p1, p2);
            wins1 += result == 1 ? 1 : 0;
            wins2 += result == -1 ? 1 : 0;
            System.out.print(" " + i);
        }
        System.out.println();

        System.out.print("\t\tnew nnet starts:");
        for (int i = 0; i < half; i += 1) {
            float result = resultFromGame(p2, p1);
            wins1 += result == -1 ? 1 : 0;
            wins2 += result == 1 ? 1 : 0;
            System.out.print(" " + i);
        }
        System.out.println();

        return new int[]{wins1, wins2};
    }

    float resultFromGame(MCTS p1, MCTS p2) {
        int player = 1, action;
        NDArray board = _game.initial(_nnet.getNDManager());
        float value = 0;
        while (value == 0) {
            NDArray canonical = _game.canonical(board, player);
            NDArray pi;
            if (player > 0) {
                pi = p1.policy(canonical, 0);
            } else {
                pi = p2.policy(canonical, 0);
            }
            action = pi.argMax().getInt(0);
            board = _game.next(board, player, action);
            value = _game.value(board, player);
            player = -player;
        }
        return player * _game.value(board, player);
    }

    ArrayList<NDList> examplesFromGame() {
        ArrayList<Object[]> examples = new ArrayList<>();
        NDArray board = _game.initial(_nnet.getNDManager());
        int player = 1, turn = 1, action;
        float value = 0;
        while (turn <= 10) {
            System.out.println("\n" + _game.str(board));
            int temperature = turn < TEMP_THRESHOLD ? 1 : 0;
            NDArray canonical = _game.canonical(board, player);
            NDArray policy = _mcts.policy(canonical, temperature);
            examples.add(new Object[]{canonical, player, policy});
            action = chooseAction(policy);
            board = _game.next(board, player, action);
            player = -player;
            value = _game.value(board, player);
            turn += 1;
        }
        System.exit(0);
        ArrayList<NDList> valuedExamples = new ArrayList<>();
        for (Object[] ex : examples) {
            NDArray b = (NDArray) ex[0];
            NDArray pi = (NDArray) ex[2];
            boolean samePlayer = (int) ex[1] == player;
            NDArray v = _nnet.getNDManager().create(
                    samePlayer ? value : -value);
            valuedExamples.add(new NDList(b, pi, v));
        }
        return valuedExamples;
    }

    int chooseAction(NDArray policy) {
        int action = -1;
        float cumSum = 0, rand = Main.RAND.nextFloat();
        while (cumSum < rand) {
            action += 1;
            cumSum += policy.getFloat(action);
        }
        return action;
    }

    private Queue<ArrayList<NDList>> _exampleHistory;
    private NeuralNet _nnet;
    private MCTS _mcts;
    private Game _game;
}