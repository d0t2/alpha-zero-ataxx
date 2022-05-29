package ataxx;

import java.util.Random;

public class Main {

    static final float EPS = 1e-8f;
    static final Random RAND = new Random(0);

    public static void main(String[] args) {
        AtaxxGame game = new AtaxxGame();
        NeuralNet nnet = new NeuralNet(game);
        SelfPlay selfPlay = new SelfPlay(game, nnet);
        selfPlay.learn();
    }
}
