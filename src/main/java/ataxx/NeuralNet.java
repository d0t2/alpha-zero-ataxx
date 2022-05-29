package ataxx;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import java.util.ArrayList;
import java.util.List;

public class NeuralNet {

    static final float LEARNING_RATE = 0.001f;
    static final float DROPOUT = 0.3f;
    static final int EPOCHS = 10;
    static final int BATCH_SIZE = 64;
    static final int NUM_CHANNELS = 512;

    NeuralNet(Game game) {
        _game = game;
        _manager = NDManager.newBaseManager();
        _inShape = new Shape(1).addAll(_game.boardShape());
        _cnn = new CNN(_game, DROPOUT, NUM_CHANNELS);
        _cnn.initialize(_manager, DataType.FLOAT32, _inShape);
        _params = new ParameterStore(_manager, false);
        Tracker lr = Tracker.fixed(LEARNING_RATE);
        Optimizer optimizer = Optimizer.adam()
                .optLearningRateTracker(lr).build();
        _params.setParameterServer(
                _manager.getEngine().newParameterServer(optimizer),
                new Device[]{_manager.getDevice()});
    }

    void train(ArrayList<NDList> examples) {
        int n = examples.size();
        System.out.println("\ttraining nnet: ");
        for (int e = 0; e < EPOCHS; e += 1) {
            float vLosses = 0, piLosses = 0;
            for (int b0 = 0; b0 < n; b0 += BATCH_SIZE) {
                NDManager submanager = _manager.newSubManager();
                int b1 = Math.min(n, b0 + BATCH_SIZE);
                Object[] batch = batch(examples.subList(b0, b1), submanager);
                NDList exBoards = (NDList) batch[0];
                NDArray exPi = (NDArray) batch[1], exV = (NDArray) batch[2];
                try (GradientCollector collector = newGC()) {
                    NDList out = _cnn.forward(_params, exBoards, true);
                    NDArray outPi = out.get(0), outV = out.get(1);
                    NDArray piLoss = exPi.mul(outPi).mean().neg();
                    NDArray vLoss = exV.sub(outV).square().mean();
                    NDArray loss = piLoss.add(vLoss);
                    collector.backward(loss);
                    _params.updateAllParameters();
                    piLosses += piLoss.getFloat(0);
                    vLosses += vLoss.getFloat(0);
                }
                submanager.close();
            }
            piLosses /= BATCH_SIZE;
            vLosses /= BATCH_SIZE;
            String format = "\t\tepoch: %s, pi: %s, v: %s";
            System.out.println(String.format(format, e, piLosses, vLosses));
        }
    }

    Object[] batch(List<NDList> sublist, NDManager manager) {
        NDList bl = new NDList(), pl = new NDList(), vl = new NDList();
        for (NDList ex : sublist) {
            bl.add(ex.get(0));
            pl.add(ex.get(1));
            vl.add(ex.get(2));
        }
        NDArray b = NDArrays.stack(bl);
        NDArray p = NDArrays.stack(pl);
        NDArray v = NDArrays.stack(vl);
        b.attach(manager);
        p.attach(manager);
        v.attach(manager);
        return new Object[]{new NDList(b), p, v};
    }

    GradientCollector newGC() {
        return _manager.getEngine().newGradientCollector();
    }

    NDList predict(NDArray board) {
        NDList reshaped = new NDList(board.reshape(_inShape));
        NDList result = _cnn.forward(_params, reshaped, false);
        NDArray policy = result.get(0).get(0);
        NDArray value = result.get(1).get(0);
        return new NDList(policy, value);
    }

    NDManager getNDManager() {
        return _manager;
    }

    void save(File file) {
        try {
            FileOutputStream fos = new FileOutputStream(file);
            DataOutputStream dos = new DataOutputStream(fos);
            _cnn.saveParameters(dos);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    void load(File file) {
        try {
            FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis);
            _cnn.loadParameters(_manager, dis);
        } catch (IOException | MalformedModelException ex) {
            ex.printStackTrace();
        }
    }

    private Game _game;
    private Block _cnn;
    private Shape _inShape;
    private NDManager _manager;
    private ParameterStore _params;
}
