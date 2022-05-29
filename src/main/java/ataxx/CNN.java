package ataxx;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

class CNN extends AbstractBlock {

    public CNN(Game game, float dropOut, int numChannels) {
        super((byte) 1);
        _actionSize = game.actionSize();
        _dropOut = dropOut;
        _numChannels = numChannels;
        _convolutions = new SequentialBlock()
                .add(Conv2d.builder()
                        .setFilters(_numChannels)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1)).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(_numChannels)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1)).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(_numChannels)
                        .setKernelShape(new Shape(3, 3)).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(_numChannels)
                        .setKernelShape(new Shape(3, 3)).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu);
        _fcNNet = new SequentialBlock()
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(_numChannels * 2).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(_dropOut).build())
                .add(Linear.builder().setUnits(_numChannels).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(_dropOut).build());
        _toPolicy = Linear.builder()
                .setUnits(_actionSize).build();
        _toValue = Linear.builder()
                .setUnits(1).build();
        addChildBlock("convolutions", _convolutions);
        addChildBlock("fullyConnectedNNet", _fcNNet);
        addChildBlock("toPolicy", _toPolicy);
        addChildBlock("toValue", _toValue);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList data = new NDList(inputs.get(0).toType(DataType.FLOAT32, false));
        data = _convolutions.forward(parameterStore, data, training);
        data = _fcNNet.forward(parameterStore, data, training);
        NDList pi0 = _toPolicy.forward(parameterStore, data, training);
        NDList v0 = _toValue.forward(parameterStore, data, training);
        NDArray pi1 = pi0.get(0).logSoftmax(1);
        NDArray v1 = v0.get(0).tanh();
        return new NDList(pi1, v1);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        long batches = inputs[0].get(0);
        Shape piShape = new Shape(batches, _actionSize);
        Shape vShape = new Shape(batches, 1);
        return new Shape[] {piShape, vShape};
    }

    @Override
    protected void initializeChildBlocks(
            NDManager manager,
            DataType dataType,
            Shape... inputShapes) {
        _convolutions.initialize(manager, dataType, inputShapes);
        Shape[] toFC = _convolutions.getOutputShapes(inputShapes);
        _fcNNet.initialize(manager, dataType, toFC);
        Shape[] toOut = _fcNNet.getOutputShapes(toFC);
        _toPolicy.initialize(manager, dataType, toOut);
        _toValue.initialize(manager, dataType, toOut);
    }

    private long _actionSize;
    private float _dropOut;
    private int _numChannels;
    private Block _convolutions, _fcNNet;
    private Linear _toPolicy, _toValue;
}