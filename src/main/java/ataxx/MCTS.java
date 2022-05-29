package ataxx;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

import java.util.ArrayList;
import java.util.HashMap;

public class MCTS {

    static final int CPUCT = 1;
    static final int NUM_ITER = 25;

    MCTS(Game game, NeuralNet nnet) {
        _game = game;
        _nnet = nnet;
        _stateValues = new HashMap<>();
        _stateNumVisits = new HashMap<>();
        _statePolicies = new HashMap<>();
        _stateLegalActions = new HashMap<>();
        _edgeNumVisits = new HashMap<>();
        _edgeQValues = new HashMap<>();
    }

    NDArray policy(NDArray canonical, float temperature) {
        for (int i = 0; i < NUM_ITER; i += 1) {
            search(canonical);
        }
        String state = _game.repr(canonical);
        float[] actionWeights = new float[_game.actionSize()];
        for (int a = 0; a < actionWeights.length; a += 1) {
            String edge = state + a;
            if (_edgeNumVisits.containsKey(edge)) {
                actionWeights[a] = _edgeNumVisits.get(edge);
            }
        }
        float[] policy = new float[actionWeights.length];
        if (temperature == 0) {
            ArrayList<Integer> bestActions = maxIndices(actionWeights);
            int bestAction = bestActions.get(
                    Main.RAND.nextInt(bestActions.size()));
            policy[bestAction] = 1;
            return canonical.getManager().create(policy);
        }
        float weightTotal = 0;
        for (int a = 0; a < actionWeights.length; a += 1) {
            actionWeights[a] = pow(actionWeights[a], 1 / temperature);
            weightTotal += actionWeights[a];
        }
        for (int a = 0; a < policy.length; a += 1) {
            policy[a] = actionWeights[a] / weightTotal;
        }
        return canonical.getManager().create(policy);
    }

    ArrayList<Integer> maxIndices(float[] ar) {
        ArrayList<Integer> maxes = new ArrayList<>();
        maxes.add(0);
        for (int i = 0; i < ar.length; i += 1) {
            float max = ar[maxes.get(0)], cur = ar[i];
            if (max > cur) {
                maxes = new ArrayList<>();
                maxes.add(i);
            } else if (max == cur) {
                maxes.add(i);
            }
        }
        return maxes;
    }

    float search(NDArray canonical) {
        String state = _game.repr(canonical);
        if (!_stateValues.containsKey(state)) {
            _stateValues.put(state, _game.value(canonical, 1));
        }
        if (_stateValues.get(state) != 0) {
            return -_stateValues.get(state);
        }
        if (!_statePolicies.containsKey(state)) {
            return -evaluateLeaf(canonical, state);
        }
        NDArray legal = _stateLegalActions.get(state);
        float bestUCB = -Float.MAX_VALUE;
        int bestAction = -1;
        for (int a = 0; a < _game.actionSize(); a += 1) {
            if (legal.get(a).getInt() != 0) {
                float ucb = ucb(state, a);
                if (ucb > bestUCB) {
                    bestUCB = ucb;
                    bestAction = a;
                }
            }
        }
        NDArray next = _game.next(canonical, 1, bestAction);
        next = _game.canonical(next, -1);
        float value = search(next);
        String edge = state + bestAction;
        if (_edgeQValues.containsKey(edge)) {
            int Nsa = _edgeNumVisits.get(edge);
            float Qsa = _edgeQValues.get(edge);
            _edgeQValues.put(edge, (Nsa * Qsa + value) / (Nsa + 1));
            _edgeNumVisits.put(edge, Nsa + 1);
        } else {
            _edgeQValues.put(edge, value);
            _edgeNumVisits.put(edge, 1);
        }
        _stateNumVisits.put(state, _stateNumVisits.get(state) + 1);
        return -value;
    }

    float evaluateLeaf(NDArray canonical, String state) {
        NDList prediction = _nnet.predict(canonical);
        NDArray policy = prediction.get(0);
        float value = prediction.get(1).getFloat();
        NDArray legal = _game.legalActions(canonical, 1);
        policy = policy.mul(legal);
        NDArray policyTotal = policy.sum();
        if (policyTotal.getFloat() > 0) {
            policy = policy.div(policyTotal);
        } else {
            policy = policy.add(legal);
            policy = policy.div(policy.sum());
        }
        _stateNumVisits.put(state, 0);
        _statePolicies.put(state, policy);
        _stateLegalActions.put(state, legal);
        return value;
    }

    float ucb(String state, int action) {
        String edge = state + action;
        float Ns = _stateNumVisits.get(state);
        float prob = _statePolicies.get(state)
                .get(action).getFloat();
        if (_edgeQValues.containsKey(edge)) {
            float Qsa = _edgeQValues.get(edge);
            float Nsa = _edgeNumVisits.get(edge);
            return Qsa + CPUCT * prob * sqrt(Ns) / (1 + Nsa);
        } else {
            return CPUCT * prob * sqrt(Ns + Main.EPS);

        }
    }

    float sqrt(float x) {
        return (float) Math.sqrt(x);
    }

    float pow(float a, float b) {
        return (float) Math.pow(a, b);
    }

    private Game _game;
    private NeuralNet _nnet;
    private HashMap<String, Integer> _edgeNumVisits, _stateNumVisits;
    private HashMap<String, Float> _stateValues, _edgeQValues;
    private HashMap<String, NDArray> _statePolicies, _stateLegalActions;
}
