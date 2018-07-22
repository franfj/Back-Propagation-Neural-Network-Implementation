package cat.urv.neuronal.fjrg.bp.neuron;

import java.util.*;


public class Unit {

    private double value;
    private double input;
    private double deltaValue;
    private Map<Integer, Double> newWeights;
    private int nPreviousNodes;
    private List<Double> weights;


    public Unit(int nPreviousNodes) {
        this.newWeights = new HashMap<>();
        this.weights = new ArrayList<>();
        for (int i = 0; i < nPreviousNodes; ++i) {
            weights.add(new Random().nextDouble() % 10 + 1);
        }

        this.input = 0;
        this.nPreviousNodes = nPreviousNodes;
    }

    public void performActivateFunction() {
        value = sigmoid(input);
    }

    public void addInput(double in, int weightIndex) {
        this.input += in * weights.get(weightIndex);
    }

    public double sigmoid(double z) {
        return 1 / (1 + Math.pow(Math.E, -z));
    }

    public double sigmoidDerivative(double z) {
        return (1 - z) * z;
    }

    public double getValue() {
        return value;
    }

    public void setInput(double input) {
        this.input = input;
    }

    public int getNumberOfPreviousNodes() {
        return nPreviousNodes;
    }

    public List<Double> getWeights() {
        return weights;
    }

    public double getDeltaValue() {
        return deltaValue;
    }

    public void setDeltaValue(double deltaValue) {
        this.deltaValue = deltaValue;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public void improveWeight(double gradient, int index) {
        double newWeight = this.getWeights().get(index) - gradient;
        newWeights.put(index, newWeight);
    }

    public void performImprovement() {
        if (newWeights != null) {
            for (Map.Entry<Integer, Double> entry : newWeights.entrySet()) {
                weights.set(entry.getKey(), entry.getValue());
            }
        }
    }

    @Override
    public String toString() {
        return "\nUnit{" +
                "value=" + value +
                '}';
    }
}
