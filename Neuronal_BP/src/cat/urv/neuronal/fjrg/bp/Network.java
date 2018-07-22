package cat.urv.neuronal.fjrg.bp;

import cat.urv.neuronal.fjrg.bp.data.Data;
import cat.urv.neuronal.fjrg.bp.layer.HiddenLayer;
import cat.urv.neuronal.fjrg.bp.layer.InputLayer;
import cat.urv.neuronal.fjrg.bp.neuron.HiddenUnit;
import cat.urv.neuronal.fjrg.bp.neuron.OutputUnit;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Network {

    private final double LEARNING_RATE = 0.5;
    private final int DEFAULT_N_UNITS_PER_HIDDEN_LAYER = 3;

    // The data
    private Data data;

    // Network structure properties
    private int nInputNodes;
    private int nHiddenLayers;
    private int[] hiddenLayersSizes;

    // Layers
    private InputLayer inputLayer;
    private List<HiddenLayer> hiddenLayers;
    private OutputUnit outputUnit;

    private double error;

    private int index;

    public Network(Data d, int nHiddenLayers, int[] hiddenLayersSizes) {
        this.data = d;
        if (nHiddenLayers < 2) {
            this.nHiddenLayers = 2;
        } else {
            this.nHiddenLayers = nHiddenLayers;
        }
        this.hiddenLayersSizes = hiddenLayersSizes;

        if (!data.getX().isEmpty() && !data.getX().get(0).isEmpty()) {
            this.nInputNodes = data.getX().get(0).size();
            this.index = 0;
        }

        this.inputLayer = new InputLayer();
        initializeInputLayer();

        this.hiddenLayers = new ArrayList<>();
        initializeHiddenLayers();

        this.outputUnit = new OutputUnit(hiddenLayers.get(nHiddenLayers - 1).getNumberOfUnits());

    }

    public void clearInputs() {
        for (int i = 0; i < nHiddenLayers; ++i) {
            for (int j = 0; j < hiddenLayers.get(i).getNumberOfUnits(); ++j) {
                hiddenLayers.get(i).getHiddenUnits().get(j).setInput(0.0);
//                hiddenLayers.get(i).getHiddenUnits().get(j).setDeltaValue(0.0);
//                hiddenLayers.get(i).getHiddenUnits().get(j).setValue(0.0);
            }
        }

        outputUnit.setInput(0.0);
//        outputUnit.setDeltaValue(0.0);
//        outputUnit.setValue(0.0);
    }

    public void test(List<List<Double>> realX, List<Double> realY) {

        System.out.println(this.toString());

        normalizeTestData(realX, realY);

        double totalError = 0;

        for (int i = 0; i < realX.size(); ++i) {
            forwardTest(realX, realY, i);

            totalError += Math.abs(outputUnit.getExpectedValue() - outputUnit.getValue());

            clearInputs();

            double realValue = realY.get(i) * (data.getMaxYValue() - data.getMinYValue()) + data.getMinYValue();
            double valuePredicted = outputUnit.getValue() * (data.getMaxYValue() - data.getMinYValue()) + data.getMinYValue();
            System.out.println("Real value: " + realValue + " | Predicted value: " + valuePredicted);
        }

        System.out.println("Error rate: " + totalError / realY.size() * 100);

    }

    private void normalizeTestData(List<List<Double>> realX, List<Double> realY) {
        for (int i = 0; i < realX.size(); ++i) {
            for (int j = 0; j < realX.get(i).size(); ++j) {
                double normalizedValue = (realX.get(i).get(j) - data.getMinXValues().get(j)) / (data.getMaxXValues().get(j) - data.getMinXValues().get(j));
                realX.get(i).set(j, normalizedValue);
            }
        }

        for (int i = 0; i < realX.size(); ++i) {
            double normalizedValue = (realY.get(i) - data.getMinYValue()) / (data.getMaxYValue() - data.getMinYValue());
            realY.set(i, normalizedValue);
        }
    }

    public void forwardTest(List<List<Double>> realX, List<Double> realY, int i) {

        outputUnit.setExpectedValue(realY.get(i));

        for (int j = 0; j < nInputNodes; ++j) {
            inputLayer.getInputNodes().get(j).setValue(realX.get(i).get(j));
        }

        forwardInputToFirstHiddenLayer();
        forwardHiddenLayers();
        forwardLastHiddenLayerToOutputUnit();

    }

    public void train(boolean printResults, boolean showErrorRate) {

        double totalError = 0;

        while (index < data.getY().size()) {
            forward();

            if (printResults) {
                printResult();
            }

            backPropagation();
            calculateError();
            totalError += Math.abs(outputUnit.getExpectedValue() - outputUnit.getValue());

            clearInputs();

            index++;
        }

        if (showErrorRate) {
            System.out.println("Error rate: " + totalError / data.getY().size() * 100);
        }
        index = 0;

    }

    public void backPropagation() {

        calculateError();
        backPropagateOutputWeights();
        backPropagateHiddenLayersWeights();
        performImprovements();

    }

    public void performImprovements() {
        outputUnit.performImprovement();
        for (int i = 0; i < nHiddenLayers; ++i) {
            for (int j = 0; j < hiddenLayers.get(i).getNumberOfUnits(); ++j) {
                hiddenLayers.get(i).getHiddenUnits().get(j).performImprovement();
            }
        }
    }

    private void backPropagateHiddenLayersWeights() {

        for (int i = 0; i < hiddenLayers.get(nHiddenLayers - 1).getNumberOfUnits(); ++i) {
            HiddenUnit theHiddenUnit = hiddenLayers.get(nHiddenLayers - 1).getHiddenUnits().get(i);

            double gradient = outputUnit.getDeltaValue() * outputUnit.getWeights().get(i);
            gradient *= (theHiddenUnit.getValue() * (1 - theHiddenUnit.getValue()));
            theHiddenUnit.setDeltaValue(gradient);
            for (int j = 0; j < hiddenLayers.get(nHiddenLayers - 2).getNumberOfUnits(); ++j) {
                double inputValue = hiddenLayers.get(nHiddenLayers - 2).getHiddenUnits().get(j).getValue();
                theHiddenUnit.improveWeight(gradient * inputValue * this.LEARNING_RATE, j);
            }
        }

        for (int i = nHiddenLayers - 2; i > 0; --i) {
            backPropagateHiddenLayer(i);
        }

        for (int i = 0; i < hiddenLayers.get(0).getNumberOfUnits(); ++i) {
            HiddenUnit theHiddenUnit = hiddenLayers.get(0).getHiddenUnits().get(i);

            double gradient = 0.0;
            for (int j = 0; j < hiddenLayers.get(1).getNumberOfUnits(); ++j) {
                gradient += hiddenLayers.get(1).getHiddenUnits().get(j).getDeltaValue() * hiddenLayers.get(1).getHiddenUnits().get(j).getWeights().get(i);
            }

            gradient *= (theHiddenUnit.getValue() * (1 - theHiddenUnit.getValue()));
            theHiddenUnit.setDeltaValue(gradient);
            for (int j = 0; j < inputLayer.getInputNodes().size(); ++j) {
                double inputValue = inputLayer.getInputNodes().get(j).getValue();
                theHiddenUnit.improveWeight(gradient * inputValue * this.LEARNING_RATE, j);
            }
        }

    }

    private void backPropagateHiddenLayer(int aux) {

        for (int i = 0; i < hiddenLayers.get(aux).getNumberOfUnits(); ++i) {
            HiddenUnit theHiddenUnit = hiddenLayers.get(aux).getHiddenUnits().get(i);

            double gradient = 0.0;
            for (int j = 0; j < hiddenLayers.get(aux + 1).getNumberOfUnits(); ++j) {
                gradient += hiddenLayers.get(aux + 1).getHiddenUnits().get(j).getDeltaValue() * hiddenLayers.get(aux + 1).getHiddenUnits().get(j).getWeights().get(i);
            }

            gradient *= (theHiddenUnit.getValue() * (1 - theHiddenUnit.getValue()));
            theHiddenUnit.setDeltaValue(gradient);
            for (int j = 0; j < hiddenLayers.get(aux - 1).getNumberOfUnits(); ++j) {
                double inputValue = hiddenLayers.get(aux - 1).getHiddenUnits().get(j).getValue();
                theHiddenUnit.improveWeight(gradient * inputValue * this.LEARNING_RATE, j);
            }
        }

    }

    private void backPropagateOutputWeights() {

        for (int i = 0; i < outputUnit.getNumberOfPreviousNodes(); ++i) {
            double gradient = -(outputUnit.getExpectedValue() - outputUnit.getValue());
            gradient *= ((outputUnit.getValue() * (1 - outputUnit.getValue())));
            outputUnit.setDeltaValue(gradient);
            gradient *= hiddenLayers.get(nHiddenLayers - 1).getHiddenUnits().get(i).getValue();

            outputUnit.improveWeight(gradient * this.LEARNING_RATE, i);

        }
    }

    public void printResult() {
        System.out.println("--- Data: " + index + " -----------------------------------------------");
        System.out.println("Expected: " + data.getY().get(index) * (data.getMaxYValue() - data.getMinYValue()) + data.getMinYValue());
        System.out.println("Prediction: " + outputUnit.getValue() * (data.getMaxYValue() - data.getMinYValue()) + data.getMinYValue());
        calculateError();
        System.out.println("Error: " + error);
        System.out.println("-------------------------------------------------------------\n");
    }

    public void calculateError() {
        this.error = Math.pow((data.getY().get(index) - outputUnit.getValue()), 2) / 2;
    }

    public void forward() {

        outputUnit.setExpectedValue(data.getY().get(index));

        for (int i = 0; i < nInputNodes; ++i) {
            inputLayer.getInputNodes().get(i).setValue(data.getX().get(index).get(i));
        }

        forwardInputToFirstHiddenLayer();
        forwardHiddenLayers();
        forwardLastHiddenLayerToOutputUnit();

    }

    private void forwardLastHiddenLayerToOutputUnit() {
        for (int i = 0; i < hiddenLayers.get(nHiddenLayers - 1).getNumberOfUnits(); ++i) {
            outputUnit.addInput(hiddenLayers.get(nHiddenLayers - 1).getHiddenUnits().get(i).getValue(), i);
        }

        outputUnit.performActivateFunction();
    }

    private void forwardHiddenLayers() {
        for (int i = 1; i < nHiddenLayers; ++i) {
            for (int j = 0; j < hiddenLayers.get(i).getNumberOfUnits(); ++j) {
                for (int k = 0; k < hiddenLayers.get(i - 1).getNumberOfUnits(); ++k) {
                    hiddenLayers.get(i).getHiddenUnits().get(j).addInput(hiddenLayers.get(i - 1).getHiddenUnits().get(k).getValue(), k);
                }
            }

            for (int j = 0; j < hiddenLayers.get(i).getNumberOfUnits(); ++j) {
                hiddenLayers.get(i).getHiddenUnits().get(j).performActivateFunction();
            }
        }
    }

    private void forwardInputToFirstHiddenLayer() {
        for (int i = 0; i < nInputNodes; ++i) {
            for (int j = 0; j < hiddenLayers.get(0).getNumberOfUnits(); ++j) {
                hiddenLayers.get(0).getHiddenUnits().get(j).addInput(inputLayer.getInputNodes().get(i).getValue(), i);
            }
        }

        for (int i = 0; i < hiddenLayers.get(0).getNumberOfUnits(); ++i) {
            hiddenLayers.get(0).getHiddenUnits().get(i).performActivateFunction();
        }
    }

    private void initializeInputLayer() {
        for (int i = 0; i < nInputNodes; ++i) {
            inputLayer.addInputNode(new InputNode());
        }
    }

    private void initializeHiddenLayers() {
        for (int i = 0; i < nHiddenLayers; ++i) {
            int aux = 0;

            if (i < hiddenLayersSizes.length) {
                hiddenLayers.add(new HiddenLayer(hiddenLayersSizes[i]));
                aux = hiddenLayersSizes[i];
            } else {
                hiddenLayers.add(new HiddenLayer(DEFAULT_N_UNITS_PER_HIDDEN_LAYER));
                aux = DEFAULT_N_UNITS_PER_HIDDEN_LAYER;
            }

            for (int j = 0; j < aux; ++j) {
                if (i == 0) {
                    hiddenLayers.get(i).addHiddenUnit(new HiddenUnit(nInputNodes));
                } else {
                    hiddenLayers.get(i).addHiddenUnit(new HiddenUnit(hiddenLayers.get(i - 1).getNumberOfUnits()));
                }
            }
        }
    }

    @Override
    public String toString() {
        return "Network{" +
                "LEARNING_RATE=" + LEARNING_RATE +
                ", nInputNodes=" + nInputNodes +
                ", nHiddenLayers=" + nHiddenLayers +
                ", hiddenLayersSizes=" + Arrays.toString(hiddenLayersSizes) +
                '}';
    }
}
