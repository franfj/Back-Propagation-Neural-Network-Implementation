package cat.urv.neuronal.fjrg.bp.neuron;


public class OutputUnit extends Unit {

    private double expectedValue;

    public OutputUnit(int nPreviousNodes) {
        super(nPreviousNodes);
    }

    public double getExpectedValue() {
        return expectedValue;
    }

    public void setExpectedValue(double expectedValue) {
        this.expectedValue = expectedValue;
    }

}
