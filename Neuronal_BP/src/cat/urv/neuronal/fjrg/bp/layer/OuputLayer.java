package cat.urv.neuronal.fjrg.bp.layer;

import cat.urv.neuronal.fjrg.bp.neuron.OutputUnit;

import java.util.ArrayList;
import java.util.List;


public class OuputLayer {

    List<OutputUnit> hiddenUnits;

    public OuputLayer() {
        hiddenUnits = new ArrayList<>();
    }

    public void addOutputUnit(OutputUnit unit) {
        hiddenUnits.add(unit);
    }

    public List<OutputUnit> getOutputUnits() {
        return hiddenUnits;
    }


}
