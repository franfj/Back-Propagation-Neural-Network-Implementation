package cat.urv.neuronal.fjrg.bp.layer;

import cat.urv.neuronal.fjrg.bp.neuron.HiddenUnit;

import java.util.ArrayList;
import java.util.List;

public class HiddenLayer {

    private int nUnits;
    private List<HiddenUnit> hiddenUnits;

    public HiddenLayer(int nUnits) {
        this.nUnits = nUnits;
        hiddenUnits = new ArrayList<>();
    }

    public void addHiddenUnit(HiddenUnit unit) {
        hiddenUnits.add(unit);
    }

    public int getNumberOfUnits() {
        return nUnits;
    }

    public List<HiddenUnit> getHiddenUnits() {
        return hiddenUnits;
    }

    @Override
    public String toString() {
        return "\n\nHiddenLayer{" +
                "nUnits=" + nUnits +
                ", hiddenUnits=" + hiddenUnits +
                '}';
    }

}
