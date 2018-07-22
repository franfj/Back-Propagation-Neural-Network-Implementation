package cat.urv.neuronal.fjrg.bp.layer;


import cat.urv.neuronal.fjrg.bp.InputNode;

import java.util.ArrayList;
import java.util.List;


public class InputLayer {

    List<InputNode> hiddenUnits;

    public InputLayer() {
        hiddenUnits = new ArrayList<>();
    }

    public void addInputNode(InputNode node) {
        hiddenUnits.add(node);
    }

    public List<InputNode> getInputNodes() {
        return hiddenUnits;
    }


}
