package cat.urv.neuronal.fjrg.bp.data;

import java.util.ArrayList;
import java.util.List;

public class Data {

    private List<List<Double>> x;
    private List<Double> y;

    private List<Double> maxXValues;
    private Double maxYValue;
    private List<Double> minXValues;
    private Double minYValue;

    public Data(List<List<Double>> x, List<Double> y) {
        this.x = x;
        this.y = y;

        initializeMaxMinValues();
        normalizeValues();
    }

    private void normalizeValues() {

        if (!x.isEmpty()) {
            for (int i = 0; i < x.size(); ++i) {
                for (int j = 0; j < minXValues.size(); ++j) {
                    double normalizedValue = (x.get(i).get(j) - minXValues.get(j)) / (maxXValues.get(j) - minXValues.get(j));
                    x.get(i).set(j, normalizedValue);
                }
            }
        }

        if (!y.isEmpty()) {
            for (int i = 0; i < y.size(); ++i) {
                double normalizedValue = (y.get(i) - minYValue) / (maxYValue - minYValue);
                y.set(i, normalizedValue);
            }
        }

    }

    private void initializeMaxMinValues() {
        this.maxXValues = new ArrayList<>();
        this.minXValues = new ArrayList<>();

        if (!x.isEmpty()) {
            for (int i = 0; i < x.get(0).size(); ++i) {
                double aux = 0.0;
                double auxMin = Double.MAX_VALUE;

                for (int j = 0; j < x.size(); ++j) {
                    if (x.get(j).get(i) > aux) {
                        aux = x.get(j).get(i);

                    }
                    if (x.get(j).get(i) < auxMin) {
                        auxMin = x.get(j).get(i);
                    }
                }
                maxXValues.add(aux);
                minXValues.add(auxMin);
            }
        }

        if (!y.isEmpty()) {
            double aux = 0.0;
            double auxMin = Double.MAX_VALUE;

            for (int i = 0; i < y.size(); ++i) {
                if (y.get(i) > aux) {
                    aux = y.get(i);
                }

                if (y.get(i) < auxMin) {
                    auxMin = y.get(i);
                }
            }
            maxYValue = aux;
            minYValue = auxMin;
        }
    }

    public List<List<Double>> getX() {
        return x;
    }

    public void setX(List<List<Double>> x) {
        this.x = x;
    }

    public List<Double> getY() {
        return y;
    }

    public void setY(List<Double> y) {
        this.y = y;
    }

    public List<Double> getMaxXValues() {
        return maxXValues;
    }

    public void setMaxXValues(List<Double> maxXValues) {
        this.maxXValues = maxXValues;
    }

    public Double getMaxYValue() {
        return maxYValue;
    }

    public void setMaxYValue(Double maxYValue) {
        this.maxYValue = maxYValue;
    }

    public List<Double> getMinXValues() {
        return minXValues;
    }

    public void setMinXValues(List<Double> minXValues) {
        this.minXValues = minXValues;
    }

    public Double getMinYValue() {
        return minYValue;
    }

    public void setMinYValue(Double minYValue) {
        this.minYValue = minYValue;
    }

}
