package cat.urv.neuronal.fjrg.app;

import cat.urv.neuronal.fjrg.bp.Network;
import cat.urv.neuronal.fjrg.bp.data.Data;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


public class Application {

    public static void main(String[] args) {

        List<List<Double>> x = new ArrayList<>();
        List<Double> y = new ArrayList<>();

        String trainDataPath = "data/TRAIN.txt";
        System.out.println("Train data: " + trainDataPath);
        readFile(x, y, trainDataPath);

        int[] hiddenLayersSizes = {3, 3};

        Network network = new Network(new Data(x, y), 2, hiddenLayersSizes);

        long startTime = System.currentTimeMillis();

        final int MAX_EPOCHS = 100_000;

        System.out.println("Max epochs: " + MAX_EPOCHS);

        for (int epochs = 0; epochs < MAX_EPOCHS; ++epochs) {
            network.train(false, false);
        }

        System.out.println("Execution time: " + (System.currentTimeMillis() - startTime) / 1000.0);

        x.clear();
        y.clear();

        String testDataPath = "data/TEST.txt";
        System.out.println("Test data: " + testDataPath);
        readFile(x, y, testDataPath);

        network.test(x, y);

    }

    private static void readFile(List<List<Double>> x, List<Double> y, String pathToFile) {
        try {
            // Open the file
            FileInputStream fstream = new FileInputStream(pathToFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

            String strLine;

            //Read File Line By Line
            while ((strLine = br.readLine()) != null) {
                String[] lineSplitted = strLine.split(" ");
                List<Double> aux = new ArrayList<>();

                for (int i = 0; i < lineSplitted.length; ++i) {
                    if (i == lineSplitted.length - 1) {
                        y.add(Double.parseDouble(lineSplitted[i]));
                    } else {
                        aux.add(Double.parseDouble(lineSplitted[i]));
                    }
                }

                x.add(aux);
            }

            //Close the input stream
            br.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
