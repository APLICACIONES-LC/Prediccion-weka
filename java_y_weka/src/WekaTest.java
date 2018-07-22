
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;

/*
 * Para cambiar este encabezado de licencia, elija Encabezados de licencia en Propiedades del proyecto.
 * Para cambiar este archivo de plantilla, elija Herramientas | Plantillas
 * y abra la plantilla en el editor.
 */
/**
 *
 * @author ivan
 */
public class WekaTest {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);

        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);

        return evaluation;
    }

    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        return 100 * correct / predictions.size();
    }

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }

    public static void main(String[] args) throws Exception {
        BufferedReader datafile = readDataFile("climas.txt");

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
       
        // Hacer la validación cruzada 10-split
        Instances[][] split = crossValidationSplit(data, 10);
       
        // División separada en matrices de entrenamiento y prueba
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];

        // Usa un conjunto de clasificadores
        Classifier[] models = {
            new J48(), // un árbol de decisión
            new PART(),
            new DecisionTable(),//clasificador de mayoría de tablas de decisión
            new DecisionStump() //árbol de decisión de un nivel
        };

        // Ejecutar para cada modelo
        for (Classifier model : models) {
            // Recoja todos los grupos de predicciones para el modelo actual en un FastVector
            FastVector predictions = new FastVector();
            // Para cada par dividido de prueba de entrenamiento, entrene y pruebe el clasificador
            for (int i = 0; i < trainingSplits.length; i++) {
                Evaluation validation = classify(model, trainingSplits[i], testingSplits[i]);
                
                predictions.appendElements(validation.predictions());

                // Descomente para ver el resumen de cada par de prueba de entrenamiento.
                System.err.println(Arrays.toString(models));
            }
            // Calcule la precisión general del clasificador actual en todas las divisiones
            double accuracy = calculateAccuracy(predictions);
            // Imprima el nombre y la precisión del clasificador actual de forma complicada,
            // pero de aspecto agradable.
            System.out.println("Exactitud de " + model.getClass().getSimpleName() + ": " + String.format("%.2f%%", accuracy) + "\n---------------------------------");
        }

    }

}
