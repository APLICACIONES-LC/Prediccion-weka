/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package wekadatamining;
/**
 *
 * @author ING.HENRY.PAZ
 * @correo ing.henry.14@gmail.com
 * @http://publicacioneshenry.wordpress.com
 * 
 */
import javax.swing.JOptionPane;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.rules.NNge;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.Ridor;
import weka.classifiers.rules.ZeroR;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Algoritmos {
    //variable de resultados
    String resultado = "\n";
    //variable de datos del archivo xxxx.arff
    String datosarchivo = "\n";
    //variable de la ruta del archivo
    String ruta;
    //variable de las Instances del archivo
    Instances coleccion;
    //variable del fold del cross-validation
    int crossValidation;
    //variable de tipo de evaluacion
    int tipo;
    public int getTipo() {
        return tipo;
    }
    public void setTipo(int tipo) {
        this.tipo = tipo;
    }
    public int getCrossValidation() {
        return crossValidation;
    }
    public void setCrossValidation(int crossValidation) {
        this.crossValidation = crossValidation;
    }
    public Algoritmos(String ruta) {
        this.ruta = ruta;
    }
    //Conjunto de datos del archivo .arrf
    public String datos() {
        try {
            DataSource ar = new DataSource(ruta);
            coleccion = ar.getDataSet();
            coleccion.setClassIndex(coleccion.numAttributes() - 1);
            //presentar la coleccion de entrenamiento
            datosarchivo = datosarchivo + "\nColeccion de entrenamiento:\n";
            datosarchivo = datosarchivo + coleccion.toString();
        } catch (Exception ex) {
            datosarchivo = ex.toString();
        }
        return datosarchivo;
    }
//algoritmo RIDOR
    public String ejecutarRidor() {
        resultado = "\n";
        Ridor ridor = new Ridor();
        try {
            //Construyendo el clasificador
            ridor.buildClassifier(coleccion);
            //llama al metodo para obtener la matrixConfusion
            String matrix = matrixConfusion(ridor);
            resultado = resultado + ridor.toString();
            resultado = resultado + matrix;
        } catch (Exception exception) {
            resultado = exception.toString();
            mostrarMsj();
        }
        return resultado;
    }
//algoritmo ZeroR
    public String ejecutarZeroR() {
        resultado = "\n";
        ZeroR zeror = new ZeroR();
        try {
            zeror.buildClassifier(coleccion);
            String matrix = matrixConfusion(zeror);
            resultado = resultado + zeror.toString();
            resultado = resultado + matrix;
        } catch (Exception exception) {
            resultado = exception.toString();
            mostrarMsj();
        }
        return resultado;
    }
//algoritmo Part
    public String ejecutarPart() {
        resultado = "\n";
        PART part = new PART();
        try {
            part.buildClassifier(coleccion);
            String matrix = matrixConfusion(part);
            resultado = resultado + part.toString();
            resultado = resultado + matrix;
        } catch (Exception exception) {
            resultado = exception.toString();
            mostrarMsj();
        }
        return resultado;
    }
//algoritmo NNge
    public String ejecutarNNge() {
        resultado = "\n";
        NNge nnge = new NNge();
        try {
            nnge.buildClassifier(coleccion);
            String matrix = matrixConfusion(nnge);
            resultado = resultado + nnge.toString();
            resultado = resultado + matrix;
        } catch (Exception exception) {

            resultado = exception.toString();
            mostrarMsj();
        }
        return resultado;
    }
//algoritmo M5Rules
    public String ejecutarM5Rules() {
        resultado = "\n";
        M5Rules m5rules = new M5Rules();
        try {
            m5rules.buildClassifier(coleccion);
            String matrix = matrixConfusion(m5rules);
            resultado = resultado + m5rules.toString();
            resultado = resultado + matrix;
        } catch (Exception exception) {
            resultado = exception.toString();
            mostrarMsj();
        }
        return resultado;
    }
    public void mostrarMsj() {
        JOptionPane.showMessageDialog(null, "Trate de ejecutar con otro algoritmo", "Error", JOptionPane.ERROR_MESSAGE);
    }
//Metodo para matrixConfusion
    public String matrixConfusion(Classifier classifier) {
        String matrix = "\n";
        //Modelo de Test
        Evaluation eTest;
        try {
            eTest = new Evaluation(coleccion);
            // es 1 si utiliza cross validation caso contrario utiliza todo
            // el conjungo de datos para evaluar el modelo
            if (tipo != 1) {
                Random rand = new Random(1);
                //los FOLDS para cross-validation
                int folds = getCrossValidation();
                //llamando al metodo cross-validation
                eTest.crossValidateModel(classifier, coleccion, folds, rand);
            } else {
                eTest.evaluateModel(classifier, coleccion);
            }
            //Muestra el resumen de la evaluacion
            String strSummary = eTest.toSummaryString();
            matrix = matrix + " ===== SUMARY =====\n" + strSummary;
            // obtener confusion matrix
            matrix = matrix + eTest.toMatrixString();
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(null, ex.toString(), "ERROR", JOptionPane.ERROR_MESSAGE);
        }
        return matrix;
    }
}