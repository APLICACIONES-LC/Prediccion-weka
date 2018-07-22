package proyectoclasificacion;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class clsClasificacion {
    private static Instances train;
    Classifier Classifier;
    Instances data;
    
    public clsClasificacion(String corpus, String modelo) throws Exception{
        Classifier = (SMO) weka.core.SerializationHelper.read(modelo);
        train = ConverterUtils.DataSource.read(corpus);
        train.setClassIndex(0); 
        data = new Instances(train);
    }
    
    public String clasificar(Float temperatura) throws Exception{
        double predicted;
        Instance instance;
        clsInstanciaWeka instancia = new clsInstanciaWeka();

        if(train.numInstances()==0){
            throw new Exception("No classifier available");
        }

        instance = instancia.crearInstancia(temperatura, data);
        Classifier.buildClassifier(data);
        predicted = Classifier.classifyInstance(instance);
        
        return train.classAttribute().value((int)predicted);
    }
}
