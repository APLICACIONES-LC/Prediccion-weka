package proyectoclasificacion;

import java.io.IOException;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class clsInstanciaWeka {
    public Instance crearInstancia(Float temperatura, Instances train) throws IOException{
        /* El número 2, es el número de atributos que aparecen en el corpus (clima y temperatura). */
        Instance instance = new Instance(2);
        
        /* Se escribe sólo los atributos sin considerar las clases, en este caso, solo queda temperatura */
        Attribute atributo = train.attribute("temperatura");
        
        instance.setValue(atributo, temperatura);
        
        instance.setDataset(train);
        
        return instance;
    }
}
