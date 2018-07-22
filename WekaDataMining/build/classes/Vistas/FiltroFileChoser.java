package Vistas;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class FiltroFileChoser extends FileFilter {

    final static String arff = "arff";
    
    public boolean accept(File f) {
        if (f.isDirectory()) {
            return true;
        }
        String s = f.getName();
        int i = s.lastIndexOf('.');

        if (i > 0 && i < s.length() - 1) {
            String extension = s.substring(i + 1).toLowerCase();
            if (arff.equalsIgnoreCase(extension)) {
                return true;
            } else {
                return false;
            }
        }

        return false;
    }

    // The description of this filter
    public String getDescription() {
        return "Archivos sql";
    }
}
