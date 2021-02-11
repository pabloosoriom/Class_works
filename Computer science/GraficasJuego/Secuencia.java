
/**
 * Clase que es usada para referenciar obetos de tipo secuencia, es decir, que tienen una fila,columna y color
 * 
 * @author Pablo A. Osorio Marulanda
 * @version 27/05/2018
 */
import java.awt.Color;

public class Secuencia
{
    /** Atributo para guardar la fila del objeto referenciado*/
     int fila;
     /** Atributo para guardar la columna del objeto referenciado*/
    int columna;
    /** Atributo para guardar el color del objeto referenciado*/
    Color color;
    /** Constructor de la clase*/
    public Secuencia(int fila,int columna,Color color){
        this.fila=fila;
        this.columna=columna;
        this.color=color;
        
    }
}