
/**
 * Clase que es usada para referenciar obetos de tipo punto, es decir, que tienen una fila,columna,color,tamaño y unas coordenadas en x e y en el Jpanel
 * 
 * @author Pablo A. Osorio Marulanda
 * @version 27/05/2018
 */
import java.awt.Color;
public class Punto
{
    /** Atributo para guardar la fila*/
    int fila;
    /** Atributo para guardar la columna*/
    int columna;
    /** Atributo para guardar el color*/
    Color color;
    /** Atributo para guardar el tamaño del dot*/
    int tam;
    /** Atributo para guardar la coordenada en X*/
    int cx;
    /** Atributo para guardar la coordenada en Y*/
    int cy;
    /**Constructor de la clase */
    public Punto(int fila,int columna,Color color,int tam,int cx,int cy){
        this.fila=fila;
        this.columna=columna;
        this.color=color;
        this.tam=tam;
        this.cx=cx;
        this.cy=cy;
    }
}
