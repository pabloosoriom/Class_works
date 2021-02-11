
/**
 * Clase que se encarga de la ejecucion de la parte gráfica y de referenciar los objetos e invocar los métodos para la ejecucion total del juego
 * 
 * @author Pablo A. Osorio Marulanda 
 * @version 27/05/2018
 */
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import java.awt.Dimension;
import java.awt.Font;
import javax.swing.JPanel;
import javax.swing.JFrame;
import java.util.ArrayList;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.Scanner;

public class Interfaz extends JPanel implements MouseListener{
    /**Objeto tablero para ingresar a sus métodos y hacer todas las referencias respecto a él*/
    Tablero t;
    /** Objeto Archivo para acceder a la persistencia*/
    Archivo ar;
    /** Atributo para guardar la fila con el evento(MouseListener)*/
    int x=0;
    /** Atributo para guardar la columna con el evento(MouseListener)*/
    int y=0;
    /** Constructor de la clase*/
    
    public Interfaz(){
        this.t=new Tablero();
        this.ar=new Archivo();
        t.llenarColores();
        t.llenarMatriz();
        this.addMouseListener(this);
    }
    /** Paint component para la generacion de las gráficas y ejecucion del juego por método gráfico*/
    
    @Override
    public void paintComponent(Graphics g) {
            try{
            //GEneración de primera matriz en pantalla
                if(x==0&&y==0){
                t.generarGraficas(g);
            }
            //Condicionales para el uso de persistencia
            if((x>=t.Bguardar.enX&&x<=t.Bguardar.enX+t.tamañoB)&&(y>=t.Bguardar.enY&&y<=t.Bguardar.enY+t.tamañoB)){
                ar.leerGrabar(t);
                System.out.println("Su partida ha sido guardada");
            } else if((x>=t.Babrir.enX&&x<=t.Babrir.enX+t.tamañoB)&&(y>=t.Babrir.enY&&y<=t.Babrir.enY+t.tamañoB)){
                ar.abrir(t);
                t.generarGraficas(g);
            }
            //Generacion de circulo usado como referencia de los puntos seleccionados
            g.setColor(Color.BLACK);
            g.fillOval(x,y,10,10);
            Coordenadas c=new Coordenadas(x,y);
            //Condicional para la afirmacion de una secuencia termianda y empezar a ejecutar los métodos respectivos
            if((x>=t.Bhecho.enX&&x<=t.Bhecho.enX+t.tamañoB)&&(y>=t.Bhecho.enY&&y<=t.Bhecho.enY+t.tamañoB)){
                t.cambio();
                if(t.jugadas>0){
                    if(t.validar()){
                        t.eliminar();
                        t.ContarPuntaje();
                        t.acomodar();
                        t.agregar();
                        t.jugadas--;
                        t.generarGraficas(g);
                       
                    }else{
                        t.secuencia.clear();
                        t.generarGraficas(g);
                    }
                }else{
                    t.generarGraficas(g);
                    t.Mfinal( g);
                }

            }else{
                t.coordenadas.add(c);
            }
        }catch(Exception e){
         System.out.println(e.getMessage());
         System.out.println("Error");
        }

        
    }
    /** Método del mouse usado*/
    
    @Override 
    public void mouseClicked(MouseEvent e) {
        x=e.getX();
        y=e.getY();
        repaint();
    }

    @Override
    public void mouseEntered(MouseEvent e) {}

    @Override
    public void mouseExited(MouseEvent e) {}

    @Override
    public void mousePressed(MouseEvent e) {

    }

    @Override
    public void mouseReleased(MouseEvent e) {

    }
    
    

}
