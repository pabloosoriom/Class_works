
/**
 * Método que ejecuta el juego en su totalidad
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
public class Main
{
    
    /**Atributo para guardar el nombre del usuario, ingresado desde la consola*/
    static String nombre;
    public Main(String nombre){
        this.nombre=nombre;
    }
    /**Método main para la ejecucion del juego*/
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        System.out.println("Ingrese su nombre: ");
        nombre=sc.nextLine();
        // Crear un nuevo Frame
        JFrame frame = new JFrame("DOTS");
        // Al cerrar el frame, termina la ejecución de este programa
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Agregar un JPanel que se llama Points (esta clase)

        Interfaz ev = new Interfaz();
        frame.add(ev);
        // Asignarle tamaño
        frame.setSize(800,800);
        // Poner el frame en el centro de la pantalla
        frame.setLocationRelativeTo(null);
        // Mostrar el frame
        frame.setVisible(true);
        frame.setBackground(Color.BLACK);

    }
}
