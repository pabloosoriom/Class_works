
/**
 * Clase que contiene los métodos principales para la ejecución del juego.
 * 
 * @author Pablo Alberto Osorio Marulanda 
 * @version 27/05/2018
 */
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.Dimension;
import java.awt.Font;
import javax.swing.JPanel;
import javax.swing.JFrame;
import java.util.ArrayList;

public class Tablero extends JPanel {
    /** Atributo principal del juego-filas */
    int fil=8;
    /** Atributo principal del juego-columnas */
    int col=8;
    /** Atributo principal del juego-número de colores */
    int Ncolor=5;
    /** Atributo principal del juego-jugadas */
    int jugadas=30;
    /** Atributo principal del juego-puntaje */
    int puntaje=0;
    /** tamaño de el dot */
    int tam=50;
    /** ArrayList de tipo color para usarlos en la graficacion y como atributo de cada dot  */
    ArrayList <Color> colores=new ArrayList<Color>();
    /** Matriz principal*/
    Punto [][] matriz=new Punto[fil][col];
    /** ArrayList para guardar las coordenadas de los eventos */
    ArrayList<Coordenadas> coordenadas=new ArrayList<Coordenadas>();
    /** ArrayList que guarda la secuencia de puntos */
    ArrayList<Secuencia> secuencia=new ArrayList <Secuencia>();
    /** Botones y diseño  */
    int alejamiento=250;
    /** Botones y diseño  */
    int tamañoB=20;
    /** Botones y diseño  */
    Coordenadas Bhecho;
    /** Botones y diseño  */
    Coordenadas Bguardar;
    /** Botones y diseño  */
    Coordenadas Babrir;
    /** Método para saber que colores del tablero se usarán, llenándolos en un arrayList */
    public  void llenarColores(){
        Color [] colores1={Color.YELLOW,Color.RED,Color.BLUE,Color.GREEN,Color.PINK};
        for(int i=0;i<colores1.length;i++){
            colores.add(colores1[i]);
        }
    }

    /** Método que llena la matriz inicial con selecciones random de colores, además de la declaracion de los botones que se usarán */
    public  void llenarMatriz(){

        for(int i=0;i<fil;i++){
            for(int j=0;j<col;j++){
                int random=(int)(Math.random()*Ncolor);
                Punto p=new Punto(i,j,colores.get(random),tam,i*tam+alejamiento,j*tam+alejamiento);
                matriz[i][j]=p;
            }

        }
        //Declaracion para botones
        Bhecho=new Coordenadas(matriz[fil-1][col-1].cx,matriz[fil-1][col-1].cy+tam+10);
        Bguardar=new Coordenadas(matriz[fil-1][col-1].cx,matriz[fil-1][col-1].cy+tam+70);
        Babrir=new Coordenadas(matriz[fil-1][col-1].cx,matriz[fil-1][col-1].cy+tam+110);
    }

    /** Método que pinta la matriz anteriormente llenada en el Jpanel, además de pintar el título, botones y descripciones de botones */
    public  void generarGraficas(Graphics g){

        for(int i=0;i<matriz.length;i++){
            for(int j=0;j<matriz[0].length;j++){
                Punto p=matriz[i][j];
                g.setColor(p.color);
                g.fillOval(p.cx,p.cy,tam,tam);
                g.setColor(Color.WHITE);
                g.drawRect(p.cx,p.cy,tam,tam);
            }

        }
        //Botones
        g.setColor(Color.WHITE);
        g.fillRect(Bhecho.enX,Bhecho.enY,tamañoB,tamañoB);
        g.fillRect(Bguardar.enX,Bguardar.enY,tamañoB,tamañoB);
        g.fillRect(Babrir.enX,Babrir.enY,tamañoB,tamañoB);
        //Letra de botones
        Dimension d = this.getPreferredSize();
        int fontSize = 15;
        g.setFont(new Font("DejaVu Sans", Font.PLAIN, fontSize));
        g.setColor(Color.WHITE);
        g.drawString("Hecho/Done", Bhecho.enX+tamañoB+5, Bhecho.enY+tamañoB-5);
        g.drawString("Guardar/Save", Bguardar.enX+tamañoB+5, Bguardar.enY+tamañoB-5);
        g.drawString("Abrir/Open", Babrir.enX+tamañoB+5, Babrir.enY+tamañoB-5);
        g.setColor(Color.BLACK);
        g.fillRect(matriz[0][col-1].cx+100,matriz[0][col-1].cy+51,25,25);
        g.setColor(Color.WHITE);
        g.drawString("Puntaje/Score: "+puntaje,matriz[0][col-1].cx,matriz[0][col-1].cy+tam+20);
        g.setColor(Color.BLACK);
        g.fillRect(matriz[0][col-1].cx+110,matriz[0][col-1].cy+71,25,25);
        g.setColor(Color.WHITE);
        g.drawString("Jugadas/Moves: "+jugadas,matriz[0][col-1].cx,matriz[0][col-1].cy+tam+40);
        //Titulo

        fontSize = 50;
        g.setFont(new Font("Goudy Stout", Font.PLAIN, fontSize));

        g.setColor(Color.WHITE);
        int mitad=col/2;
        g.drawString("DOTS", (alejamiento+(tam*(mitad)))-120, matriz[0][0].cy-80);

    }

    /** Método que convierte las coordenadas ingresadas por el evento del mouse a posiciones de la matriz , esto para que se puedan usar los demás métodos con base en la raiz */
    public   void cambio(){
        for(int k=0;k<coordenadas.size();k++){
            Coordenadas c=coordenadas.get(k);

            for(int i=0;i<fil;i++){
                for(int j=0;j<col;j++){
                    //Comprobar si la cordenada en enX y enY estan en el rango de la corrdenada de cx y cy de la matriz de cada punto
                    if((c.enX>=matriz[i][j].cx&&c.enX<=matriz[i][j].cx+tam)&&(c.enY>=matriz[i][j].cy&&c.enY<=matriz[i][j].cy+tam)){
                        Secuencia sec=new Secuencia(i,j,matriz[i][j].color);
                        secuencia.add(sec);
                    }
                }
            }
        }

        coordenadas.clear();

    }

    /** Método para validar una secuencia como correcta o incorrecta (Retorna boolean)   */
    public  boolean validar(){
        boolean result=false;
        for(int i=0;i<secuencia.size()-1;i++){

            if(secuencia.get(i).color.equals(secuencia.get(i+1).color)){
                result=true;            
            }else {
                return false; 
            }

        }
        if(result==true){
            for(int i=0;i<secuencia.size()-1;i++){
                int evalF=secuencia.get(i).fila;
                int evalF2=secuencia.get(i+1).fila;
                int evalC=secuencia.get(i).columna;
                int evalC2=secuencia.get(i+1).columna;
                if(evalF!=evalF2||evalC!=evalC2){
                    if((Math.abs(evalF-evalF2)!=1)&&(Math.abs(evalC-evalC2)!=1)||(Math.abs(evalF-evalF2)==1)&&(Math.abs(evalC-evalC2)==1)){
                        return false;
                    }
                }
                result =true;
            }
        }
        return result;
    }

    /** Método para eliminar de la matriz aquellos puntos puestos como secuencia  que resultaron válidos */
    public  void eliminar(){
        boolean result=false;
        //Ciclo para verificar que cierro la secuencia (Se usará más adelante)
        int FinalCol=secuencia.get(secuencia.size()-1).columna;
        int FinalFila=secuencia.get(secuencia.size()-1).fila;
        for(int i=0;i<secuencia.size()-1;i++){
            if(FinalCol==secuencia.get(i).columna&&FinalFila==secuencia.get(i).fila){
                result=true;
                break;
            }
        }

        if(validar()){
            if(result==false){
                for(int k=0;k<matriz.length;k++){
                    for(int i=0;i<matriz[0].length;i++){
                        for(int j=0;j<secuencia.size();j++){
                            if(secuencia.get(j).columna==i&&secuencia.get(j).fila==k){
                                matriz[k][i].color=Color.BLACK;
                            }
                        }     
                    }
                }
            }else{
                for(int k=0;k<matriz.length;k++){
                    for(int i=0;i<matriz[0].length;i++){
                        Color c=secuencia.get(0).color;
                        if(matriz[k][i].color.equals(c)){
                            matriz[k][i].color=Color.BLACK;
                        }

                    }
                }
            }
        }
        secuencia.clear();
    }

    /**  Método que cuenta las posiciones sin color y las agrega al putaje*/
    public   void ContarPuntaje(){
        for(int i=0;i<matriz.length;i++){
            for(int j=0;j<matriz[0].length;j++){
                if(matriz[i][j].color.equals(Color.BLACK)){
                    puntaje++;
                }
            }
        }

    }

    /**Método que pone aquellos lugares sin color en la parte de abajo de la matriz*/
    public void acomodar(){
        for(int i=0;i<matriz.length;i++){
            for(int j=matriz[0].length-1;j>=0;j--){
                for(int k=j;k>=0;k--){
                    if(matriz[i][j].color.equals(Color.BLACK)&&!matriz[i][k].color.equals(Color.BLACK)){
                        matriz[i][j].color=matriz[i][k].color;
                        matriz[i][k].color=Color.BLACK;
                    }
                }
            }
        }
    }

    /** Método que agrega colores aleatorios a aquellas posiciones sin color y que no tienen casillas arriba */
    public   void agregar(){
        for(int i=0;i<matriz.length;i++){
            for(int j=0;j<matriz[i].length;j++){
                if(matriz[i][j].color.equals(Color.BLACK)){
                    int random=(int)(Math.random()*Ncolor);
                    matriz[i][j].color=colores.get(random);
                }
            }     
        }
    }

    /** Método que pinta "Juego terminado" cuando el usuario no tiene más juagadas */
    public void Mfinal(Graphics g){
        Dimension d = this.getPreferredSize();
        int fontSize = 50;
        g.setFont(new Font("Arial", Font.PLAIN, fontSize));

        g.setColor(Color.red);
        int mitad=col/2;
        g.drawString("GAME OVER", (alejamiento+(tam*(mitad)))-150, matriz[0][0].cy-40);

    }

    /** Arreglo para usar la coleccion de colores enla persistencia a través de la clase archivo*/
    ArrayList<String> Scolores=new ArrayList<String>();
    /**Método para agregar los colores a un string ArrayLits  */
    public void arregloColores(){
        for(int i=0;i<colores.size();i++){
            if(colores.get(i).equals(Color.YELLOW)){
                Scolores.add("Y");
            }else if(colores.get(i).equals(Color.RED)){
                Scolores.add("R");
            }else if(colores.get(i).equals(Color.BLUE)){
                Scolores.add("B");
            }else if(colores.get(i).equals(Color.GREEN)){
                Scolores.add("G");
            }else if(colores.get(i).equals(Color.PINK)){
                Scolores.add("P");
            }
        }
    }
}
