
/**
 * Clase con el propósito de usar la persistencia en el juego, es decir guardar y abrir una partida.
 * 
 * @author Pablo A. Osorio Marulanda
 * @version 27/05/2018
 */
import java.io.*;
import java.util.Scanner;
import java.awt.Color;
public class Archivo 
{
    /**Atributo para guardar el nombre del usuario desde el main*/
    String nombre=Main.nombre;
    /**Método para grabar la partida*/
    public  void leerGrabar(Tablero t ) throws Exception{
        t.arregloColores();
        try{
            PrintStream out=new PrintStream(new File(nombre+".txt"));
            out.print(t.fil+" ");
            out.print(t.col+" ");
            out.print(t.Ncolor+" ");
            out.print(t.jugadas+" ");
            out.print(t.puntaje+" ");
            out.print(t.tam);
            out.println();
            //Graba los colores
            for(int i=0;i<t.Ncolor;i++){
                out.print(t.Scolores.get(i)+" ");
            }
            out.println();
            //Graba la matriz
            for(int i=0;i<t.matriz.length;i++){
                for(int j=0;j<t.matriz[0].length;j++){
                    if(t.matriz[i][j].color.equals(Color.YELLOW)){
                        out.print("Y ");
                    }else if(t.matriz[i][j].color.equals(Color.RED)){
                        out.print("R ");
                    }else if(t.matriz[i][j].color.equals(Color.BLUE)){
                        out.print("B ");
                    }else if(t.matriz[i][j].color.equals(Color.GREEN)){
                        out.print("G ");
                    }else if(t.matriz[i][j].color.equals(Color.PINK)){
                        out.print("P ");
                    } 
                }
                out.println();
            }
        }catch(Exception e){
            System.out.println("Error");
            System.out.println( e.getMessage());

        }
    }

    /**Método para abrir una partida guardada*/
    public  void abrir(Tablero t)throws FileNotFoundException{
        try{
            String archivo=nombre+".txt";
            Scanner input=new Scanner (new File(archivo));
            while (input.hasNextLine()){

                String line=input.nextLine();
                Scanner lineScan=new Scanner (line);
                t.fil=lineScan.nextInt();
                t.col=lineScan.nextInt();
                t.Ncolor=lineScan.nextInt();
                t.jugadas=lineScan.nextInt();
                t.puntaje=lineScan.nextInt();
                t.tam=lineScan.nextInt();
                String line2=input.nextLine();
                Scanner lineScan2=new Scanner (line2);
                String color;
                for(int i=0;i<t.Ncolor;i++){
                    color=lineScan2.next();
                    Color c=Color.BLACK;
                    if(color.equals("Y")){
                        c=Color.YELLOW;
                    }else if(color.equals("R")){
                        c=Color.RED;
                    }else if(color.equals("G")){
                        c=Color.GREEN;
                    }else if(color.equals("B")){
                        c=Color.BLUE;
                    }else if(color.equals("P")){
                        c=Color.PINK;
                    }
                    t.colores.add(c);
                }
                t.llenarColores();
                t.llenarMatriz();
                for(int i=0;i<t.fil;i++){
                    String lineN=input.nextLine();
                    Scanner lineScanN=new Scanner(lineN);
                    for(int j =0;j<t.col;j++){
                        String colorM=lineScanN.next();
                        Color c=Color.BLACK;
                        if(colorM.equals("Y")){
                            c=Color.YELLOW;
                        }else if(colorM.equals("R")){
                            c=Color.RED;
                        }else if(colorM.equals("G")){
                            c=Color.GREEN;
                        }else if(colorM.equals("B")){
                            c=Color.BLUE;
                        }else if(colorM.equals("P")){
                            c=Color.PINK;
                        }
                        Punto p=new Punto(i,j,c,t.tam,i*t.tam+t.alejamiento,j*t.tam+t.alejamiento);
                        t.matriz[i][j]=p;
                    }
                }

            }
        }catch(Exception e){
            System.out.println("El archivo no existe");
        }

    }

}
