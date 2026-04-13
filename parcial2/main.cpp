#include <stdint.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>  // No disponible en Ubuntu; requerido solo para FREAK (Punto 2)
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>


using namespace std;
using namespace cv;
// using namespace cv::xfeatures2d;  // Ver include arriba
using std::cout;
using std::endl;


String face_cascade_name ="../data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name ="../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;


int main()
{
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };

    //::========================================   PUNTO 1  ====================================================================================================================
      //incio Parte I: =================  Aca  detector objeto interes =====================================

    // Abrir el video para obtener un frame de referencia donde detectar el rostro
    VideoCapture cap_ref("../data/blais.mp4");
    if (!cap_ref.isOpened())
    {
        cout << "Unable to open video for reference frame!" << endl;
        return -1;
    }

    Mat frame_ref, gray_ref;
    std::vector<Rect> faces;

    // Buscar un frame donde se detecte un rostro (el primer frame puede estar en negro)
    for (int intentos = 0; intentos < 300; intentos++)
    {
        cap_ref >> frame_ref;
        if (frame_ref.empty())
            break;

        cvtColor(frame_ref, gray_ref, COLOR_BGR2GRAY);
        equalizeHist(gray_ref, gray_ref);

        face_cascade.detectMultiScale(gray_ref, faces, 1.1, 3, 0, Size(30, 30));
        if (!faces.empty())
        {
            cout << "Rostro detectado en frame #" << intentos << endl;
            break;
        }
    }
    cap_ref.release();

    if (faces.empty())
    {
        cout << "No se detecto ningun rostro en los primeros frames" << endl;
        return -1;
    }

    // Tomar el rostro mas grande como objeto de interes
    Rect face_roi = faces[0];
    for (size_t i = 1; i < faces.size(); i++)
    {
        if (faces[i].area() > face_roi.area())
            face_roi = faces[i];
    }

    // Extraer ROI del objeto de interes (rostro) a color
    Mat img_object = frame_ref(face_roi).clone();

    // Visualizar la deteccion del objeto de interes de referencia
    Mat frame_detect = frame_ref.clone();
    rectangle(frame_detect, face_roi, Scalar(0, 255, 0), 2);
    imshow("Objeto de interes (ROI)", img_object);
    waitKey(1);  // no bloquear, dejar la ventana abierta mientras corre el video

    //fin Parte I: ========================================================================================================================================
    //incio Parte II: =================  Paso 1: Detectar caracteristicos en el obejeto de interes usando Detector BRISK de opencv  ===================================================================






    //Fin Parte II: ==================================================================================================================================================
    //incio Parte III: ============================= Define vectores puntos caracteristicos y descriptores visuales  ==============================================
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    
    // para acceder a la coordenas de los key ponits
    //cooredenada_x=(int)keypoints_scene[i].pt.x;
    //cooredenada_y=(int)keypoints_scene[i].pt.y;





    //Fin Parte III:=============================================================================================================================================
    //incio Parte IV:============= Calcular los descriptores imagen en ROI con descriptores BRISK ========================================================================================








    //Fin IV:======================================================================================================================================================
    //::========================================  FIN PUNTO 1  ====================================================================================================================

    //***********************************************************************************************************************************************************************
    //::========================================   PUNTO 2   ====================================================================================================================
    //::===============   HACER LOS MISMO QUE EL PUNTO 1 USANDO EL DESCRIPTOR FREAK de opencv====================================================================================================================






    //::======================================== FIN  PUNTO 2   ====================================================================================================================
    //::=========================================================================================================================================================


    //=================================================== lee video ============================================================================================================================
     VideoCapture capture("../data/blais.mp4");
     if (!capture.isOpened())
        {
        cout << "Unable to open file!" << endl;
        return 0;
        }
     Mat img_scene,img_scen_gris;

 while(true)
      {
      capture >> img_scene;
      if (img_scene.empty())
          break;

      // Punto 1 - Deteccion del objeto de interes (rostro) por Haar en cada frame
      cvtColor(img_scene, img_scen_gris, COLOR_BGR2GRAY);
      equalizeHist(img_scen_gris, img_scen_gris);

      std::vector<Rect> faces_scene;
      face_cascade.detectMultiScale(img_scen_gris, faces_scene, 1.1, 3, 0, Size(30, 30));

      // Dibujar rectangulo sobre cada rostro detectado
      for (size_t i = 0; i < faces_scene.size(); i++)
      {
          rectangle(img_scene, faces_scene[i], Scalar(0, 255, 0), 2);

          // Deteccion de ojos dentro del rostro
          Mat face_gray = img_scen_gris(faces_scene[i]);
          std::vector<Rect> eyes;
          eyes_cascade.detectMultiScale(face_gray, eyes, 1.1, 3, 0, Size(15, 15));
          for (size_t j = 0; j < eyes.size(); j++)
          {
              Point eye_center(faces_scene[i].x + eyes[j].x + eyes[j].width/2,
                               faces_scene[i].y + eyes[j].y + eyes[j].height/2);
              int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
              circle(img_scene, eye_center, radius, Scalar(255, 0, 0), 2);
          }
      }

      imshow("Punto 1 - Deteccion Haar en video", img_scene);

     //=================================================== lee video ============================================================================================================================
     //============================================== Procesar cada frame  ================================================================================================================================

      //::======================================== FIN PUNTO 1  ====================================================================================================================
      //incio Parte V:================= Calcular keyponits y descriptores imagen escena ==============================================================================





       //Fin Parte V:============================================================================================================================================================
       //incio Parte VI: ================= Paso 2: Emparejar vectores descriptor basado en algoritmo DescriptorMatcher::BRUTEFORCE_SL2 ================================================================





      //incio VII:================================= Dibujar matches o emparejamientos objeto vs escena==============================================================================================================





      //Fin VII:===============================================================================================================================================
      //incio XI:================ Filtrar keypoints objeto en la escena  ================================================================




      // ===========================================dibujar Matches key ponit del obejto y escena unicamente====================================================================================================




       //Fin Parte XI::============================================================================================================================================================
      //::======================================== FIN PUNTO 1  ====================================================================================================================









      //***************************************************************************************************************************************************************************
      //::========================================   PUNTO 2  ====================================================================================================================

      //::======================================== hacer los mismos paso del I al XI usando el DescriptoR FREAK  ======================================================================================
      //::========================================hacer los mismos paso del I al XI====================================================================================================================







      //::======================================== FIN PUNTO 2  ====================================================================================================================

      //::=========================================================================================================================================================




           int keyboard = waitKey(30);
           if (keyboard == 'q' || keyboard == 27)
               break;

       //===================================================Fin lee video ============================================================================================================================
       }

return 0;
}


void FaceDetector()
{




}









































































