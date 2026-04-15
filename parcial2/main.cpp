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
#include <opencv2/xfeatures2d.hpp>  // Requerido para FREAK (Punto 2)
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
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

    // Convertir ROI del rostro a escala de grises para la deteccion de caracteristicas
    Mat img_object_gray;
    cvtColor(img_object, img_object_gray, COLOR_BGR2GRAY);
    equalizeHist(img_object_gray, img_object_gray);

    // Crear detector/descriptor BRISK (thresh=30, octaves=3, patternScale=1.0)
    Ptr<BRISK> brisk = BRISK::create(30, 3, 1.0f);

    //Fin Parte II: ==================================================================================================================================================
    //incio Parte III: ============================= Define vectores puntos caracteristicos y descriptores visuales  ==============================================
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

    // Vectores adicionales para FREAK (Punto 2) sobre la misma ROI y sobre la escena
    std::vector<KeyPoint> keypoints_object_freak, keypoints_scene_freak;
    Mat descriptors_object_freak, descriptors_scene_freak;

    // para acceder a la coordenas de los key ponits
    //cooredenada_x=(int)keypoints_scene[i].pt.x;
    //cooredenada_y=(int)keypoints_scene[i].pt.y;

    //Fin Parte III:=============================================================================================================================================
    //incio Parte IV:============= Calcular los descriptores imagen en ROI con descriptores BRISK ========================================================================================

    // Detectar keypoints y calcular descriptores BRISK en un solo paso sobre la ROI del rostro
    brisk->detectAndCompute(img_object_gray, noArray(), keypoints_object, descriptors_object);

    cout << "[BRISK] Keypoints detectados en ROI rostro: " << keypoints_object.size() << endl;
    cout << "[BRISK] Matriz de descriptores: "
         << descriptors_object.rows << " x " << descriptors_object.cols << endl;

    // Visualizar los keypoints BRISK sobre la ROI
    Mat img_object_kp_brisk;
    drawKeypoints(img_object, keypoints_object, img_object_kp_brisk,
                  Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Punto 1 - Keypoints BRISK en ROI rostro", img_object_kp_brisk);

    //Fin IV:======================================================================================================================================================
    //::========================================  FIN PUNTO 1  ====================================================================================================================

    //***********************************************************************************************************************************************************************
    //::========================================   PUNTO 2   ====================================================================================================================
    //::===============   HACER LOS MISMO QUE EL PUNTO 1 USANDO EL DESCRIPTOR FREAK de opencv====================================================================================================================

    // FREAK es un descriptor binario, NO incluye detector propio.
    // Usamos BRISK como detector de keypoints y FREAK unicamente para describirlos.
    Ptr<FREAK> freak = FREAK::create();

    // Detectar keypoints con BRISK sobre la misma ROI del rostro
    brisk->detect(img_object_gray, keypoints_object_freak);

    // Calcular los descriptores FREAK en esos keypoints
    freak->compute(img_object_gray, keypoints_object_freak, descriptors_object_freak);

    cout << "[FREAK] Keypoints (detector BRISK) en ROI rostro: " << keypoints_object_freak.size() << endl;
    cout << "[FREAK] Matriz de descriptores: "
         << descriptors_object_freak.rows << " x " << descriptors_object_freak.cols << endl;

    // Visualizar los keypoints usados por FREAK sobre la ROI
    Mat img_object_kp_freak;
    drawKeypoints(img_object, keypoints_object_freak, img_object_kp_freak,
                  Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Punto 2 - Keypoints FREAK en ROI rostro", img_object_kp_freak);

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

      // Clonar el frame para dibujar la deteccion Haar sin contaminar la escena usada en matching
      Mat img_scene_haar = img_scene.clone();
      for (size_t i = 0; i < faces_scene.size(); i++)
      {
          rectangle(img_scene_haar, faces_scene[i], Scalar(255, 0, 255), 2);

          // Deteccion de ojos dentro del rostro
          Mat face_gray = img_scen_gris(faces_scene[i]);
          std::vector<Rect> eyes;
          eyes_cascade.detectMultiScale(face_gray, eyes, 1.1, 3, 0, Size(15, 15));
          for (size_t j = 0; j < eyes.size(); j++)
          {
              Point eye_center(faces_scene[i].x + eyes[j].x + eyes[j].width/2,
                               faces_scene[i].y + eyes[j].y + eyes[j].height/2);
              int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
              circle(img_scene_haar, eye_center, radius, Scalar(255, 0, 0), 2);
          }
      }

      imshow("Punto 1 - Deteccion Haar en video", img_scene_haar);

     //=================================================== lee video ============================================================================================================================
     //============================================== Procesar cada frame  ================================================================================================================================

      //::======================================== FIN PUNTO 1  ====================================================================================================================
      //incio Parte V:================= Calcular keyponits y descriptores imagen escena ==============================================================================

      // BRISK sobre toda la escena (frame completo en gris)
      brisk->detectAndCompute(img_scen_gris, noArray(), keypoints_scene, descriptors_scene);

       //Fin Parte V:============================================================================================================================================================
       //incio Parte VI: ================= Paso 2: Emparejar vectores descriptor basado en algoritmo DescriptorMatcher::BRUTEFORCE_SL2 ================================================================

      // BRISK produce descriptores binarios => BFMatcher con NORM_HAMMING
      std::vector<std::vector<DMatch>> knn_matches_brisk;
      std::vector<DMatch> good_matches_brisk;

      if (!descriptors_object.empty() && !descriptors_scene.empty())
      {
          BFMatcher matcher_brisk(NORM_HAMMING);
          matcher_brisk.knnMatch(descriptors_object, descriptors_scene, knn_matches_brisk, 2);

          // Filtrado con Lowe's ratio test
          const float ratio_thresh_brisk = 0.75f;
          for (size_t i = 0; i < knn_matches_brisk.size(); i++)
          {
              if (knn_matches_brisk[i].size() >= 2 &&
                  knn_matches_brisk[i][0].distance < ratio_thresh_brisk * knn_matches_brisk[i][1].distance)
              {
                  good_matches_brisk.push_back(knn_matches_brisk[i][0]);
              }
          }
      }

      //incio VII:================================= Dibujar matches o emparejamientos objeto vs escena==============================================================================================================

      // Dibujar matches filtrados objeto (ROI) vs escena completa (BRISK)
      Mat img_matches_brisk;
      drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
                  good_matches_brisk, img_matches_brisk,
                  Scalar::all(-1), Scalar::all(-1),
                  std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      //Fin VII:===============================================================================================================================================
      //incio XI:================ Filtrar keypoints objeto en la escena  ================================================================

      // Homografia con RANSAC y proyeccion de las esquinas de la ROI sobre la escena (BRISK)
      if (good_matches_brisk.size() >= 10)
      {
          std::vector<Point2f> obj_pts_brisk, scene_pts_brisk;
          for (size_t i = 0; i < good_matches_brisk.size(); i++)
          {
              obj_pts_brisk.push_back(keypoints_object[good_matches_brisk[i].queryIdx].pt);
              scene_pts_brisk.push_back(keypoints_scene[good_matches_brisk[i].trainIdx].pt);
          }

          Mat H_brisk = findHomography(obj_pts_brisk, scene_pts_brisk, RANSAC);
          if (!H_brisk.empty())
          {
              std::vector<Point2f> obj_corners(4);
              obj_corners[0] = Point2f(0.0f, 0.0f);
              obj_corners[1] = Point2f((float)img_object.cols, 0.0f);
              obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
              obj_corners[3] = Point2f(0.0f, (float)img_object.rows);

              std::vector<Point2f> scene_corners(4);
              perspectiveTransform(obj_corners, scene_corners, H_brisk);

              // Offset horizontal porque drawMatches pone la escena a la derecha del objeto
              Point2f off((float)img_object.cols, 0.0f);
              line(img_matches_brisk, scene_corners[0] + off, scene_corners[1] + off, Scalar(255, 0, 255), 4);
              line(img_matches_brisk, scene_corners[1] + off, scene_corners[2] + off, Scalar(255, 0, 255), 4);
              line(img_matches_brisk, scene_corners[2] + off, scene_corners[3] + off, Scalar(255, 0, 255), 4);
              line(img_matches_brisk, scene_corners[3] + off, scene_corners[0] + off, Scalar(255, 0, 255), 4);
          }
      }

      // ===========================================dibujar Matches key ponit del obejto y escena unicamente====================================================================================================

      imshow("Punto 1 - BRISK matches + Homografia", img_matches_brisk);

       //Fin Parte XI::============================================================================================================================================================
      //::======================================== FIN PUNTO 1  ====================================================================================================================



      //***************************************************************************************************************************************************************************
      //::========================================   PUNTO 2  ====================================================================================================================

      //::======================================== hacer los mismos paso del I al XI usando el DescriptoR FREAK  ======================================================================================
      //::========================================hacer los mismos paso del I al XI====================================================================================================================

      // Parte V (FREAK): detectar keypoints con BRISK y calcular descriptores FREAK sobre toda la escena
      brisk->detect(img_scen_gris, keypoints_scene_freak);
      freak->compute(img_scen_gris, keypoints_scene_freak, descriptors_scene_freak);

      // Parte VI (FREAK): matching por fuerza bruta con NORM_HAMMING (FREAK es binario)
      std::vector<std::vector<DMatch>> knn_matches_freak;
      std::vector<DMatch> good_matches_freak;

      if (!descriptors_object_freak.empty() && !descriptors_scene_freak.empty())
      {
          BFMatcher matcher_freak(NORM_HAMMING);
          matcher_freak.knnMatch(descriptors_object_freak, descriptors_scene_freak, knn_matches_freak, 2);

          const float ratio_thresh_freak = 0.75f;
          for (size_t i = 0; i < knn_matches_freak.size(); i++)
          {
              if (knn_matches_freak[i].size() >= 2 &&
                  knn_matches_freak[i][0].distance < ratio_thresh_freak * knn_matches_freak[i][1].distance)
              {
                  good_matches_freak.push_back(knn_matches_freak[i][0]);
              }
          }
      }

      // Parte VII (FREAK): dibujar los matches
      Mat img_matches_freak;
      drawMatches(img_object, keypoints_object_freak, img_scene, keypoints_scene_freak,
                  good_matches_freak, img_matches_freak,
                  Scalar::all(-1), Scalar::all(-1),
                  std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      // Parte XI (FREAK): homografia y proyeccion de la ROI sobre la escena
      if (good_matches_freak.size() >= 10)
      {
          std::vector<Point2f> obj_pts_freak, scene_pts_freak;
          for (size_t i = 0; i < good_matches_freak.size(); i++)
          {
              obj_pts_freak.push_back(keypoints_object_freak[good_matches_freak[i].queryIdx].pt);
              scene_pts_freak.push_back(keypoints_scene_freak[good_matches_freak[i].trainIdx].pt);
          }

          Mat H_freak = findHomography(obj_pts_freak, scene_pts_freak, RANSAC);
          if (!H_freak.empty())
          {
              std::vector<Point2f> obj_corners(4);
              obj_corners[0] = Point2f(0.0f, 0.0f);
              obj_corners[1] = Point2f((float)img_object.cols, 0.0f);
              obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
              obj_corners[3] = Point2f(0.0f, (float)img_object.rows);

              std::vector<Point2f> scene_corners(4);
              perspectiveTransform(obj_corners, scene_corners, H_freak);

              Point2f off((float)img_object.cols, 0.0f);
              line(img_matches_freak, scene_corners[0] + off, scene_corners[1] + off, Scalar(255, 0, 255), 4);
              line(img_matches_freak, scene_corners[1] + off, scene_corners[2] + off, Scalar(255, 0, 255), 4);
              line(img_matches_freak, scene_corners[2] + off, scene_corners[3] + off, Scalar(255, 0, 255), 4);
              line(img_matches_freak, scene_corners[3] + off, scene_corners[0] + off, Scalar(255, 0, 255), 4);
          }
      }

      imshow("Punto 2 - FREAK matches + Homografia", img_matches_freak);

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
