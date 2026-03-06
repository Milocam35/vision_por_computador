#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

// Declaraciones de funciones
cv::Mat conversion_gray(cv::Mat frame);
cv::Mat conversion_yuv(cv::Mat frame);
cv::Mat conversion_hsv(cv::Mat frame);
cv::Mat conversion_hsv_a_rgb(cv::Mat frame);
cv::Mat bgr_a_rgb(cv::Mat frame);


cv::Mat bgr_a_rgb(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            resultado.at<cv::Vec3b>(i, j)[0] = frame.at<cv::Vec3b>(i, j)[2];
            resultado.at<cv::Vec3b>(i, j)[2] = frame.at<cv::Vec3b>(i, j)[0];
        }
    }
    return resultado;
}

// Implementacion de funciones
cv::Mat conversion_gray(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            uchar gray = (pixel[0] + pixel[1] + pixel[2]) / 3;
            resultado.at<cv::Vec3b>(i, j) = cv::Vec3b(gray, gray, gray);
        }
    }
    return resultado;
}

cv::Mat conversion_yuv(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);

            // OpenCV usa BGR, no RGB
            float B = pixel[0];
            float G = pixel[1];
            float R = pixel[2];

            float Y = R * 0.299 + G * 0.587 + B * 0.114;
            float U = (B - Y) * 0.492;
            float V = (R - Y) * 0.877;

            // Normalizar para visualizacion
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(Y);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(U + 128);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(V + 128);
        }
    }
    return resultado;
}

cv::Mat conversion_hsv(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);

            // OpenCV usa BGR
            float B = pixel[0] / 255.0;
            float G = pixel[1] / 255.0;
            float R = pixel[2] / 255.0;

            float max = std::max(R, std::max(G, B));
            float min = std::min(R, std::min(G, B));
            float delta = max - min;

            float H = 0, S = 0, V = max;

            if (delta != 0)
            {
                S = delta / max;

                if (max == R)
                {
                    H = 60 * fmod((G - B) / delta, 6);
                }
                else if (max == G)
                {
                    H = 60 * ((B - R) / delta + 2);
                }
                else
                {
                    H = 60 * ((R - G) / delta + 4);
                }

                if (H < 0)
                    H += 360;
            }

            // Normalizar para OpenCV (H: 0-180, S: 0-255, V: 0-255)
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(H / 2);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(S * 255);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(V * 255);
        }
    }
    return resultado;
}

cv::Mat conversion_hsv_a_rgb(cv::Mat frame)
{
    cv::Mat resultado(frame.rows, frame.cols, CV_8UC3);

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);

            // Desnormalizar de formato OpenCV a valores reales
            float H = pixel[0] * 2.0;
            float S = pixel[1] / 255.0;
            float V = pixel[2] / 255.0;

            float C = V * S;
            float X = C * (1 - fabs(fmod(H / 60.0, 2) - 1));
            float m = V - C;

            float R1, G1, B1;

            if (H < 60)
            {
                R1 = C;
                G1 = X;
                B1 = 0;
            }
            else if (H < 120)
            {
                R1 = X;
                G1 = C;
                B1 = 0;
            }
            else if (H < 180)
            {
                R1 = 0;
                G1 = C;
                B1 = X;
            }
            else if (H < 240)
            {
                R1 = 0;
                G1 = X;
                B1 = C;
            }
            else if (H < 300)
            {
                R1 = X;
                G1 = 0;
                B1 = C;
            }
            else
            {
                R1 = C;
                G1 = 0;
                B1 = X;
            }

            // Convertir a BGR (formato OpenCV) con rango 0-255
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>((B1 + m) * 255);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>((G1 + m) * 255);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>((R1 + m) * 255);
        }
    }
    return resultado;
}

int main()
{
    cv::Mat frame1;
    cv::Mat frame2;
    cv::Mat frame3;
    cv::Mat dinosaurio_a;
    cv::Mat dinosaurio_b;

    frame1 = cv::imread("../Data/lena.png");
    frame2 = cv::imread("../Data/butterfly.png");
    frame3 = cv::imread("../Data/babuino.png");
    dinosaurio_a = cv::imread("../Data/imA.bmp");
    dinosaurio_b = cv::imread("../Data/imB.png");
    if (frame1.empty() || frame2.empty() || frame3.empty())
    {
        cerr << "Error: no se pudo abrir una o mas imagenes" << endl;
        return -1;
    }

    // Redimensionar todas las imagenes al mismo tamaño
    cv::Size tamano(512, 512);
    cv::resize(frame1, frame1, tamano);
    cv::resize(frame2, frame2, tamano);
    cv::resize(frame3, frame3, tamano);

    cv::Mat gray1, hsv_babuino, borders_butterfly;
    //Conversiones a gray, hsv y yuv
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame3, hsv_babuino, cv::COLOR_BGR2HSV);

    // Efecto brillo en HSV (modificando canal V)
    cv::Mat hsv_brillante = hsv_babuino.clone();
    for (int i = 0; i < hsv_babuino.rows; i++)
    {
        for (int j = 0; j < hsv_babuino.cols; j++)
        {
            // Brillo: aumentar V en 50
            hsv_brillante.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(hsv_babuino.at<cv::Vec3b>(i, j)[2] + 50);
        }
    }

    cv::Laplacian(frame2, borders_butterfly, CV_8U);

    // Convertir de vuelta a BGR para mostrar
    cv::Mat rgb_hsv_brillante;
    cv::cvtColor(hsv_brillante, rgb_hsv_brillante, cv::COLOR_HSV2BGR);

    //cv::imshow("Lena original", frame1);
    //cv::imshow("Butterfly original", frame2);
    //cv::imshow("Butterfly bordes", borders_butterfly);
    //cv::imshow("Babuino original", frame3);
    //cv::imshow("Lena gris", gray1);
    //cv::imshow("Babuino HSV brillante", rgb_hsv_brillante);

    cv::Mat gray_bgr;
    cv::Mat resultado(frame1.rows*2, frame1.cols*3, CV_8UC3);
    cv::cvtColor(gray1, gray_bgr, cv::COLOR_GRAY2BGR);
    int cols = frame1.cols;
    int rows = frame1.rows;
    
    // Copiar cada imagen a su lugar correspondiente en el resultado
    for(int i = 0; i < resultado.rows; i++){
        for(int j = 0; j < resultado.cols; j++){
            if(i < rows && j < cols){
                resultado.at<cv::Vec3b>(i, j) = frame1.at<cv::Vec3b>(i, j);
            }else if(i < rows && (j > cols && j < cols*2)){
                resultado.at<cv::Vec3b>(i, j) = frame3.at<cv::Vec3b>(i, j);
            }else if(i < rows && j > cols*2){
                resultado.at<cv::Vec3b>(i, j) = frame2.at<cv::Vec3b>(i, j);
            }else if((i < rows*2 && i > rows) && j < cols){
                resultado.at<cv::Vec3b>(i, j) = gray_bgr.at<cv::Vec3b>(i - rows, j);
            }else if((i < rows*2 && i > rows) && (j > cols && j < cols*2)){
                resultado.at<cv::Vec3b>(i, j) = rgb_hsv_brillante.at<cv::Vec3b>(i - rows, j);
            }else{
                resultado.at<cv::Vec3b>(i, j) = borders_butterfly.at<cv::Vec3b>(i - rows, j);
            }
            
        }
    }

    cv::Mat resultado2 = frame1.clone();
    for(int i = 100; i < rows; i++){
        for(int j = 100; j < cols; j++){
            if(i < (rows - 100) && j < (cols - 100)){
                resultado2.at<cv::Vec3b>(i, j) = gray_bgr.at<cv::Vec3b>(i, j);
            }
            
        }
    };

    // Binarizar solo la región en escala de grises (umbral 127)
    cv::Mat resultado3 = resultado2.clone();
    for (int i = 100; i < rows - 100; i++)
    {
        for (int j = 100; j < cols - 100; j++)
        {
            uchar valor = gray_bgr.at<uchar>(i, j);
            uchar bin = (valor > 127) ? 255 : 0;
            resultado3.at<cv::Vec3b>(i, j) = cv::Vec3b(bin, bin, bin);
        }
    }

    imshow("Matriz resultado punto 2 a", resultado2);
    imshow("Matriz resultado punto 2 b", resultado3);

    imshow("Matriz resultado", resultado);

    cv::Mat resultado_dinosaurio1(dinosaurio_a.rows, dinosaurio_a.cols, CV_8UC3);

    for(int i = 0; i < resultado_dinosaurio1.rows; i++){
        for(int j = 0; j < resultado_dinosaurio1.cols; j++){
            resultado_dinosaurio1.at<cv::Vec3b>(i, j) = dinosaurio_b.at<cv::Vec3b>(i, j) - dinosaurio_a.at<cv::Vec3b>(i, j);
        }
    }

    imshow("Solo triceratops", resultado_dinosaurio1);

    // Bounding box del triceratops (diferencia)
    cv::Mat dino_gray;
    cv::cvtColor(resultado_dinosaurio1, dino_gray, cv::COLOR_BGR2GRAY);
    int umbral = 30;
    int min_x1 = resultado_dinosaurio1.cols, min_y1 = resultado_dinosaurio1.rows;
    int max_x1 = 0, max_y1 = 0;

    for (int i = 0; i < dino_gray.rows; i++)
    {
        for (int j = 0; j < dino_gray.cols; j++)
        {
            if (dino_gray.at<uchar>(i, j) > umbral)
            {
                if (j < min_x1) min_x1 = j;
                if (j > max_x1) max_x1 = j;
                if (i < min_y1) min_y1 = i;
                if (i > max_y1) max_y1 = i;
            }
        }
    }

    int centro_x1 = (min_x1 + max_x1) / 2;
    int centro_y1 = (min_y1 + max_y1) / 2;

    cout << "Bounding Box (diferencia): (" << min_x1 << ", " << min_y1 << ") -> (" << max_x1 << ", " << max_y1 << ")" << endl;
    cout << "Centro de masa (diferencia): (" << centro_x1 << ", " << centro_y1 << ")" << endl;

    cv::Mat dino_visual1 = dinosaurio_b.clone();
    cv::rectangle(dino_visual1, cv::Point(min_x1, min_y1), cv::Point(max_x1, max_y1), cv::Scalar(0, 255, 0), 2);
    cv::circle(dino_visual1, cv::Point(centro_x1, centro_y1), 5, cv::Scalar(0, 0, 255), -1);
    imshow("Bounding Box Triceratops (diferencia)", dino_visual1);

    // Imagen (c): triceratops con sus colores de gris originales, fondo blanco
    cv::Mat dino_mask_gray;
    cv::cvtColor(resultado_dinosaurio1, dino_mask_gray, cv::COLOR_BGR2GRAY);

    cv::Mat resultado_c(dinosaurio_b.rows, dinosaurio_b.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int i = 0; i < dinosaurio_b.rows; i++)
    {
        for (int j = 0; j < dinosaurio_b.cols; j++)
        {
            if (dino_mask_gray.at<uchar>(i, j) > 30)
            {
                resultado_c.at<cv::Vec3b>(i, j) = dinosaurio_b.at<cv::Vec3b>(i, j);
            }
        }
    }
    imshow("Triceratops colores originales", resultado_c);

    // Bounding box del triceratops a partir de imagen (c)
    cv::Mat dino_c_gray;
    cv::cvtColor(resultado_c, dino_c_gray, cv::COLOR_BGR2GRAY);

    // En resultado_c el fondo es blanco (255), el triceratops es oscuro (< 250)
    int min_x = resultado_c.cols, min_y = resultado_c.rows;
    int max_x = 0, max_y = 0;

    for (int i = 0; i < dino_c_gray.rows; i++)
    {
        for (int j = 0; j < dino_c_gray.cols; j++)
        {
            if (dino_c_gray.at<uchar>(i, j) < 250)
            {
                if (j < min_x) min_x = j;
                if (j > max_x) max_x = j;
                if (i < min_y) min_y = i;
                if (i > max_y) max_y = i;
            }
        }
    }

    int centro_x = (min_x + max_x) / 2;
    int centro_y = (min_y + max_y) / 2;

    cout << "Bounding Box: (" << min_x << ", " << min_y << ") -> (" << max_x << ", " << max_y << ")" << endl;
    cout << "Centro de masa (bounding box): (" << centro_x << ", " << centro_y << ")" << endl;

    // Dibujar bounding box y centro sobre la imagen (c)
    cv::Mat dino_visual = resultado_c.clone();
    cv::rectangle(dino_visual, cv::Point(min_x, min_y), cv::Point(max_x, max_y), cv::Scalar(0, 255, 0), 2);
    cv::circle(dino_visual, cv::Point(centro_x, centro_y), 5, cv::Scalar(0, 0, 255), -1);

    imshow("Bounding Box Triceratops", dino_visual);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}