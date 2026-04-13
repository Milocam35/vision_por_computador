#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    std::string base = PROJECT_DIR;

    cv::Mat img;
    int opcion = 0;
    cv::Mat resultado;
    cv::Mat kernel;

    while(opcion == 0){
        int tipoKernel = 0, tamKernel = 5;

        std::cout << "\nSeleccione forma del kernel:" << std::endl;
        std::cout << "1. Rectangulo" << std::endl;
        std::cout << "2. Elipse" << std::endl;
        std::cout << "3. Cruz" << std::endl;
        std::cin >> tipoKernel;

        std::cout << "Tamaño del kernel (impar, ej: 3, 5, 7): ";
        std::cin >> tamKernel;

        cv::MorphShapes forma;
        switch (tipoKernel) {
        case 1: forma = cv::MORPH_RECT; break;
        case 2: forma = cv::MORPH_ELLIPSE; break;
        case 3: forma = cv::MORPH_CROSS; break;
        default: forma = cv::MORPH_RECT; break;
        }
        kernel = cv::getStructuringElement(forma, cv::Size(tamKernel, tamKernel));

        std::cout << "\nSeleccione una operación:" << std::endl;
        std::cout << "0. Salir" << std::endl;
        std::cout << "1. Dilatación" << std::endl;
        std::cout << "2. Erosión" << std::endl;
        std::cout << "3. Apertura" << std::endl;
        std::cout << "4. Cierre" << std::endl;
        std::cout << "5. Gradiente Morfológico" << std::endl;
        std::cout << "6. Separar canales BGR" << std::endl;
        std::cout << "7. Partiendo a lena" << std::endl;
        std::cout << "8. Resaltar color (HSV)" << std::endl;

        std::cin >> opcion;

        switch (opcion)
        {
        case 1:
            img = cv::imread(base + "/data/Taller2/tutor.png");
            cv::dilate(img, resultado, kernel);
            break;
        case 2:
            img = cv::imread(base + "/data/Taller2/tutor.png");
            cv::erode(img, resultado, kernel);
            break;
        case 3:
            img = cv::imread(base + "/data/Taller2/ave2.png", cv::IMREAD_GRAYSCALE);
            cv::morphologyEx(img, resultado, cv::MORPH_OPEN, kernel);
            break;
        case 4:
            img = cv::imread(base + "/data/Taller2/ave.jpg", cv::IMREAD_GRAYSCALE);
            cv::morphologyEx(img, resultado, cv::MORPH_CLOSE, kernel);
            break;
        case 6: {
            img = cv::imread(base + "/data/Taller2/imagenBGR.png");
            std::vector<cv::Mat> canales;
            cv::split(img, canales);
            // canales[0] = Blue, canales[1] = Green, canales[2] = Red

            cv::Mat zeros = cv::Mat::zeros(img.size(), CV_8UC1);

            // Imagen solo con el canal azul
            std::vector<cv::Mat> azul = {canales[0], zeros, zeros};
            cv::Mat imgAzul;
            cv::merge(azul, imgAzul);

            // Imagen solo con el canal verde
            std::vector<cv::Mat> verde = {zeros, canales[1], zeros};
            cv::Mat imgVerde;
            cv::merge(verde, imgVerde);

            // Imagen solo con el canal rojo
            std::vector<cv::Mat> rojo = {zeros, zeros, canales[2]};
            cv::Mat imgRojo;
            cv::merge(rojo, imgRojo);

            cv::imshow("Azul", imgAzul);
            cv::imshow("Verde", imgVerde);
            cv::imshow("Rojo", imgRojo);
            cv::imshow("Original", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
            opcion = 0;
            continue;
        }
        case 7: {
            img = cv::imread(base + "/data/Taller2/lenanoise.png");
            cv::medianBlur(img, resultado, 5);
            break;
        }
        case 8: {
            img = cv::imread(base + "/data/Taller2/mes.jpg");

            cv::Mat hsv;
            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

            // Imagen en gris (3 canales) para el fondo
            cv::Mat gris;
            cv::cvtColor(img, gris, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gris, gris, cv::COLOR_GRAY2BGR);

            int r = 80;

            // Rojo (h=0), Verde (h=120), Azul (h=240)
            int colores[] = {0, 120, 240};
            std::string nombres[] = {"Rojo", "Verde", "Azul"};

            for (int i = 0; i < 3; i++) {
                int h = colores[i];
                int h1 = (h - r / 2 + 360) % 360;
                int h2 = (h + r / 2 + 360) % 360;
                int h1_cv = h1 / 2;
                int h2_cv = h2 / 2;

                cv::Mat mask;
                if (h1_cv <= h2_cv) {
                    cv::inRange(hsv, cv::Scalar(h1_cv, 50, 50), cv::Scalar(h2_cv, 255, 255), mask);
                } else {
                    cv::Mat mask1, mask2;
                    cv::inRange(hsv, cv::Scalar(h1_cv, 50, 50), cv::Scalar(180, 255, 255), mask1);
                    cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(h2_cv, 255, 255), mask2);
                    mask = mask1 | mask2;
                }

                cv::Mat res = gris.clone();
                img.copyTo(res, mask);
                cv::imshow(nombres[i], res);
            }

            cv::imshow("Original", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
            opcion = 0;
            continue;
        }
        default:
            opcion = 0;
        }

        cv::imshow("Imagen Original", img);
        cv::imshow("Imagen Procesada", resultado);
    }
    
    cv::waitKey(0);

    return 0;
}
