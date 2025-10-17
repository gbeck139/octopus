#include "mainwindow.h"
#include "version.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle(QString("SlicerApp (%1)").arg(APP_VERSION));
    w.show();
    return a.exec();
}
