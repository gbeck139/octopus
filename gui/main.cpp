#include "mainwindow.h"
#include "version.h"

#include <QApplication>
#include <QFont>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // Set App Font
    QFont appFont("Inter");
    appFont.setPointSize(10);
    a.setFont(appFont);

    MainWindow w;
    w.setWindowTitle(QString("SlicerApp (%1)").arg(APP_VERSION));
    w.show();

    return a.exec();
}
