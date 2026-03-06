#include "quickviewer.h"
#include <QQuickWidget>
#include <QVBoxLayout>
#include <QUrl>
#include <QQmlProperty>
#include <QQuickItem>
#include <QFileInfo>

QuickViewer::QuickViewer(QWidget *parent)
    : QWidget{parent}
{
    quickWidget = new QQuickWidget(this);
    quickWidget->setResizeMode(QQuickWidget::SizeRootObjectToView);
    quickWidget->setSource(QUrl("qrc:/qml/tabs/prepare/QuickViewer/QuickViewer.qml"));

    // Fill this QuickViewer
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(quickWidget);

    //quickWidget->setProperty("scale", QVector3D(0.1f, 0.1f, 0.1f));

    //auto *layout = new QVBoxLayout(this);
    //layout->setContentsMargins(0,0,0,0);
    //layout->addWidget(quickWidget);

    // Move camera in front
    //quickWidget->setProperty("position", QVector3D(0,0,300));
    //quickWidget->setProperty("eulerRotation", QVector3D(0,0,0));

    // Move camera to top
    //quickWidget->setProperty("position", QVector3D(0,300,0));
    //quickWidget->setProperty("eulerRotation", QVector3D(-90,0,0));

    // Rotate camera side
    //quickWidget->setProperty("position", QVector3D(300,0,0));
    //quickWidget->setProperty("eulerRotation", QVector3D(0,90,0));

}

void QuickViewer::loadSTL(const QString &filePath)
{
    QFileInfo check(filePath);
    if (!check.exists()) {
        qDebug() << "STL file does not exist:" << filePath;
        return;
    }

    QObject *root = quickWidget->rootObject();
    if (!root) return;

    QObject *model = root->findChild<QObject*>("model");
    if (model) {
        model->setProperty("source", "file:///" + filePath);
        model->setProperty("scale", QVector3D(0.1f, 0.1f, 0.1f));
    }
}

void QuickViewer::viewFront()
{
    QObject* root = quickWidget->rootObject();
    if (!root) return;

    QObject* camera = root->findChild<QObject*>("camera");
    if (camera) {
        camera->setProperty("position", QVector3D(0,0,300));
        camera->setProperty("eulerRotation", QVector3D(0,0,0));
    }
}

void QuickViewer::viewTop()
{
    QObject* root = quickWidget->rootObject();
    if (!root) return;

    QObject* camera = root->findChild<QObject*>("camera");
    if (camera) {
        camera->setProperty("position", QVector3D(0,300,0));
        camera->setProperty("eulerRotation", QVector3D(-90,0,0));
    }
}

void QuickViewer::viewSide()
{
    QObject* root = quickWidget->rootObject();
    if (!root) return;

    QObject* camera = root->findChild<QObject*>("camera");
    if (camera) {
        camera->setProperty("position", QVector3D(300,0,0));
        camera->setProperty("eulerRotation", QVector3D(0,90,0));
    }
}

void QuickViewer::resetView()
{
    viewFront();
}
