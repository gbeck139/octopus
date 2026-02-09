#include "viewerwidget.h"

#include <QVBoxLayout>
#include <QUrl>
#include <Qt3DExtras/QForwardRenderer>
#include <QDebug>
#include <QFileInfo>

ViewerWidget::ViewerWidget(QWidget *parent)
    : QWidget{parent}
{
    // Qt 3D window
    view = new Qt3DExtras::Qt3DWindow();
    view->defaultFrameGraph()->setClearColor(QColor(30, 30, 30));

    container = QWidget::createWindowContainer(view, this);
    container->setMinimumSize(200, 200);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0,0,0,0);
    layout->addWidget(container);

    // Scene root
    rootEntity = new Qt3DCore::QEntity();
    view->setRootEntity(rootEntity);

    // Camera
    camera = view->camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.f/9.f, 0.1f, 1000.f);
    camera->setPosition(QVector3D(0, 0, 10));
    camera->setViewCenter(QVector3D(0, 0, 0));

    // Mouse controls
    auto *camController = new Qt3DExtras::QOrbitCameraController(rootEntity);
    camController->setLinearSpeed(50.0f);
    camController->setLookSpeed(180.0f);
    camController->setCamera(camera);

    // Model entity
    modelEntity = new Qt3DCore::QEntity(rootEntity);
    modelTransform = new Qt3DCore::QTransform();
    modelEntity->addComponent(modelTransform);

    auto *light = new Qt3DRender::QPointLight(lightEntity);
    light->setColor(Qt::white);
    light->setIntensity(1.0f);

    auto *lightTransform = new Qt3DCore::QTransform(lightEntity);
    lightTransform->setTranslation(QVector3D(10, 10, 10));

    lightEntity->addComponent(light);
    lightEntity->addComponent(lightTransform);
}

void ViewerWidget::loadSTL(const QString &filePath)
{
    QFileInfo check_file(filePath);
    if (!check_file.exists()) {
        qDebug() << "File does not exist:" << filePath;
        return;
    }

    // Clear previous model
    for (auto *c : modelEntity->components())
        modelEntity->removeComponent(c);

    modelEntity->addComponent(modelTransform);
    modelTransform->setScale(0.1f);

    auto *mesh = new Qt3DRender::QMesh(modelEntity);
    mesh->setSource(QUrl::fromLocalFile(filePath));

    auto *material = new Qt3DExtras::QPhongMaterial(modelEntity);
    material->setDiffuse(QColor(200, 200, 200));

    modelEntity->addComponent(mesh);
    modelEntity->addComponent(material);

    qDebug() << "Loaded STL:" << filePath;
}

void ViewerWidget::viewFront()
{
    camera->setPosition(QVector3D(0, 0, 10));
    camera->setViewCenter(QVector3D(0, 0, 0));
}

void ViewerWidget::viewTop()
{
    camera->setPosition(QVector3D(0, 10, 0));
    camera->setUpVector(QVector3D(0, 0, -1));
    camera->setViewCenter(QVector3D(0, 0, 0));
}

void ViewerWidget::viewSide()
{
    camera->setPosition(QVector3D(10, 0, 0));
    camera->setViewCenter(QVector3D(0, 0, 0));
}

void ViewerWidget::resetView()
{
    camera->setPosition(QVector3D(0, 0, 10));
    camera->setUpVector(QVector3D(0, 1, 0));
    camera->setViewCenter(QVector3D(0, 0, 0));
}
