#include "ViewerWidget.h"
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QDirectionalLight>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <QQuaternion>
#include <QVBoxLayout>
#include <Qt3DRender/QCamera>
#include <QCuboidMesh>
#include <QPlaneMesh>
#include <Qt3DCore/QEntity>
#include <QDiffuseSpecularMaterial>

ViewerWidget::ViewerWidget(QWidget *parent)
    : QWidget(parent)
{
    // -------------------- Qt3D Window --------------------
    Qt3DExtras::Qt3DWindow *view = new Qt3DExtras::Qt3DWindow();
    QWidget *container = QWidget::createWindowContainer(view);
    container->setParent(this);
    container->setMinimumSize(QSize(400, 400));
    container->show();

    // -------------------- Root Entity --------------------
    rootEntity = new Qt3DCore::QEntity();
    view->setRootEntity(rootEntity);

    // -------------------- Camera --------------------
    Qt3DRender::QCamera *camera = view->camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(30, 20, 40));
    camera->setViewCenter(QVector3D(0, 5, 0));

    // -------------------- Forward Renderer --------------------
    Qt3DExtras::QForwardRenderer *renderer = new Qt3DExtras::QForwardRenderer();
    renderer->setCamera(camera);
    renderer->setClearColor(QColor(200, 200, 200));
    view->setActiveFrameGraph(renderer);

    // -------------------- Orbit Controller --------------------
    Qt3DExtras::QOrbitCameraController *camController =
        new Qt3DExtras::QOrbitCameraController(rootEntity);
    camController->setCamera(camera);

    // -------------------- Build Volume Prism --------------------
    prismEntity = new Qt3DCore::QEntity(rootEntity);

    auto *prismMesh = new Qt3DExtras::QCuboidMesh();
    prismMesh->setXExtent(20.0f);
    prismMesh->setYExtent(10.0f);
    prismMesh->setZExtent(15.0f);

    auto *prismMat = new Qt3DExtras::QPhongMaterial();
    QColor prismColor(100, 200, 255, 100); // semi-transparent
    prismMat->setDiffuse(prismColor);
    prismMat->setAmbient(prismColor);
    prismMat->setSpecular(QColor(255, 255, 255, 100));

    cylinderTransform = new Qt3DCore::QTransform();
    cylinderTransform->setTranslation(QVector3D(0, 5, 0)); // half height above floor

    prismEntity->addComponent(prismMesh);
    prismEntity->addComponent(prismMat);
    prismEntity->addComponent(cylinderTransform);
    prismEntity->setEnabled(true);

    // -------------------- Floor / Build Plate --------------------
    // auto *plateEntity = new Qt3DCore::QEntity(rootEntity);
    // auto *plateMesh = new Qt3DExtras::QPlaneMesh();
    // plateMesh->setWidth(22.0f);
    // plateMesh->setHeight(17.0f);

    // auto *plateMat = new Qt3DExtras::QPhongMaterial();
    // plateMat->setDiffuse(QColor(50, 50, 50, 255));

    // auto *plateTransform = new Qt3DCore::QTransform();
    // plateTransform->setRotation(QQuaternion::fromEulerAngles(-90, 0, 0));

    // plateEntity->addComponent(plateMesh);
    // plateEntity->addComponent(plateMat);
    // plateEntity->addComponent(plateTransform);

    // -------------------- Axes --------------------
    createAxes();
}

void ViewerWidget::setModelVisible(bool visible)
{
    prismEntity->setEnabled(visible);
}

void ViewerWidget::setRotation(int x, int y, int z)
{
    QQuaternion q = QQuaternion::fromEulerAngles(x, y, z);
    cylinderTransform->setRotation(q);
}

void ViewerWidget::createAxes()
{
    float axisLength = 10.0f;
    float axisRadius = 0.1f;

    // -------- X Axis (Orange) --------
    auto *xEntity = new Qt3DCore::QEntity(rootEntity);
    auto *xMesh = new Qt3DExtras::QCylinderMesh();
    xMesh->setRadius(axisRadius);
    xMesh->setLength(axisLength);

    auto *xMaterial = new Qt3DExtras::QPhongMaterial();
    xMaterial->setDiffuse(QColor(255, 165, 0, 200)); // orange, slightly transparent

    auto *xTransform = new Qt3DCore::QTransform();
    xTransform->setRotation(QQuaternion::fromEulerAngles(0, 0, 90));
    xTransform->setTranslation(QVector3D(axisLength/2, 0, 0));

    xEntity->addComponent(xMesh);
    xEntity->addComponent(xMaterial);
    xEntity->addComponent(xTransform);

    // -------- Y Axis (Purple) --------
    auto *yEntity = new Qt3DCore::QEntity(rootEntity);
    auto *yMesh = new Qt3DExtras::QCylinderMesh();
    yMesh->setRadius(axisRadius);
    yMesh->setLength(axisLength);

    auto *yMaterial = new Qt3DExtras::QPhongMaterial();
    yMaterial->setDiffuse(QColor(128, 0, 128, 200)); // purple, slightly transparent

    auto *yTransform = new Qt3DCore::QTransform();
    yTransform->setTranslation(QVector3D(0, axisLength/2, 0));

    yEntity->addComponent(yMesh);
    yEntity->addComponent(yMaterial);
    yEntity->addComponent(yTransform);

    // -------- Z Axis (Cyan) --------
    auto *zEntity = new Qt3DCore::QEntity(rootEntity);
    auto *zMesh = new Qt3DExtras::QCylinderMesh();
    zMesh->setRadius(axisRadius);
    zMesh->setLength(axisLength);

    auto *zMaterial = new Qt3DExtras::QPhongMaterial();
    zMaterial->setDiffuse(QColor(0, 255, 255, 200)); // cyan, slightly transparent

    auto *zTransform = new Qt3DCore::QTransform();
    zTransform->setRotation(QQuaternion::fromEulerAngles(90, 0, 0));
    zTransform->setTranslation(QVector3D(0, 0, axisLength/2));

    zEntity->addComponent(zMesh);
    zEntity->addComponent(zMaterial);
    zEntity->addComponent(zTransform);
}
