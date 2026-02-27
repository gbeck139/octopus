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

ViewerWidget::ViewerWidget(QWidget *parent)
    : QWidget(parent)
{
    // Create Qt3D window
    Qt3DExtras::Qt3DWindow *view = new Qt3DExtras::Qt3DWindow();
    QWidget *container = QWidget::createWindowContainer(view);
    container->setMinimumSize(QSize(400, 400));

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(container);
    setLayout(layout);

    // Root entity
    rootEntity = new Qt3DCore::QEntity();

    view->setRootEntity(rootEntity);

    // Camera
    Qt3DRender::QCamera *camera = view->camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(0, 0, 20));
    camera->setViewCenter(QVector3D(0, 0, 0));

    //Qt3DExtras::QForwardRenderer *renderer = new Qt3DExtras::QForwardRenderer();
    //renderer->setCamera(view->camera());
    //renderer->setClearColor(QColor(53, 53, 53)); // optional background
    //view->setActiveFrameGraph(renderer);

    // Orbit controller (lets you drag rotate camera)
    Qt3DExtras::QOrbitCameraController *camController =
        new Qt3DExtras::QOrbitCameraController(rootEntity);
    camController->setCamera(camera);

    // Create cylinder (initially hidden)
    cylinderEntity = new Qt3DCore::QEntity(rootEntity);

    auto *mesh = new Qt3DExtras::QCylinderMesh();
    mesh->setRadius(3);
    mesh->setLength(8);
    mesh->setRings(50);
    mesh->setSlices(20);

    auto *material = new Qt3DExtras::QPhongMaterial();
    QColor color(100, 200, 255, 150);
    material->setDiffuse(color);
    material->setAmbient(color);
    //material->setShininess(50);

    cylinderTransform = new Qt3DCore::QTransform();
    cylinderTransform->setTranslation(QVector3D(0,0,0));
    cylinderTransform->setTranslation(QVector3D(0, 0, 0)); // Y-axis up

    cylinderEntity->addComponent(mesh);
    cylinderEntity->addComponent(material);
    cylinderEntity->addComponent(cylinderTransform);

    cylinderEntity->setEnabled(false); // hidden until STL loaded

    createAxes();
}

void ViewerWidget::setModelVisible(bool visible)
{
    cylinderEntity->setEnabled(visible);
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

    // -------- X Axis (Red) --------
    auto *xEntity = new Qt3DCore::QEntity(rootEntity);
    auto *xMesh = new Qt3DExtras::QCylinderMesh();
    xMesh->setRadius(axisRadius);
    xMesh->setLength(axisLength);

    auto *xMaterial = new Qt3DExtras::QPhongMaterial();
    xMaterial->setDiffuse(Qt::red);

    auto *xTransform = new Qt3DCore::QTransform();
    xTransform->setRotation(QQuaternion::fromEulerAngles(0, 0, 90));
    xTransform->setTranslation(QVector3D(axisLength/2, 0, 0));

    xEntity->addComponent(xMesh);
    xEntity->addComponent(xMaterial);
    xEntity->addComponent(xTransform);


    // -------- Y Axis (Green) --------
    auto *yEntity = new Qt3DCore::QEntity(rootEntity);
    auto *yMesh = new Qt3DExtras::QCylinderMesh();
    yMesh->setRadius(axisRadius);
    yMesh->setLength(axisLength);

    auto *yMaterial = new Qt3DExtras::QPhongMaterial();
    yMaterial->setDiffuse(Qt::green);

    auto *yTransform = new Qt3DCore::QTransform();
    yTransform->setTranslation(QVector3D(0, axisLength/2, 0));

    yEntity->addComponent(yMesh);
    yEntity->addComponent(yMaterial);
    yEntity->addComponent(yTransform);


    // -------- Z Axis (Blue) --------
    auto *zEntity = new Qt3DCore::QEntity(rootEntity);
    auto *zMesh = new Qt3DExtras::QCylinderMesh();
    zMesh->setRadius(axisRadius);
    zMesh->setLength(axisLength);

    auto *zMaterial = new Qt3DExtras::QPhongMaterial();
    zMaterial->setDiffuse(Qt::blue);

    auto *zTransform = new Qt3DCore::QTransform();
    zTransform->setRotation(QQuaternion::fromEulerAngles(90, 0, 0));
    zTransform->setTranslation(QVector3D(0, 0, axisLength/2));

    zEntity->addComponent(zMesh);
    zEntity->addComponent(zMaterial);
    zEntity->addComponent(zTransform);
}
