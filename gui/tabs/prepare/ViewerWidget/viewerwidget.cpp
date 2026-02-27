#include "ViewerWidget.h"
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QDirectionalLight>
#include <Qt3DExtras/QForwardRenderer>

ViewerWidget::ViewerWidget(QWidget *parent)
    : QWidget(parent)
{
    // Layout for title + 3D view
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0,0,0,0);

    // Title
    QLabel *title = new QLabel("3D Modeller");
    title->setAlignment(Qt::AlignHCenter);
    title->setStyleSheet("font-size:24px; font-weight:bold;");
    layout->addWidget(title);

    // Qt3D window
    m_view = new Qt3DExtras::Qt3DWindow();
    QWidget *container = QWidget::createWindowContainer(m_view);
    container->setMinimumSize(800, 500);
    layout->addWidget(container);

    // Root entity
    m_sceneRoot = new Qt3DCore::QEntity();

    // Camera
    m_camera = m_view->camera();
    m_camera->lens()->setPerspectiveProjection(45.0f, float(container->width())/container->height(), 0.1f, 2000.0f);
    m_camera->setPosition(QVector3D(0, 100, 600));
    m_camera->setViewCenter(QVector3D(0, 50, 0));

    // Forward renderer (needed for background color)
    Qt3DExtras::QForwardRenderer *frameGraph = new Qt3DExtras::QForwardRenderer();
    frameGraph->setCamera(m_camera);
    frameGraph->setClearColor(QColor("lightgray"));
    m_view->setActiveFrameGraph(frameGraph);

    // Orbit camera controller
    m_camController = new Qt3DExtras::QOrbitCameraController(m_sceneRoot);
    m_camController->setCamera(m_camera);
    m_camController->setLinearSpeed(50.0f);
    m_camController->setLookSpeed(180.0f);

    // Directional light
    Qt3DRender::QDirectionalLight *light = new Qt3DRender::QDirectionalLight();
    light->setWorldDirection(QVector3D(-1, -1, -2));
    light->setColor(Qt::white);
    light->setIntensity(1.0f);
    Qt3DCore::QEntity *lightEntity = new Qt3DCore::QEntity(m_sceneRoot);
    lightEntity->addComponent(light);

    // STL Mesh entity (no path yet)
    m_stlMesh = new Qt3DRender::QMesh();
    m_stlMaterial = new Qt3DExtras::QPhongMaterial();
    m_stlMaterial->setDiffuse(QColor("lightsteelblue"));
    m_stlMaterial->setSpecular(QColor("white"));
    m_stlMaterial->setShininess(50.0f);

    Qt3DCore::QTransform *stlTransform = new Qt3DCore::QTransform();
    stlTransform->setScale3D(QVector3D(1,1,1));
    stlTransform->setRotation(QQuaternion::fromEulerAngles(0,0,0));
    stlTransform->setTranslation(QVector3D(0,0,0));

    m_stlEntity = new Qt3DCore::QEntity(m_sceneRoot);
    m_stlEntity->addComponent(m_stlMesh);
    m_stlEntity->addComponent(m_stlMaterial);
    m_stlEntity->addComponent(stlTransform);

    // Ground plane
    m_groundMesh = new Qt3DExtras::QPlaneMesh();
    m_groundMesh->setWidth(1000);
    m_groundMesh->setHeight(1000);

    Qt3DCore::QTransform *groundTransform = new Qt3DCore::QTransform();
    groundTransform->setTranslation(QVector3D(0,-100,0));
    groundTransform->setRotation(QQuaternion::fromEulerAngles(-90,0,0));

    m_groundMaterial = new Qt3DExtras::QPhongMaterial();
    m_groundMaterial->setDiffuse(QColor("lightgray"));

    m_groundEntity = new Qt3DCore::QEntity(m_sceneRoot);
    m_groundEntity->addComponent(m_groundMesh);
    m_groundEntity->addComponent(m_groundMaterial);
    m_groundEntity->addComponent(groundTransform);

    // Set root entity to Qt3D view
    m_view->setRootEntity(m_sceneRoot);
}

// Optional STL loader
void ViewerWidget::loadSTL(const QString &stlPath)
{
    if (!stlPath.isEmpty())
        m_stlMesh->setSource(QUrl::fromLocalFile(stlPath));
}
