#include "ViewerWidget.h"
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QDirectionalLight>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DCore/QEntity>
#include <Qt3DRender/QMesh>
#include <Qt3DCore/QTransform>
#include <QQuaternion>
#include <QVBoxLayout>
#include <Qt3DRender/QCamera>
#include <QCuboidMesh>
#include <QPlaneMesh>
#include <Qt3DCore/QEntity>
#include <QDiffuseSpecularMaterial>
#include <Qt3DRender/QBlendEquationArguments>
#include <Qt3DRender/QBlendEquation>
#include <Qt3DRender/QNoDepthMask>
#include <QFileInfo>
#include <QDir>
#include <QUrl>
#include <QTemporaryFile>
#include <QUuid>

ViewerWidget::ViewerWidget(QWidget *parent)
    : QWidget(parent)
{
    // -------------------- Qt3D Window --------------------
    auto *view = new Qt3DExtras::Qt3DWindow();
    QWidget *container = QWidget::createWindowContainer(view);

    // ✅ FIX: Make it fill entire widget
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(container);

    // -------------------- Root Entity --------------------
    rootEntity = new Qt3DCore::QEntity();
    view->setRootEntity(rootEntity);

    // -------------------- Camera --------------------
    auto *camera = view->camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.f/9.f, 0.1f, 1000.f);
    camera->setPosition(QVector3D(buildVolumeX * 1.5f, buildVolumeY * 1.2f, buildVolumeZ * 2.0f));
    camera->setViewCenter(QVector3D(0, buildVolumeY/2, 0));

    // -------------------- Renderer --------------------
    auto *renderer = new Qt3DExtras::QForwardRenderer();
    renderer->setCamera(camera);
    renderer->setClearColor(QColor(230, 230, 230));
    view->setActiveFrameGraph(renderer);

    // -------------------- Camera Controls --------------------
    auto *camController = new Qt3DExtras::QOrbitCameraController(rootEntity);
    camController->setCamera(camera);

    // -------------------- Scene --------------------
    createBuildVolume();  // transparent box
    createBuildPlate();   // fallback / base
    createAxes();         // XYZ axes

    qDebug() << "[ViewerWidget] Initialized successfully";

}

void ViewerWidget::createBuildVolume()
{
    // Use the member variables for dimensions
    float x = buildVolumeX;
    float y = buildVolumeY;
    float z = buildVolumeZ;

    // Create parent entity and transform for the build volume
    buildVolumeEntity = new Qt3DCore::QEntity(rootEntity);
    buildVolumeTransform = new Qt3DCore::QTransform();
    buildVolumeEntity->addComponent(buildVolumeTransform);

    QColor color(100, 200, 255); // wireframe color

    // Lambda to create a single edge
    auto createEdge = [&](QVector3D start, QVector3D end)
    {
        auto *entity = new Qt3DCore::QEntity(buildVolumeEntity); // parented to buildVolumeEntity

        auto *mesh = new Qt3DExtras::QCylinderMesh();
        mesh->setRadius(0.02f);
        mesh->setLength((end - start).length());

        auto *material = new Qt3DExtras::QPhongMaterial();
        material->setDiffuse(color);

        QVector3D mid = (start + end) / 2.0f;
        QVector3D dir = (end - start).normalized();
        QQuaternion rotation = QQuaternion::rotationTo(QVector3D(0,1,0), dir);

        auto *transform = new Qt3DCore::QTransform();
        transform->setTranslation(mid);
        transform->setRotation(rotation);

        entity->addComponent(mesh);
        entity->addComponent(material);
        entity->addComponent(transform);
    };

    // 8 corners based on the member variables
    QVector3D p000(-x/2, 0, -z/2);
    QVector3D p001(-x/2, 0,  z/2);
    QVector3D p010(-x/2, y, -z/2);
    QVector3D p011(-x/2, y,  z/2);

    QVector3D p100( x/2, 0, -z/2);
    QVector3D p101( x/2, 0,  z/2);
    QVector3D p110( x/2, y, -z/2);
    QVector3D p111( x/2, y,  z/2);

    // bottom square
    createEdge(p000, p001);
    createEdge(p001, p101);
    createEdge(p101, p100);
    createEdge(p100, p000);

    // top square
    createEdge(p010, p011);
    createEdge(p011, p111);
    createEdge(p111, p110);
    createEdge(p110, p010);

    // verticals
    createEdge(p000, p010);
    createEdge(p001, p011);
    createEdge(p100, p110);
    createEdge(p101, p111);

    qDebug() << "[ViewerWidget] Wireframe build volume created with dimensions:"
             << x << y << z;
}

void ViewerWidget::createBuildPlate()
{
    auto *plateEntity = new Qt3DCore::QEntity(rootEntity);

    auto *mesh = new Qt3DExtras::QPlaneMesh();
    mesh->setWidth(buildVolumeX + 2.0f);   // a little bigger than the box
    mesh->setHeight(buildVolumeZ + 2.0f);  // a little bigger than the box

    auto *material = new Qt3DExtras::QPhongMaterial();
    material->setDiffuse(QColor(80, 80, 80)); // dark gray

    auto *transform = new Qt3DCore::QTransform();
    transform->setRotation(QQuaternion::fromEulerAngles(0, 0, 0)); // horizontal plane
    transform->setTranslation(QVector3D(0, 0, 0)); // sitting at y=0

    plateEntity->addComponent(mesh);
    plateEntity->addComponent(material);
    plateEntity->addComponent(transform);

    qDebug() << "[ViewerWidget] Build plate created (dynamic size based on build volume)";
}

void ViewerWidget::setModelVisible(bool visible)
{
    buildVolumeEntity->setEnabled(visible);
}

void ViewerWidget::setRotation(int x, int y, int z)
{
    QQuaternion q = QQuaternion::fromEulerAngles(x, y, z);
    buildVolumeTransform->setRotation(q);

    qDebug() << "[ViewerWidget] Rotation set:" << x << y << z;
}

void ViewerWidget::createAxes()
{
    float axisLength = 3.0f;
    float axisRadius = 0.08f;

    // Choose corner for axes: e.g., bottom-front-right
    QVector3D corner(-buildVolumeX/2, 0, -buildVolumeZ/2);
    corner += axesOffset; // optional extra offset


    // -------- X Axis (Red) --------
    auto *xEntity = new Qt3DCore::QEntity(rootEntity);
    auto *xMesh = new Qt3DExtras::QCylinderMesh();
    xMesh->setRadius(axisRadius);
    xMesh->setLength(axisLength);

    auto *xMaterial = new Qt3DExtras::QPhongMaterial();
    xMaterial->setAmbient(QColor(255, 0, 0));

    auto *xTransform = new Qt3DCore::QTransform();
    xTransform->setRotation(QQuaternion::fromEulerAngles(0, 0, 90));
    xTransform->setTranslation(corner + QVector3D(axisLength/2, 0, 0));

    xEntity->addComponent(xMesh);
    xEntity->addComponent(xMaterial);
    xEntity->addComponent(xTransform);

    // -------- Y Axis (Green) --------
    auto *yEntity = new Qt3DCore::QEntity(rootEntity);
    auto *yMesh = new Qt3DExtras::QCylinderMesh();
    yMesh->setRadius(axisRadius);
    yMesh->setLength(axisLength);

    auto *yMaterial = new Qt3DExtras::QPhongMaterial();
    yMaterial->setAmbient(QColor(0, 255, 0));

    auto *yTransform = new Qt3DCore::QTransform();
    yTransform->setTranslation(corner + QVector3D(0, axisLength/2, 0));

    yEntity->addComponent(yMesh);
    yEntity->addComponent(yMaterial);
    yEntity->addComponent(yTransform);

    // -------- Z Axis (Blue) --------
    auto *zEntity = new Qt3DCore::QEntity(rootEntity);
    auto *zMesh = new Qt3DExtras::QCylinderMesh();
    zMesh->setRadius(axisRadius);
    zMesh->setLength(axisLength);

    auto *zMaterial = new Qt3DExtras::QPhongMaterial();
    zMaterial->setAmbient(QColor(0, 0, 255));

    auto *zTransform = new Qt3DCore::QTransform();
    zTransform->setRotation(QQuaternion::fromEulerAngles(90, 0, 0));
    zTransform->setTranslation(corner + QVector3D(0, 0, axisLength/2));

    zEntity->addComponent(zMesh);
    zEntity->addComponent(zMaterial);
    zEntity->addComponent(zTransform);

    qDebug() << "[ViewerWidget] Axes created at corner:" << corner;
}

// QString ViewerWidget::loadSTL(const QString &path)
// {
//     QFileInfo info(path);
//     QString tempPath = QDir::temp().filePath(info.fileName());

//     // Remove old temp file if it exists
//     if (QFile::exists(tempPath))
//         QFile::remove(tempPath);

//     if (!QFile::copy(path, tempPath)) {
//         qDebug() << "[ViewerWidget] Failed to copy STL to temp:" << tempPath;
//         return QString();
//     }

//     qDebug() << "[ViewerWidget] STL copied to temp:" << tempPath;
//     return tempPath;
// }

void ViewerWidget::addSTLModel(const QString &stlPath)
{
    if (stlPath.isEmpty())
        return;

    // Remove previous model if exists
    if (modelEntity) {
        delete modelEntity;
        modelEntity = nullptr;
    }

    modelEntity = new Qt3DCore::QEntity(rootEntity);

    auto *mesh = new Qt3DRender::QMesh();
    mesh->setSource(QUrl::fromLocalFile(stlPath));

    auto *material = new Qt3DExtras::QPhongMaterial();
    material->setDiffuse(QColor(180, 180, 180));

    auto *transform = new Qt3DCore::QTransform();
    modelTransform = transform; // keep for rotations

    modelEntity->addComponent(mesh);
    modelEntity->addComponent(material);
    modelEntity->addComponent(transform);

    // Automatically scale & center inside build volume
    fitSTLToBuildVolume(modelEntity);

    qDebug() << "[ViewerWidget] STL model loaded and fitted:" << stlPath;
}

void ViewerWidget::rotateModel(int x, int y, int z)
{
    if (!modelTransform) return;

    modelTransform->setRotation(QQuaternion::fromEulerAngles(x, y, z));

}

void ViewerWidget::fitSTLToBuildVolume(Qt3DCore::QEntity *entity)
{
    if (!entity) return;

    // First, get bounding info from the STL mesh
    auto *mesh = entity->componentsOfType<Qt3DRender::QMesh>().first();
    if (!mesh) return;

    // QMesh itself doesn't provide bounds, so we'll use approximate approach:
    // Use default scale factor, e.g., 1.0 for now, later you can compute from STL metadata
    // For simplicity, we'll center it at the build volume center

    auto *transform = new Qt3DCore::QTransform();

    // Compute scale factors for X, Y, Z (just uniform scale for now)
    // This ensures the STL fits inside the build volume
    float scaleFactor = qMin(buildVolumeX, qMin(buildVolumeY, buildVolumeZ)) / 80.0f;

    //transform->setScale3D(QVector3D(scaleFactor, scaleFactor, scaleFactor));

    // Center the model in the build volume
    //transform->setTranslation(QVector3D(0, buildVolumeY / 2.0f, 0));

    // Uniform scale
    transform->setScale3D(QVector3D(scaleFactor, scaleFactor, scaleFactor));

    // Rotate STL upright (assuming STL Y-up, viewer Z-up)
    QQuaternion uprightRotation = QQuaternion::fromEulerAngles(-90, 0, 0); // rotate -90° X
    transform->setRotation(uprightRotation);

    // Center in build volume
    //QVector3D centerOffset(0, buildVolumeY / 2.0f, 0);
    QVector3D centerOffset(0, 0, 0);

    transform->setTranslation(centerOffset);

    entity->addComponent(transform);

    // Keep a reference if you want to rotate or move it later
    modelTransform = transform;

    qDebug() << "[ViewerWidget] STL fitted to build volume with scale:" << scaleFactor;
}
