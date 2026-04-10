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
#include <Qt3DRender/QBlendEquationArguments>
#include <Qt3DRender/QBlendEquation>
#include <Qt3DRender/QNoDepthMask>
#include <QFileInfo>
#include <QDir>
#include <QUrl>
#include <QTemporaryFile>
#include <QUuid>
#include <cfloat>
#include <QTimer>

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
    camera->setPosition(QVector3D(buildVolumeX * 0.0f, buildVolumeY * 1.2f, buildVolumeZ * 2.0f));
    camera->setViewCenter(QVector3D(0, buildVolumeY/2, 0));

    // -------------------- Camera Light (Slicer-style) --------------------
    auto *lightEntity = new Qt3DCore::QEntity(rootEntity);

    auto *light = new Qt3DRender::QDirectionalLight();
    light->setColor(QColor(255, 255, 255));
    light->setIntensity(0.8f);

    lightEntity->addComponent(light);

    // 🔥 Update light EVERY time camera moves
    auto updateLight = [=]() {
        QVector3D dir = (camera->viewCenter() - camera->position()).normalized();
        light->setWorldDirection(dir);
    };

    // initial direction
    updateLight();

    // update on camera movement
    QObject::connect(camera, &Qt3DRender::QCamera::positionChanged,
                     [=](const QVector3D &) { updateLight(); });

    QObject::connect(camera, &Qt3DRender::QCamera::viewCenterChanged,
                     [=](const QVector3D &) { updateLight(); });

    // -------------------- Soft Light --------------------

    auto *fillLightEntity = new Qt3DCore::QEntity(rootEntity);

    auto *fillLight = new Qt3DRender::QDirectionalLight();
    fillLight->setColor(QColor(200, 200, 200));
    fillLight->setIntensity(0.2f);

    // angled light (not camera-following)
    fillLight->setWorldDirection(QVector3D(-0.5f, -0.5f, -1.0f).normalized());

    fillLightEntity->addComponent(fillLight);

    // -------------------- Renderer --------------------
    auto *renderer = new Qt3DExtras::QForwardRenderer();
    renderer->setCamera(camera);
    renderer->setClearColor(QColor(230, 230, 230));
    view->setActiveFrameGraph(renderer);

    // -------------------- Camera Controls --------------------
    auto *camController = new Qt3DExtras::QOrbitCameraController(rootEntity);
    camController->setLinearSpeed(80.0f);   // Left-drag translation
    camController->setLookSpeed(200.0f);     // Scroll zoom in/out
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
        material->setSpecular(QColor(0, 0, 0));

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
    //material->setDiffuse(QColor(80, 80, 80)); // dark gray
    material->setDiffuse(QColor(48, 48, 48));   // plate color
    material->setSpecular(QColor(0, 0, 0));     // no shiny reflection
    material->setAmbient(QColor(48, 48, 48));   // optional: make overall lighting uniform
    material->setShininess(0.0f);               // zero shininess

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
    xMaterial->setSpecular(QColor(0, 0, 0));

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
    yMaterial->setSpecular(QColor(0, 0, 0));

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
    zMaterial->setSpecular(QColor(0, 0, 0));

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

    if (modelRoot) {
        delete modelRoot;
        modelRoot = nullptr;
    }

    // ROOT (controls final placement)
    modelRoot = new Qt3DCore::QEntity(rootEntity);
    modelRootTransform = new Qt3DCore::QTransform();
    modelRoot->addComponent(modelRootTransform);

    // CENTER ENTITY (pivot fix)
    modelCenterEntity = new Qt3DCore::QEntity(modelRoot);
    modelCenterTransform = new Qt3DCore::QTransform();
    modelCenterEntity->addComponent(modelCenterTransform);

    // MESH ENTITY
    meshEntity = new Qt3DCore::QEntity(modelCenterEntity);

    auto *mesh = new Qt3DRender::QMesh();
    mesh->setSource(QUrl::fromLocalFile(stlPath));

    auto *material = new Qt3DExtras::QPhongMaterial();
    material->setDiffuse(QColor(30, 43, 112));
    material->setAmbient(QColor(21, 33, 92));
    material->setSpecular(QColor(0, 0, 0));

    meshTransform = new Qt3DCore::QTransform();

    meshEntity->addComponent(mesh);
    meshEntity->addComponent(material);
    meshEntity->addComponent(meshTransform);

    computeBoundingBox(mesh);

    // scale
    float scaleFactor = qMin(buildVolumeX, qMin(buildVolumeY, buildVolumeZ)) / 80.0f;
    meshTransform->setScale3D(QVector3D(scaleFactor, scaleFactor, scaleFactor));

    // upright
    meshTransform->setRotation(QQuaternion::fromEulerAngles(-90, 0, 0));

    qDebug() << "MODEL LOADED";
}

void ViewerWidget::rotateModel(int x, int y, int z)
{
    if (!meshTransform) return;

    QQuaternion q = QQuaternion::fromEulerAngles(x, y, z);
    QQuaternion upright = QQuaternion::fromEulerAngles(-90, 0, 0);

    meshTransform->setRotation(upright * q); // ✅ FIXED ORDER

    recenterModel();
}

// void ViewerWidget::fitSTLToBuildVolume(Qt3DCore::QEntity *entity)
// {
//     if (!entity) return;

//     // First, get bounding info from the STL mesh
//     auto *mesh = entity->componentsOfType<Qt3DRender::QMesh>().first();
//     if (!mesh) return;

//     auto *transform = new Qt3DCore::QTransform();

//     float scaleFactor = qMin(buildVolumeX, qMin(buildVolumeY, buildVolumeZ)) / 80.0f;

//     // Uniform scale
//     transform->setScale3D(QVector3D(scaleFactor, scaleFactor, scaleFactor));

//     // Rotate STL upright (assuming STL Y-up, viewer Z-up)
//     QQuaternion uprightRotation = QQuaternion::fromEulerAngles(-90, 0, 0); // rotate -90° X
//     transform->setRotation(uprightRotation);

//     // Center in build volume
//     QVector3D centerOffset(0, 0, 0);

//     transform->setTranslation(centerOffset);

//     entity->addComponent(transform);

//     // Keep a reference if you want to rotate or move it later
//     modelTransform = transform;

//     qDebug() << "[ViewerWidget] STL fitted to build volume with scale:" << scaleFactor;
// }

void ViewerWidget::recenterModel()
{
    qDebug() << "[recenterModel] CALLED";

    if (!meshTransform || !modelRootTransform) {
        qDebug() << "missing transforms";
        return;
    }

    if (originalMin == originalMax) {
        qDebug() << "bbox not ready";
        return;
    }

    QQuaternion rot = meshTransform->rotation();
    float s = meshTransform->scale3D().x(); // uniform scale

    float minY = FLT_MAX;

    for (int i = 0; i < 8; i++) {
        QVector3D c(
            (i & 1) ? originalMax.x() : originalMin.x(),
            (i & 2) ? originalMax.y() : originalMin.y(),
            (i & 4) ? originalMax.z() : originalMin.z()
            );

        float worldY = s * rot.rotatedVector(c).y();
        minY = qMin(minY, worldY);
    }

    float rootY = modelCenter.y() - minY;

    qDebug() << "[recenterModel] rot:" << rot.toEulerAngles()
             << "minY:" << minY
             << "modelCenter.y:" << modelCenter.y()
             << "rootY:" << rootY;

    modelRootTransform->setTranslation(QVector3D(0, rootY, 0));
}

void ViewerWidget::computeBoundingBox(Qt3DRender::QMesh* mesh)
{
    if (!mesh) return;

    QObject::connect(mesh, &Qt3DRender::QMesh::statusChanged,
                     this, [=](Qt3DRender::QMesh::Status status)
                     {
                         if (status != Qt3DRender::QMesh::Ready)
                             return;

                         qDebug() << "[Mesh Ready]";

                         auto geom = mesh->geometry();
                         if (!geom) {
                             qDebug() << "geometry null";
                             return;
                         }

                         auto attr = geom->boundingVolumePositionAttribute();
                         if (!attr) {
                             qDebug() << "attr null";
                             return;
                         }

                         auto buffer = attr->buffer();
                         QByteArray data = buffer->data();

                         const float* positions = reinterpret_cast<const float*>(data.constData());
                         int count = attr->count();
                         int stride = attr->byteStride() / sizeof(float);
                         if (stride == 0) stride = 3;

                         modelMin = QVector3D(FLT_MAX, FLT_MAX, FLT_MAX);
                         modelMax = QVector3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);

                         for (int i = 0; i < count; ++i)
                         {
                             QVector3D v(
                                 positions[i * stride + 0],
                                 positions[i * stride + 1],
                                 positions[i * stride + 2]
                                 );

                             modelMin.setX(qMin(modelMin.x(), v.x()));
                             modelMin.setY(qMin(modelMin.y(), v.y()));
                             modelMin.setZ(qMin(modelMin.z(), v.z()));

                             modelMax.setX(qMax(modelMax.x(), v.x()));
                             modelMax.setY(qMax(modelMax.y(), v.y()));
                             modelMax.setZ(qMax(modelMax.z(), v.z()));
                         }

                         modelCenter = (modelMin + modelMax) / 2.0f;
                         originalMin = modelMin;
                         originalMax = modelMax;

                         qDebug() << "[BBox]"
                                  << "Min:" << modelMin
                                  << "Max:" << modelMax
                                  << "Center:" << modelCenter;

                         modelCenterTransform->setTranslation(-modelCenter);

                         recenterModel();

                         qDebug() << "[computeBoundingBox DONE]";
                     });
}
