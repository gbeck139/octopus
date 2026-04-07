#pragma once

#include <QWidget>
#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>

class ViewerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ViewerWidget(QWidget *parent = nullptr);

    void setModelVisible(bool visible);
    void setRotation(int x, int y, int z);
    //QString loadSTL(const QString &path);
    void addSTLModel(const QString &stlPath);
    void rotateModel(int x, int y, int z);

private:
    Qt3DCore::QEntity *rootEntity;

    Qt3DCore::QEntity *buildVolumeEntity;
    Qt3DCore::QTransform *buildVolumeTransform;

    void createAxes();
    void createBuildVolume();
    void createBuildPlate();
    void fitSTLToBuildVolume(Qt3DCore::QEntity *entity);

    void recenterModel();

    // Build volume dimensions
    float buildVolumeX = 20.0f;
    float buildVolumeY = 20.0f;
    float buildVolumeZ = 20.0f;

    // --- Bounding box (VERY IMPORTANT) ---
    QVector3D modelMin;
    QVector3D modelMax;
    QVector3D modelCenter;

    // Optional: corner offset for axes
    QVector3D axesOffset = QVector3D(0, 0, 0); // default at origin, can move

    // --- Model data ---
    Qt3DCore::QEntity* modelEntity = nullptr;
    Qt3DCore::QTransform* modelTransform = nullptr;
};
