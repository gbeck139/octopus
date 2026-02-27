#pragma once

#include <QWidget>
#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DRender/QMesh>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QPlaneMesh>
#include <Qt3DRender/QDirectionalLight>
#include <QVBoxLayout>
#include <QLabel>

class ViewerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ViewerWidget(QWidget *parent = nullptr);
    void loadSTL(const QString &stlPath);

private:
    Qt3DExtras::Qt3DWindow *m_view;
    Qt3DCore::QEntity *m_sceneRoot;
    Qt3DRender::QCamera *m_camera;
    Qt3DExtras::QOrbitCameraController *m_camController;
    Qt3DCore::QEntity *m_stlEntity;
    Qt3DRender::QMesh *m_stlMesh;
    Qt3DExtras::QPhongMaterial *m_stlMaterial;
    Qt3DCore::QEntity *m_groundEntity;
    Qt3DExtras::QPlaneMesh *m_groundMesh;
    Qt3DExtras::QPhongMaterial *m_groundMaterial;
};
