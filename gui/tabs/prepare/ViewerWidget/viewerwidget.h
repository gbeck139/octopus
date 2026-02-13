#ifndef VIEWERWIDGET_H
#define VIEWERWIDGET_H

#include <QWidget>

#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DRender/QMesh>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QPointLight>

class ViewerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ViewerWidget(QWidget *parent = nullptr);
    void loadSTL(const QString &filePath);

    // view helpers
    void viewFront();
    void viewTop();
    void viewSide();
    void resetView();

private:
    Qt3DExtras::Qt3DWindow *view;
    QWidget *container;

    Qt3DCore::QEntity *rootEntity;
    Qt3DCore::QEntity *modelEntity;

    Qt3DRender::QCamera *camera;
    Qt3DCore::QTransform *modelTransform;

    Qt3DCore::QEntity *lightEntity = new Qt3DCore::QEntity(rootEntity);
    // Qt3DCore::QEntity *light2Entity = new Qt3DCore::QEntity(rootEntity);


signals:
};

#endif // VIEWERWIDGET_H
