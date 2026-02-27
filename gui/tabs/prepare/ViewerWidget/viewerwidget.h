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

private:
    Qt3DCore::QEntity *rootEntity;
    Qt3DCore::QEntity *cylinderEntity;
    Qt3DCore::QTransform *cylinderTransform;

private:
    void createAxes();
};
