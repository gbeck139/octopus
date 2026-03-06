#ifndef QUICKVIEWER_H
#define QUICKVIEWER_H

#include <QWidget>

class QQuickWidget;

class QuickViewer : public QWidget
{
    Q_OBJECT
public:
    explicit QuickViewer(QWidget *parent = nullptr);

    void loadSTL(const QString &filePath);
    void viewFront();
    void viewTop();
    void viewSide();
    void resetView();
private:
    QQuickWidget *quickWidget;
signals:
};

#endif // QUICKVIEWER_H
