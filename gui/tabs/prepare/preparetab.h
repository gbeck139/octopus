#ifndef PREPARETAB_H
#define PREPARETAB_H

#include <QWidget>
#include "settingsmenuwidget.h"
#include "model3d.h"

namespace Ui {
class PrepareTab;
}

class PrepareTab : public QWidget
{
    Q_OBJECT

public:
    explicit PrepareTab(QWidget *parent = nullptr);
    ~PrepareTab();
    void setModel(Model3D* model);   // 👈 ADD THIS
    void displaySTLInViewer(const QString &filePath);
    void rotateModel(int x, int y, int z);

    SettingsMenuWidget* getSettingsMenu() const;

public slots:
    void onPrinterChanged(const QString& printerId);

private:
    Ui::PrepareTab *ui;
    Model3D* model3D = nullptr;
};

#endif // PREPARETAB_H
