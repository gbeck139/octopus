#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "appconfig.h"
#include "model3d.h"
#include "setupwizard.h"
#include "profilemanager.h"
#include "slicerrunner.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    AppConfig *appConfig;
    Model3D *model3D;
    ProfileManager *profileManager;
    SlicerRunner *slicerRunner;

private slots:
    void onImportClicked();
    void onExportClicked();
    void onQuitClicked();
    void onUndoClicked();
    void onRedoClicked();
    void onCutClicked();
    void onCopyClicked();
    void onPasteClicked();
    void onViewClicked();
    void onPreferencesClicked();
    void onSetupWizardClicked();
    void onAboutClicked();
    void onSetupCompleted();
    void connectWizard(SetupWizard* wizard);
    void onSettingsMenuEditPrinterClicked();
    void onSliceClicked();

};
#endif // MAINWINDOW_H
