#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "appconfig.h"
#include "model3d.h"
#include "setupwizard.h"
#include "profilemanager.h"
#include "slicerrunner.h"
#include "slicerloadingdialog.h"

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
    SlicerLoadingDialog *loadingDialog;

    // these should probably be in model3d
    int currentRotX = 0;
    int currentRotY = 0;
    int currentRotZ = 0;

    // TODO: need to put this in slicerrrunner
    QString lastGeneratedGcodePath;

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
    void onRotateXClicked();
    void onRotateYClicked();
    void onRotateZClicked();
    void onRotationChanged(const QString &face);

};
#endif // MAINWINDOW_H
