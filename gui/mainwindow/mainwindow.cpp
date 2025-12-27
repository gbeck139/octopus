#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "setupwizard.h"
#include "profilepage.h"

#include <QFileDialog>
#include <QMessageBox>

#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Model references
    appConfig = new AppConfig(this);
        // set python slicer path
        //appConfig->setPythonSlicerPath("");
        // read it later
        //QString path = appConfig->getPythonSlicerPath();

    model3D = new Model3D(this);

    // UI Connects
    connect(ui->actionImport, &QAction::triggered, this, &MainWindow::onImportClicked);
    connect(ui->actionExport, &QAction::triggered, this, &MainWindow::onExportClicked);
    connect(ui->actionQuit, &QAction::triggered, this, &MainWindow::onQuitClicked);
    connect(ui->actionUndo, &QAction::triggered, this, &MainWindow::onUndoClicked);
    connect(ui->actionRedo, &QAction::triggered, this, &MainWindow::onRedoClicked);
    connect(ui->actionCut, &QAction::triggered, this, &MainWindow::onCutClicked);
    connect(ui->actionCopy, &QAction::triggered, this, &MainWindow::onCopyClicked);
    connect(ui->actionPaste, &QAction::triggered, this, &MainWindow::onPasteClicked);
    connect(ui->menuView, &QMenu::triggered, this, &MainWindow::onViewClicked);
    connect(ui->menuPreferences, &QMenu::triggered, this, &MainWindow::onPreferencesClicked);
    connect(ui->actionSetupWizard, &QAction::triggered, this, &MainWindow::onSetupWizardClicked);
    connect(ui->actionAbout, &QAction::triggered, this, &MainWindow::onAboutClicked);

    // Model Connects


    // Setup Wizard
    if (appConfig->isFirstRun()) {
        SetupWizard wizard(appConfig->isFirstRun(), this);
        if (wizard.exec() == QDialog::Accepted) {
            //connect(wizard, &QWizard::finished, appConfig, &AppConfig::setFirstRunCompleted);
            appConfig->setFirstRunCompleted();
            connect(wizard.profilePage, &ProfilePage::printerTypeSelected, appConfig, &AppConfig::setDefaultPrinter);
        }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::onImportClicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Import STL", "", "STL FILES (*.stl)");
    if (!filePath.isEmpty()) {
        model3D->loadModel(filePath);
        qDebug() << "STL file loaded successfully in GUI";
    } else {
        qDebug() << "STL not loaded into GUI";
        QMessageBox::warning(this, "Import failed", "Could not import 3D model file.");
    }
}

void MainWindow::onExportClicked()
{
    qDebug() << "Export Clicked";
    // TODO: SlicerRunner's exportGCode file
    //QString filePath = QFileDialog::getSaveFileName(this, "Export G-code", "", "G-code Files (*.gcode)");
    //if (!filePath.isEmpty()) {
    //    if (!slicerRunner->exportGCode(filePath)) {
    //        QMessageBox::warning(this, "Export Failed", "Could not export G-code file.");
    //    }
    //}
}

void MainWindow::onQuitClicked()
{
    // Provides an "Are you sure" message box if a model has been loaded in
    if (model3D->isLoaded()) {
        auto result = QMessageBox::question(this, "Quit", "Are you sure you want to quit?");
        if (result == QMessageBox::Yes) {
            QApplication::quit();
        }
    } else {
        QApplication::quit();
    }
}

void MainWindow::onUndoClicked()
{
    // TODO:
}

void MainWindow::onRedoClicked()
{
    // TODO:
}

void MainWindow::onCutClicked()
{
    // TODO:
}

void MainWindow::onCopyClicked()
{

}

void MainWindow::onPasteClicked()
{
    // TODO:
}

void MainWindow::onViewClicked()
{
    // TODO:
}

void MainWindow::onPreferencesClicked()
{
    // TODO:
    // Dialog ~ general, profiles, advanced
}

void MainWindow::onSetupWizardClicked()
{
        SetupWizard wizard(appConfig->isFirstRun(), this);
        if (wizard.exec() == QDialog::Accepted) {
            connect(wizard.profilePage, &ProfilePage::printerTypeSelected, appConfig, &AppConfig::setDefaultPrinter);
            qDebug() << "UPDATE: Changed printer profile";
        }
}

void MainWindow::onAboutClicked()
{
    QMessageBox::about(this, "About", "Slicer App");
}

