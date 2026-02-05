#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "preparetab.h"
#include "printerprofiledialog.h"

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
    model3D = new Model3D(this);
    profileManager = new ProfileManager(this);
    slicerRunner = new SlicerRunner(this);

    // Set Printer Profiles
    if (!appConfig->isFirstRun()) {
        profileManager->setActivePrinter(appConfig->getActivePrinterId());
    }

    //TESTING
    //profileManager->deleteUserPrinter("cylinderOne(1)");


    // Settings, populate profiles
    SettingsMenuWidget* settingsMenu = ui->prepareTabWidget->getSettingsMenu();
    // Populate printer combo
    settingsMenu->populatePrinterCombo(profileManager->getSystemPrintersForView(), profileManager->getUserPrintersForView(), profileManager->getActivePrinter());

    //Attach Slice and Print buttons to tab bar
    //TODO: make a qwidget class for this
    //ui->tabWidget->setCornerWidget(ui->slicePrintHWidget, Qt::TopRightCorner);


    //testing purposes~~~~~~~~~~~~~~~~~~~~~
    //appConfig->setFirstRunCompleted(false);

    // Show First Run Wizard
    if (appConfig->isFirstRun()) {
        SetupWizard* firstWizard = new SetupWizard(this);
        firstWizard->setFirstRunMode(true);

        QList<PrinterViewData> printers = profileManager->getSystemPrintersForView();
        firstWizard->setAvailablePrinters(printers);

        connectWizard(firstWizard);

        firstWizard->exec();
        firstWizard->deleteLater();
    }


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
    connect(ui->sliceButton, &QPushButton::clicked, this, &MainWindow::onSliceClicked);

    // Model Connects
    connect(settingsMenu, &SettingsMenuWidget::printerSelected, appConfig, &AppConfig::setActivePrinterId);
    connect(appConfig, &AppConfig::activePrinterChanged, profileManager, &ProfileManager::setActivePrinter);
    connect(profileManager, &ProfileManager::activePrinterChanged, settingsMenu, [=](const QString& id) {
        settingsMenu->populatePrinterCombo(profileManager->getSystemPrintersForView(), profileManager->getUserPrintersForView(), id);});

    connect(settingsMenu, &SettingsMenuWidget::settingsMenuEditPrinterClicked, this, &MainWindow::onSettingsMenuEditPrinterClicked);
    connect(profileManager, &ProfileManager::printersChanged, settingsMenu, [=]() {
        settingsMenu->rebuildPrinterCombo(profileManager->getSystemPrintersForView(), profileManager->getUserPrintersForView());});
    connect(profileManager, &ProfileManager::activePrinterDataChanged, settingsMenu, &SettingsMenuWidget::refreshActivePrinterDisplay);

    // Slicer Connects
    connect(slicerRunner, &SlicerRunner::sliceFinished, this, [](const QString& path){
        qDebug() << "[MAIN SLICER] Slice finished, G-code:" << path;
    });

    connect(slicerRunner, &SlicerRunner::sliceFailed, this, [](const QString& err){
        qDebug() << "[MAIN SLICER] Slice failed:" << err;
    });

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
    SetupWizard* setupWizard = new SetupWizard(this);
    setupWizard->setFirstRunMode(false);

    QList<PrinterViewData> printers = profileManager->getSystemPrintersForView();
    setupWizard->setAvailablePrinters(printers);

    connectWizard(setupWizard);

    setupWizard->exec();
    setupWizard->deleteLater();
}

void MainWindow::onAboutClicked()
{
    QMessageBox::about(this, "About", "Slicer App");
}

void MainWindow::onSetupCompleted()
{
    if (appConfig->isFirstRun()) {
        appConfig->markFirstRunCompleted();
    }
}

void MainWindow::connectWizard(SetupWizard *wizard)
{
    connect(wizard, &SetupWizard::printerTypeSelected, appConfig, &AppConfig::setActivePrinterId);
    connect(wizard, &SetupWizard::setupCompleted, this, &MainWindow::onSetupCompleted);
    connect(wizard, &SetupWizard::printerTypeSelected, profileManager, &ProfileManager::setActivePrinter);
}

void MainWindow::onSettingsMenuEditPrinterClicked()
{
    PrinterProfile* currentPrinter = profileManager->getActivePrinterProfile();

    auto* dialog = new printerProfileDialog(currentPrinter, this);

    connect(dialog, &printerProfileDialog::saveRequested, profileManager, &ProfileManager::updateUserPrinter);
    connect(dialog, &printerProfileDialog::saveAsRequested, profileManager, &ProfileManager::addUserPrinter);

    dialog->exec();
}

void MainWindow::onSliceClicked()
{
    //auto* printer = profileManager->getActivePrinterDataForView()
    //auto* material = profileManager->getActiveMaterialProfile();
    //auto* process = profileManager->getActiveProcessProfile();

    // if (!printer || !material || !process) {
    //     qDebug() << "[MAIN] Missing profile, cannot slice";
    //     return;
    // }

    SliceParameters params;
    // params.layerHeight = process->getLayerHeight();
    // params.wallLoops = process->getWallLoops();
    // params.infillDensity = process->getInfillDensity();
    // params.supportsEnabled = process->supportsEnabled();
    // params.nozzleTemp = material->getNozzleTemp();
    // params.bedTemp = material->getBedTemp();

    qDebug() << "[MAIN] Triggering slice";

    slicerRunner->runSlice("currentStlPath", params);
}

