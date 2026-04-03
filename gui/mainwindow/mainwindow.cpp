#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "preparetab.h"
#include "printerprofiledialog.h"

#include <QFileDialog>
#include <QMessageBox>

#include <QDebug>
#include <QLabel>
#include <QWindow>
#include <QSlider>
#include <QCheckBox>

//TODO: move this to model3d model
struct ModelRotation {
    int x = 0;
    int y = 0;
    int z = 0;
};
ModelRotation currentRotation;

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
    loadingDialog = new SlicerLoadingDialog(this);

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

    ui->exportButton->setEnabled(false);

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
    connect(ui->exportButton, &QPushButton::clicked, this, &MainWindow::onExportClicked);

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

    qDebug() << "[GUI] Current PrusaSlicer Path: " + appConfig->getPrusaSlicerPath();

    connect(ui->rotateXButton, &QPushButton::clicked, this, &MainWindow::onRotateXClicked);
    connect(ui->rotateYButton, &QPushButton::clicked, this, &MainWindow::onRotateYClicked);
    connect(ui->rotateZButton, &QPushButton::clicked, this, &MainWindow::onRotateZClicked);
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
        if (model3D->loadModel(filePath)) {
            ui->exportButton->setEnabled(false);
        }
        qDebug() << "STL file loaded successfully in GUI";
    } else {
        qDebug() << "STL not loaded into GUI";
        QMessageBox::warning(this, "Import failed", "Could not import 3D model file.");
    }
}

void MainWindow::onExportClicked()
{
    qDebug() << "Export Clicked";

    // Check model is loaded
    if (!model3D || !model3D->isLoaded()) {
        QMessageBox::warning(this, "No Model", "Please load a 3D model before exporting.");
        return;
    }

    if (lastGeneratedGcodePath.isEmpty() ||
        !QFile::exists(lastGeneratedGcodePath)) {
        QMessageBox::warning(this,
                             "No G-code",
                             "Please slice the model first.");
        return;
    }

    // Where to save g-code file
    QString savePath = QFileDialog::getSaveFileName(
        this,
        "Export G-code",
        "",
        "G-code Files (*.gcode)"
        );

    if (savePath.isEmpty())
        return;

    if (!savePath.endsWith(".gcode", Qt::CaseInsensitive))
        savePath += ".gcode";

    if (!QFile::copy(lastGeneratedGcodePath, savePath)) {
        QMessageBox::warning(this,
                             "Export Failed",
                             "Could not save file.");
        return;
    }

    QMessageBox::information(this,
                             "Export Complete",
                             "G-code saved successfully.");




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
    //TODO: if model has been exported/saved, then it's ok?
    //TODO: override closeEvent - just copy and paste, you can't connect together cuz quit() will make it a loop
        // In closeEvent: kill visualizerprocess:
        // if (visualizerProcess) {
        //     if (visualizerProcess->state() != QProcess::NotRunning) {
        //         visualizerProcess->kill();              // force kill
        //         visualizerProcess->waitForFinished();   // block until it's dead
        //     }
        // }

        // event->accept();
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
    connect(wizard, &SetupWizard::printerTypeSelected, profileManager, &ProfileManager::setActivePrinter);
    connect(wizard, &SetupWizard::prusaSlicerPathSelected, appConfig, &AppConfig::setPrusaSlicerPath);
    connect(wizard, &SetupWizard::setupCompleted, this, &MainWindow::onSetupCompleted);
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
    qDebug() << "================ START SLICE ================";

    if (!model3D || !model3D->isLoaded()) {
        qDebug() << "[ERROR] Model not loaded";
        QMessageBox::warning(this, "No Model", "Please load a 3D model before slicing.");
        return;
    }

    QString stlPath = model3D->getSourceFilePath();
    QString modelName = QFileInfo(stlPath).completeBaseName();

    qDebug() << "[MAIN] STL Path:" << stlPath;
    qDebug() << "[MAIN] Model Name:" << modelName;

    loadingDialog->show();
    qDebug() << "[UI] Loading dialog shown";

    // ---- Start slicer process ----
    QProcess *proc = new QProcess(this);
    proc->setProcessChannelMode(QProcess::MergedChannels);

    connect(proc, &QProcess::started, this, []() {
        qDebug() << "[SLICER] Process started";
    });

    connect(proc, &QProcess::readyReadStandardOutput, this, [proc]() {
        qDebug() << "[SLICER OUT]" << proc->readAllStandardOutput();
    });

    connect(proc, &QProcess::readyReadStandardError, this, [proc]() {
        qDebug() << "[SLICER ERR]" << proc->readAllStandardError();
    });

    connect(proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        qDebug() << "[ERROR] Slicer process error:" << e;
        loadingDialog->hide();
    });

    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [=](int exitCode, QProcess::ExitStatus status) {

                qDebug() << "[SLICER] Finished. ExitCode:" << exitCode << "Status:" << status;

                loadingDialog->hide();
                qDebug() << "[UI] Loading dialog hidden";

                if (status != QProcess::NormalExit || exitCode != 0) {
                    qDebug() << "[ERROR] Slicing failed";
                    return;
                }

                qDebug() << "[MAIN] Slicing SUCCESS";

                lastGeneratedGcodePath =
                    QCoreApplication::applicationDirPath()
                    + "/slicerbundle/output_gcode/" + modelName + "_reformed.gcode";

                qDebug() << "[MAIN] GCODE PATH:" << lastGeneratedGcodePath;

                // Kill old visualizer process
                if (this->visualProcess) {
                    qDebug() << "[VIS] Killing old visualizer";
                    this->visualProcess->kill();
                    this->visualProcess->deleteLater();
                    this->visualProcess = nullptr;
                }

                // ---- Start visualizer process ----
                QProcess *visualProcess = new QProcess(this);
                visualProcess->setProcessChannelMode(QProcess::MergedChannels);
                this->visualProcess = visualProcess;

                QString visualPath =
                    QCoreApplication::applicationDirPath()
                    + "/slicerbundle/this_visualizer.exe";

                qDebug() << "[VIS] EXE PATH:" << visualPath;

                QStringList args;
                args << "--gcode" << lastGeneratedGcodePath;

                // Debug outputs
                connect(visualProcess, &QProcess::started, this, []() {
                    qDebug() << "[VIS] Process STARTED";
                });

                connect(visualProcess, &QProcess::errorOccurred, this, [](QProcess::ProcessError e) {
                    qDebug() << "[VIS ERROR]" << e;
                });

                connect(visualProcess, &QProcess::readyReadStandardOutput, this, [=]() {
                    qDebug() << "[VIS] readyReadStandardOutput fired";

                    while (visualProcess->canReadLine()) {
                        QString line = visualProcess->readLine().trimmed();
                        qDebug() << "[VIS RAW]" << line;

                        bool ok = false;
                        WId wid = line.toULongLong(&ok);

                        if (!ok) {
                            qDebug() << "[VIS] Not a WId, skipping...";
                            continue;
                        }

                        qDebug() << "[VIS] GOT WID:" << wid;

                        // ---- Delay to ensure native window is ready ----
                        QTimer::singleShot(200, this, [=]() {
                            qDebug() << "[EMBED] Attempting fromWinId...";

                            QWindow *foreignWindow = QWindow::fromWinId(wid);
                            if (!foreignWindow) {
                                qDebug() << "[EMBED ERROR] foreignWindow is NULL";
                                return;
                            }

                            qDebug() << "[EMBED] foreignWindow created";

                            // Force native window initialization
                            foreignWindow->show();
                            QCoreApplication::processEvents();
                            foreignWindow->hide();

                            qDebug() << "[EMBED] Forcing native window ready";

                            // ---- Second delay before embedding ----
                            QTimer::singleShot(200, this, [=]() {
                                qDebug() << "[EMBED] Creating container...";

                                QWidget *container = QWidget::createWindowContainer(foreignWindow, this);
                                if (!container) {
                                    qDebug() << "[EMBED ERROR] container is NULL";
                                    return;
                                }

                                qDebug() << "[EMBED] container created";

                                container->setMinimumSize(800, 600);
                                container->setFocusPolicy(Qt::StrongFocus);

                                qDebug() << "[EMBED] Clearing old widgets";
                                QLayoutItem *child;
                                while ((child = ui->previewTabLayout->takeAt(0)) != nullptr) {
                                    delete child->widget();
                                    delete child;
                                }

                                qDebug() << "[EMBED] Adding container";
                                ui->previewTabLayout->addWidget(container);

                                qDebug() << "[EMBED] Switching tab";
                                ui->tabWidget->setCurrentWidget(ui->previewTab);

                                qDebug() << "[EMBED] DONE";
                            });
                        });

                        disconnect(visualProcess, &QProcess::readyReadStandardOutput, nullptr, nullptr);
                        break;
                    }
                });

                connect(visualProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                        this, [](int code, QProcess::ExitStatus status) {
                            qDebug() << "[VIS] Process finished. Code:" << code << "Status:" << status;
                        });

                qDebug() << "[VIS] STARTING PROCESS NOW";
                visualProcess->start(visualPath, args);

                qDebug() << "================ END SETUP ================";

                proc->deleteLater();
            });

    QString slicerPath =
        QCoreApplication::applicationDirPath()
        + "/slicerbundle/slicer_pipeline.exe";

    qDebug() << "[MAIN] Slicer path:" << slicerPath;

    QString prusaPath = appConfig->getPrusaSlicerPath();
    qDebug() << "[MAIN] Prusa path:" << prusaPath;

    QStringList slicerArgs;
    slicerArgs << "--stl" << stlPath
               << "--model" << modelName
               << "--prusa" << prusaPath;

    qDebug() << "[MAIN] Starting slicer process...";
    proc->start(slicerPath, slicerArgs);
}

void MainWindow::onRotateXClicked()
{

}

void MainWindow::onRotateYClicked()
{

}

void MainWindow::onRotateZClicked()
{

}

